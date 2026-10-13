// Producer-side proof for constrained decoding.
//
// The sampler's mask seam (`ops::SamplingConfig::token_mask`) has a consumer but no producer on
// its own. This test exercises the producer directly, with no engine, no artifact and no service:
//
//   1. a compiled grammar really does restrict the next token (the frontend's own contract, and
//      the input to the plane);
//   2. `StructuredRound::fill` writes exactly the plane a grammar state describes, at this lane's
//      Program-persistent address, with `stride()` words per column;
//   3. the plane is born unconstrained, so a lane whose request carries no grammar samples
//      exactly as it always has;
//   4. the DFlash staging path (`begin_dflash` / `set_dflash_row` / `enqueue_dflash`) really fills
//      the plane from drafts that only exist on device, via the captured host node.
//
// What it does NOT prove (and cannot, without starting the engine): that the three schedule fill
// sites and the two commit sites are reached during a live decode. Those are compile-time wired
// and inspected, not runtime-measured.

#include "core/arena.h"
#include "core/device.h"
#include "core/tensor.h"
#include "targets/qwen3_6/impl/runtime/structured_round.h"
#include "text/structured_output.h"

#include <cuda_runtime.h>

#include <cstdint>
#include <cstring>
#include <iostream>
#include <stdexcept>
#include <string>
#include <vector>

using namespace ninfer;
using namespace ninfer::text;
using ninfer::targets::qwen3_6::detail::StructuredRound;

namespace {

constexpr int kSkipCode = 77;

void require(bool value, const char* what) {
    if (!value) { throw std::runtime_error(what); }
}

bool bit(const std::vector<std::uint32_t>& words, int token) {
    return (words[static_cast<std::size_t>(token) / 32U] &
            (1U << (static_cast<unsigned>(token) % 32U))) != 0U;
}

std::size_t bit_count(const std::vector<std::uint32_t>& words, int vocab) {
    std::size_t count = 0;
    for (int token = 0; token < vocab; ++token) { count += bit(words, token) ? 1U : 0U; }
    return count;
}

// A 257-token vocabulary: 0..255 are single bytes, 256 is the EOS/stop token, exactly the shape
// the engine's own structured-output fixture uses.
StructuredCompiler make_compiler() {
    std::vector<std::string> vocab(257);
    for (int i = 0; i < 256; ++i) { vocab[i] = std::string(1, static_cast<char>(i)); }
    return StructuredCompiler(std::move(vocab), {256});
}

void run() {
    constexpr int kVocab = 257;
    constexpr int kWords = (kVocab + 31) / 32; // 9
    constexpr int kWidth = 2;                  // one draft column + one verify column
    constexpr int kLanes = 2;
    constexpr int kPlaneWords = kWords * kWidth * kLanes;

    StructuredCompiler compiler = make_compiler();
    const auto schema =
        R"({"type":"object","properties":{"a":{"type":"integer"}},"required":["a"],"additionalProperties":false})";

    // ---- 1. The grammar itself restricts the next token -------------------------------------
    auto grammar = compiler.compile({StructuredOutputKind::JsonSchema, schema});
    std::vector<std::uint32_t> plane(kWords, 0);
    grammar->fill_masks(plane, {});
    require(bit(plane, '{'), "grammar must permit the object start");
    require(!bit(plane, '['), "grammar must reject a non-object root");
    require(!bit(plane, '9'), "grammar must reject a bare digit at the root");
    require(!bit(plane, 256), "grammar must reject EOS at the root");

    // `{"a":` is committed: the legality of individual tokens must have moved with it.
    std::vector<TokenId> committed;
    for (unsigned char ch : std::string("{\"a\":")) { committed.push_back(ch); }
    auto advanced = grammar->fork();
    advanced->accept(committed);
    std::vector<std::uint32_t> shifted(kWords, 0);
    advanced->fill_masks(shifted, {});
    require(shifted != plane, "accept() did not move the grammar");
    require(!bit(shifted, '{'), "grammar still permits the object start mid-value");
    require(bit(shifted, '9') || bit(shifted, '-'), "integer value position permits no digit");
    require(!bit(plane, '9'), "the root position should not have permitted a digit");

    // A contraction, not just a change: an enum choice narrows to the tokens that remain viable.
    auto enumerated = compiler.compile(
        {StructuredOutputKind::JsonSchema,
         R"({"type":"object","properties":{"a":{"enum":["xy","xz"]}},"required":["a"],"additionalProperties":false})"});
    std::vector<TokenId> head;
    for (unsigned char ch : std::string("{\"a\":\"")) { head.push_back(ch); }
    enumerated->accept(head);
    std::vector<std::uint32_t> narrow(kWords, 0);
    enumerated->fill_masks(narrow, {});
    require(bit(narrow, 'x'), "enum prefix must be the only viable continuation");
    enumerated->accept(std::vector<TokenId>{'x'});
    enumerated->fill_masks(narrow, {});
    require(bit(narrow, 'y') && bit(narrow, 'z'), "enum branch must keep both alternatives");
    const std::size_t widened = bit_count(narrow, kVocab);
    enumerated->accept(std::vector<TokenId>{'y'});
    enumerated->fill_masks(narrow, {});
    const std::size_t narrowed = bit_count(narrow, kVocab);
    require(narrowed < widened, "accept() must contract the legal set");
    require(!bit(narrow, 'z'), "the rejected enum alternative must be gone");
    require(bit(narrow, '"'), "the surviving alternative must be committable");

    // ---- 2/3. The mask plane: geometry, default state, and fill -----------------------------
    if (cudaSetDevice(0) != cudaSuccess) {
        std::cout << "[skip] no usable CUDA device\n";
        std::exit(kSkipCode);
    }
    cudaStream_t stream = nullptr;
    require(cudaStreamCreate(&stream) == cudaSuccess, "cudaStreamCreate");

    DeviceBuffer storage(static_cast<std::size_t>(kPlaneWords) * sizeof(std::uint32_t));
    Tensor masks(storage.p, DType::I32, {kWords, kWidth, kLanes});
    StructuredRound round(masks, kVocab, kWidth, kLanes, stream);
    require(round.stride() == kWords, "stride must be the plane's word count");
    require(round.width() == kWidth && round.lanes() == kLanes, "plane geometry reported wrong");
    require(reinterpret_cast<const std::uint8_t*>(round.device_mask(1)) -
                reinterpret_cast<const std::uint8_t*>(round.device_mask(0)) ==
                static_cast<std::ptrdiff_t>(kWidth * kWords * sizeof(std::uint32_t)),
            "lane base is not lane-major at width*words");

    const auto read_back = [&](std::vector<std::uint32_t>& out) {
        out.assign(kPlaneWords, 0);
        require(cudaMemcpyAsync(out.data(), masks.data, out.size() * sizeof(std::uint32_t),
                                cudaMemcpyDeviceToHost, stream) == cudaSuccess,
                "plane read-back");
        require(cudaStreamSynchronize(stream) == cudaSuccess, "plane read-back sync");
    };

    std::vector<std::uint32_t> device_plane;
    read_back(device_plane);
    for (int word = 0; word < kPlaneWords; ++word) {
        require(device_plane[static_cast<std::size_t>(word)] == ~std::uint32_t{0},
                "an unfilled plane must be unconstrained");
    }

    // fill() must place this grammar's column zero at this lane's base, on device.
    round.fill(0, *grammar, {}, stream);
    require(cudaStreamSynchronize(stream) == cudaSuccess, "fill sync");
    read_back(device_plane);
    for (int word = 0; word < kWords; ++word) {
        require(device_plane[static_cast<std::size_t>(word)] == plane[static_cast<std::size_t>(word)],
                "fill() did not reproduce the grammar's column zero");
    }
    // Lane 1 was never filled: still unconstrained, so a lane with no grammar is untouched.
    for (int word = kWidth * kWords; word < 2 * kWidth * kWords; ++word) {
        require(device_plane[static_cast<std::size_t>(word)] == ~std::uint32_t{0},
                "fill() leaked into another lane's plane");
    }

    // The advanced state's column zero must be what the plane carries after a second fill.
    round.fill(0, *grammar, {}, stream);
    round.fill(1, *advanced, {}, stream);
    require(cudaStreamSynchronize(stream) == cudaSuccess, "advanced fill sync");
    read_back(device_plane);
    for (int word = 0; word < kWords; ++word) {
        require(device_plane[static_cast<std::size_t>(kWidth * kWords + word)] ==
                    shifted[static_cast<std::size_t>(word)],
                "lane 1 did not receive its own grammar state");
    }

    // ---- 4. DFlash staging: masks from device-only drafts -----------------------------------
    // Drafts { '{' } written on device; nothing on the host knows them until the host node runs.
    DeviceBuffer draft_storage(2 * sizeof(TokenId));
    const TokenId host_draft = '{';
    require(cudaMemcpyAsync(draft_storage.p, &host_draft, sizeof(TokenId),
                            cudaMemcpyHostToDevice, stream) == cudaSuccess,
            "draft upload");
    require(cudaStreamSynchronize(stream) == cudaSuccess, "draft upload sync");
    Tensor drafts(draft_storage.p, DType::I32, {1});

    // Two columns for lane 0: column 0 is the round's current state, column 1 the state after the
    // draft. This is the shape every verify leg consumes.
    auto brace_state = grammar->fork();
    brace_state->accept(std::vector<TokenId>{'{'});
    std::vector<std::uint32_t> after_brace(kWords, 0);
    brace_state->fill_masks(after_brace, {});
    std::vector<std::uint32_t> staged(2 * kWords, 0);
    grammar->fill_masks(staged, std::span<const TokenId>(&host_draft, 1));
    for (int word = 0; word < kWords; ++word) {
        require(staged[static_cast<std::size_t>(word)] == plane[static_cast<std::size_t>(word)],
                "drafted column 0 must be the round's current state");
        require(staged[static_cast<std::size_t>(kWords + word)] ==
                    after_brace[static_cast<std::size_t>(word)],
                "drafted column 1 must be the state after the draft");
    }
    require(after_brace != plane, "fixture does not exercise per-column conditioning");

    // Poison the plane so a no-op staging path cannot pass by accident.
    std::vector<std::uint32_t> poison(kPlaneWords, 0);
    require(cudaMemcpyAsync(masks.data, poison.data(), poison.size() * sizeof(std::uint32_t),
                            cudaMemcpyHostToDevice, stream) == cudaSuccess,
            "plane poison");
    require(cudaStreamSynchronize(stream) == cudaSuccess, "plane poison sync");

    round.begin_dflash();
    round.set_dflash_row(0, 0, 1, grammar);
    round.enqueue_dflash(drafts, stream);
    require(cudaStreamSynchronize(stream) == cudaSuccess, "dflash staging sync");
    round.check();
    read_back(device_plane);
    for (int word = 0; word < 2 * kWords; ++word) {
        require(device_plane[static_cast<std::size_t>(word)] ==
                    staged[static_cast<std::size_t>(word)],
                "staged DFlash plane does not match the drafted grammar");
    }
    // Lanes and columns the staging step did not name come from the host mirror, which is
    // unconstrained by default: the plane-wide copy-up can never publish an invented zero.
    for (int word = 0; word < kWords; ++word) {
        require(device_plane[static_cast<std::size_t>(kWidth * kWords + word)] ==
                    shifted[static_cast<std::size_t>(word)],
                "staging must preserve another lane's live mask");
        require(device_plane[static_cast<std::size_t>(kWidth * kWords + kWords + word)] ==
                    ~std::uint32_t{0},
                "staging must leave an unwritten column unconstrained");
    }

    require(cudaStreamDestroy(stream) == cudaSuccess, "cudaStreamDestroy");
    std::cout << "structured round producer ok: vocab=" << kVocab << " words=" << kWords
              << " width=" << kWidth << " lanes=" << kLanes << " enum legal=" << widened << " -> "
              << narrowed << "\n";
}

} // namespace

int main() {
    try {
        run();
        return 0;
    } catch (const std::exception& error) {
        std::cerr << "FAILED: " << error.what() << "\n";
        return 1;
    }
}
