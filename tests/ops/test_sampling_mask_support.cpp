// Regression contract for the constrained-decoding mask on the multi-block sampling route.
//
// The live engine failed its first constrained request with
//
//   target returned a token outside the 248077-token domain
//
// because the partial top-k kernel encoded an inadmissible token as the workspace's *empty key*
// (`0ull`) instead of as a lowest-ranked score. Two things follow from that key: its id decodes to
// INT_MAX, and its value decodes to NaN, so a column whose candidate list is not completely full —
// which is the normal state of a constrained column, since a grammar admits a handful of the
// vocabulary — ends with a NaN weight in the truncated distribution and the draw returns INT_MAX.
//
// The contract this file pins down, on the exact production geometry:
//
//   1. a non-empty admissible set NEVER returns a token outside it (nor INT_MAX);
//   2. a single admissible token returns exactly that token;
//   3. an empty admissible set reports kSamplerNonFiniteToken, never a plausible-looking id;
//   4. a null mask is unchanged: the same geometry still samples the unconstrained support.
//
// Under the pre-fix kernel cases 1 and 3 return INT_MAX, so this test fails before the fix and
// passes after it. `ninfer_sampling_test`'s own mask_contract() covers only the greedy
// (temperature <= 0) multi-block path, which hides the defect: with cap == 1 a single admissible
// candidate outranks every empty key.

#include "ninfer/ops/sampling.h"
#include "ops/op_tester.h"

#include <algorithm>
#include <cstddef>
#include <cstdint>
#include <iostream>
#include <memory>
#include <stdexcept>
#include <vector>

using namespace ninfer;
using namespace ninfer::test;

namespace {

// The production vocabulary the serve line runs at; a constrained column admits a handful of it.
constexpr int kProductionVocab = 248077;

std::vector<std::uint32_t> allow(int vocab, std::initializer_list<int> tokens) {
    std::vector<std::uint32_t> mask(static_cast<std::size_t>((vocab + 31) / 32), 0U);
    for (int token : tokens) {
        mask[static_cast<std::size_t>(token) / 32U] |= (1U << (token % 32));
    }
    return mask;
}

struct Draw {
    int token          = 0;
    bool mask_evil     = false; // the mask buffer's guards were touched
    bool output_evil   = false;
};

// One constrained draw on the route `token_domain` selects. `mask` empty means a null mask.
Draw draw(int token_domain, const ops::SamplingConfig& base, const std::vector<std::uint32_t>& mask,
          int position) {
    constexpr int kBatch = 1;
    std::vector<float> logits(static_cast<std::size_t>(token_domain), -3.0f);
    // A deterministic, non-trivial score profile: the unmasked argmax is 1, the admissible tokens
    // carry distinct values so the draw has a real distribution to follow.
    logits[0] = 0.5f;
    logits[1] = 9.0f;
    logits[2] = -1.0f;
    const auto profile = [&logits, token_domain](int token, float value) {
        if (token < token_domain) { logits[static_cast<std::size_t>(token)] = value; }
    };
    profile(17, 4.0f);
    profile(4000, 2.0f);
    profile(2048, 3.0f);
    profile(123456, 1.0f);
    profile(248076, 0.0f);

    DeviceBuffer device_logits = to_device_bf16(logits);
    ops::SamplingConfig config = base;
    std::unique_ptr<GuardedDeviceBuffer> device_mask;
    if (!mask.empty()) {
        device_mask = std::make_unique<GuardedDeviceBuffer>(
            mask.size() * sizeof(std::uint32_t));
        device_mask->copy_from_host(mask.data(), mask.size() * sizeof(std::uint32_t));
        config.token_mask        = static_cast<const std::uint32_t*>(device_mask->data());
        config.token_mask_stride = static_cast<std::int32_t>(mask.size());
    }
    DeviceBuffer device_configs   = to_device(std::vector<ops::SamplingConfig>{config});
    DeviceBuffer device_positions = to_device(std::vector<int>{position});
    GuardedDeviceBuffer device_out(static_cast<std::size_t>(kBatch) * sizeof(std::int32_t));

    Tensor logits_tensor(device_logits.p, DType::BF16, {token_domain, kBatch});
    Tensor out_tensor(device_out.data(), DType::I32, {kBatch});
    Tensor positions_tensor(device_positions.p, DType::I32, {kBatch});
    const std::size_t workspace_bytes =
        ops::sampling_workspace_capacity_bytes(token_domain, kBatch, kBatch);
    WorkspaceArena workspace(std::max<std::size_t>(256, workspace_bytes));
    ops::sample(logits_tensor, out_tensor, token_domain,
                static_cast<const ops::SamplingConfig*>(device_configs.p), positions_tensor,
                ops::kSamplePurposeDecode, workspace, nullptr);
    cuda_synchronize();

    Draw result;
    result.token     = from_device<int>(device_out.data(), kBatch)[0];
    result.mask_evil = device_mask != nullptr && device_mask->verify_guards("sample mask") != 0;
    result.output_evil = device_out.verify_guards("sample output") != 0;
    return result;
}

bool admissible(const std::vector<std::uint32_t>& mask, int vocab, int token) {
    if (token < 0 || token >= vocab) { return false; }
    return (mask[static_cast<std::size_t>(token) / 32U] & (1U << (token % 32))) != 0U;
}

int failures = 0;

void require(bool condition, const char* what) {
    if (!condition) {
        std::cerr << "sampling mask support: " << what << '\n';
        ++failures;
    }
}

// A grammar-admitting column is never completely full, so the candidate list always carries
// entries that are not admissible. Before the fix those entries were the workspace's zero key.
void multiblock_stochastic_stays_admissible(int vocab) {
    const std::vector<std::uint32_t> mask = allow(vocab, {17, 2048, 4000});
    ops::SamplingConfig stochastic;
    stochastic.temperature = 0.8f;
    stochastic.top_k       = 20;
    stochastic.top_p       = 0.95f;
    stochastic.min_p       = 0.05f;
    stochastic.seed        = 1234ULL;

    std::vector<int> seen;
    for (int position = 0; position < 32; ++position) {
        const Draw d = draw(vocab, stochastic, mask, 700 + position);
        require(!d.mask_evil && !d.output_evil, "guarded buffers were touched");
        if (!admissible(mask, vocab, d.token)) {
            std::cerr << "sampling mask support: vocab=" << vocab << " position=" << position
                      << " drew " << d.token << " outside the admissible set\n";
            ++failures;
        }
        seen.push_back(d.token);
    }
    // The distribution is not a constant: at least one of the two higher-scoring admissible tokens
    // must appear, otherwise the support itself collapsed.
    require(std::find(seen.begin(), seen.end(), 17) != seen.end() ||
                std::find(seen.begin(), seen.end(), 2048) != seen.end(),
            "a 32-draw masked sample never reached its dominant admissible token");
}

// The support the grammar admits can be a single token; the draw must be exactly it.
void single_admissible_token_is_exact(int vocab, int token) {
    const std::vector<std::uint32_t> mask = allow(vocab, {token});
    ops::SamplingConfig stochastic;
    stochastic.temperature = 0.8f;
    stochastic.top_k       = 20;
    stochastic.top_p       = 0.95f;
    stochastic.seed        = 99ULL;
    for (int position = 0; position < 8; ++position) {
        const Draw d = draw(vocab, stochastic, mask, 40 + position);
        if (d.token != token) {
            std::cerr << "sampling mask support: vocab=" << vocab << " unique admissible token "
                      << token << " drew " << d.token << '\n';
            ++failures;
        }
    }
}

// An exhausted mask is a producer bug (the grammar frontend rejects it), but the sampler must not
// cover it up by licensing whichever id the empty key happened to encode.
void empty_admissible_set_fails_loudly(int vocab) {
    ops::SamplingConfig stochastic;
    stochastic.temperature = 0.8f;
    stochastic.top_k       = 20;
    stochastic.seed        = 5ULL;
    ops::SamplingConfig greedy;
    for (const ops::SamplingConfig& config : {stochastic, greedy}) {
        const Draw d = draw(vocab, config, allow(vocab, {}), 11);
        const std::vector<int> expected{ops::kSamplerNonFiniteToken};
        failures += verify_exact("sample empty-mask sentinel", std::vector<int>{d.token}, expected);
    }
}

// A null mask must be untouched by all of the above.
void unconstrained_is_unchanged(int vocab) {
    ops::SamplingConfig greedy;
    failures += verify_exact("sample unmasked greedy argmax",
                             std::vector<int>{draw(vocab, greedy, {}, 3).token}, std::vector<int>{1});
    ops::SamplingConfig stochastic;
    stochastic.temperature = 0.8f;
    stochastic.top_p       = 1.0f;
    stochastic.seed        = 7ULL;
    const int token        = draw(vocab, stochastic, {}, 4).token;
    require(token >= 0 && token < vocab, "unmasked stochastic draw left the vocabulary");
}

} // namespace

int main() {
    if (cuda_unavailable()) {
        std::cout << "SKIP: no usable CUDA device\n";
        return 77;
    }
    constexpr int kSmallVocab = 4096; // > kSamplerTileItems: the multi-block route owns it
    multiblock_stochastic_stays_admissible(kSmallVocab);
    single_admissible_token_is_exact(kSmallVocab, 4000);
    empty_admissible_set_fails_loudly(kSmallVocab);
    unconstrained_is_unchanged(kSmallVocab);
    multiblock_stochastic_stays_admissible(kProductionVocab);
    single_admissible_token_is_exact(kProductionVocab, 248076);
    empty_admissible_set_fails_loudly(kProductionVocab);

    std::cout << (failures == 0 ? "OK" : "FAIL") << " sampling mask support contract\n";
    return failures == 0 ? 0 : 1;
}
