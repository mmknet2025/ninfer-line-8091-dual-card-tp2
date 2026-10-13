#pragma once

#include "core/arena.h"
#include "core/device.h"
#include "core/tensor.h"
#include "text/structured_output.h"

#include <array>
#include <cstddef>
#include <cstdint>
#include <exception>
#include <memory>
#include <span>
#include <algorithm>
#include <stdexcept>

#include <cuda_runtime.h>

namespace ninfer::targets::qwen3_6::detail {

// Producer of the constrained-decoding mask plane.
//
// The sampler's mask seam (`ops::SamplingConfig::token_mask` / `token_mask_stride`, consumed by
// `sampling_token_admissible` in the decode/verify kernels) has no producer on its own: something
// has to turn a request's compiled grammar into per-lane, per-draft-column bitsets and place them
// where the captured ingress records point. This class is that producer.
//
// Ownership and addresses, mirroring the 5258 line:
//   * The plane lives in the Program's PERSISTENT arena (`PersistentLayout::grammar_masks`), so the
//     device address captured into every round's pinned ingress record is stable for the life of
//     the Program. Only the CONTENTS are refreshed, once per round, on the same stream the round
//     runs on.
//   * The grammar state itself belongs to the REQUEST, never to a reusable KV checkpoint: a
//     retained lane carries no grammar, and the request-level state is what `accept()` advances.
//   * The host mirror is a pinned buffer refreshed by `text::GrammarState::fill_masks` and copied
//     up with `cudaMemcpyAsync`, so a CUDA-graph-replayed round reads the newest bits.
//
// DFlash verification drafts are produced on device inside the same graph that verifies them, so
// that leg cannot precompute its masks on the host before launch. It stages them instead: the
// drafts are copied down, a host node fills the plane while the graph is mid-flight, and the whole
// plane is copied back up -- all three enqueued into the captured stream.
class StructuredRound {
public:
    StructuredRound(Tensor masks, std::uint32_t vocab, std::uint32_t width, std::uint32_t lanes,
                    cudaStream_t stream)
        : masks_(masks),
          words_(static_cast<int>((vocab + 31U) / 32U)),
          width_(std::max(1U, width)),
          lanes_(lanes),
          host_masks_(masks.bytes()),
          host_drafts_(static_cast<std::size_t>(std::max(1U, width - 1U)) *
                       static_cast<std::size_t>(std::max(1U, lanes)) * sizeof(TokenId)) {
        if (masks.dtype != DType::I32 || masks.data == nullptr || !masks.is_contiguous() ||
            words_ <= 0 || masks.bytes() <
                               static_cast<std::size_t>(lanes_) * static_cast<std::size_t>(width_) *
                                   static_cast<std::size_t>(words_) * sizeof(std::uint32_t)) {
            throw std::invalid_argument("structured round mask plane has an invalid geometry");
        }
        // Unconstrained until a grammar fills a column: every bit set means "no restriction".
        // Both mirrors start that way, so neither an unwritten column nor the plane-wide copy-up
        // the DFlash staging step performs can ever publish a zero that no grammar asked for.
        std::fill_n(static_cast<std::uint32_t*>(host_masks_.data()), masks.bytes() / 4U,
                    ~std::uint32_t{0});
        CUDA_CHECK(cudaMemsetAsync(masks_.data, 0xFF, masks_.bytes(), stream));
    }

    StructuredRound(const StructuredRound&)            = delete;
    StructuredRound& operator=(const StructuredRound&) = delete;

    // Base of this lane's columns. `stride()` words per column, `width_` columns per lane.
    [[nodiscard]] const std::uint32_t* device_mask(std::uint32_t lane) const {
        return static_cast<const std::uint32_t*>(masks_.data) +
               static_cast<std::size_t>(lane) * static_cast<std::size_t>(width_) *
                   static_cast<std::size_t>(words_);
    }

    [[nodiscard]] int stride() const { return words_; }

    [[nodiscard]] std::uint32_t width() const { return width_; }

    [[nodiscard]] std::uint32_t lanes() const { return lanes_; }

    // Column i is conditioned on drafts[0..i). Writes (drafts.size() + 1) columns at this lane and
    // copies them into the captured plane on `stream`. May only be called when the calling thread
    // owns the grammar (the request's own decode/resolve path).
    void fill(std::uint32_t lane, const text::GrammarState& grammar,
              std::span<const TokenId> drafts, cudaStream_t stream) {
        const std::size_t columns = drafts.size() + 1U;
        if (lane >= lanes_ || columns > width_) {
            throw std::logic_error("structured round fill is outside the planned mask plane");
        }
        auto* host = static_cast<std::uint32_t*>(host_masks_.data()) +
                     static_cast<std::size_t>(lane) * static_cast<std::size_t>(width_) *
                         static_cast<std::size_t>(words_);
        grammar.fill_masks(
            {host, columns * static_cast<std::size_t>(words_)}, drafts);
        CUDA_CHECK(cudaMemcpyAsync(const_cast<std::uint32_t*>(device_mask(lane)), host,
                                   columns * static_cast<std::size_t>(words_) *
                                       sizeof(std::uint32_t),
                                   cudaMemcpyHostToDevice, stream));
    }

    // --- DFlash staging ---------------------------------------------------------------------
    // One row per decode-batch slot, naming the lane and the proposal extent whose drafts this
    // row's verify columns are conditioned on.
    void begin_dflash() {
        rows_  = {};
        error_ = nullptr;
    }

    void set_dflash_row(std::uint32_t row, std::uint32_t lane, std::uint32_t extent,
                        std::shared_ptr<text::GrammarState> grammar) {
        if (row >= rows_.size() || lane >= lanes_ || extent >= width_) {
            throw std::logic_error("structured round DFlash row is outside the planned plane");
        }
        rows_[row] = Row{lane, extent, std::move(grammar)};
    }

    // Enqueue the staging transfer pair around `cudaLaunchHostFunc`. No CUDA call or exception
    // escapes the host node, so the three nodes are captured together and verification cannot
    // observe a mask before the CPU matcher has filled it.
    void enqueue_dflash(const Tensor& drafts, cudaStream_t stream) {
        if (drafts.data == nullptr || drafts.bytes() > host_drafts_.size()) {
            throw std::logic_error("structured round DFlash drafts do not fit the staging buffer");
        }
        CUDA_CHECK(cudaMemcpyAsync(host_drafts_.data(), drafts.data, drafts.bytes(),
                                   cudaMemcpyDeviceToHost, stream));
        CUDA_CHECK(cudaLaunchHostFunc(stream, &StructuredRound::callback, this));
        CUDA_CHECK(cudaMemcpyAsync(masks_.data, host_masks_.data(), masks_.bytes(),
                                   cudaMemcpyHostToDevice, stream));
    }

    // Rethrows anything the host node caught. Called after the round's stream is synchronized.
    void check() const {
        if (error_) { std::rethrow_exception(error_); }
    }

private:
    struct Row {
        std::uint32_t lane   = 0;
        std::uint32_t extent = 0;
        std::shared_ptr<text::GrammarState> grammar;
    };

    static void CUDART_CB callback(void* opaque) noexcept {
        auto& self = *static_cast<StructuredRound*>(opaque);
        try {
            for (std::size_t row = 0; row < self.rows_.size(); ++row) {
                const Row& entry = self.rows_[row];
                if (!entry.grammar) { continue; }
                auto* mask = static_cast<std::uint32_t*>(self.host_masks_.data()) +
                             static_cast<std::size_t>(entry.lane) *
                                 static_cast<std::size_t>(self.width_) *
                                 static_cast<std::size_t>(self.words_);
                auto* row_drafts = static_cast<TokenId*>(self.host_drafts_.data()) +
                                   row * static_cast<std::size_t>(self.width_ - 1U);
                entry.grammar->fill_masks(
                    {mask, (static_cast<std::size_t>(entry.extent) + 1U) *
                               static_cast<std::size_t>(self.words_)},
                    {row_drafts, entry.extent});
            }
        } catch (...) { self.error_ = std::current_exception(); }
    }

    Tensor masks_;
    int words_ = 0;
    std::uint32_t width_ = 0;
    std::uint32_t lanes_ = 0;
    PinnedHostBuffer host_masks_;
    PinnedHostBuffer host_drafts_;

    std::array<Row, 8> rows_{};
    std::exception_ptr error_;
};

} // namespace ninfer::targets::qwen3_6::detail
