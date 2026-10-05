// Grouped-GEMM fused MXFP4 x bf16 MoE kernel pair for MI300A (gfx942/CDNA3).
//
// Mirrors the two-kernel Triton production pipeline
// (fused_moe_kernel_gptq_awq's use_mxfp4_w4a16 branch, invoked twice by
// sglang/srt/layers/moe/moe_runner/triton_utils/fused_moe.py's
// _fused_moe_kernel_sequence) with the same grouped-GEMM contract:
// moe_align_block_size's sorted_token_ids/expert_ids/num_tokens_post_padded
// scheme, BLOCK_M=16 (matching the MFMA tile), top_k-strided A-row gather,
// and the flat (token, top_k-slot) row indexing that lets the final
// reduction be a plain view+sum. See ../../../../../../../scratchpad/loop/
// moe-fused-grouped-design.md for the full index-math derivation this file
// implements, and scratchpad/loop/moe-fused-tile/hip/tile_gemm.cu (copied
// into produce_b_fragments_32k / gemm16x16_fp4_accumulate below, verbatim
// except for parameterizing the row/column base pointers) for the MFMA
// operand-layout facts this kernel depends on -- that file's header cites
// the exact CK source lines the per-lane (blk, rc) -> (A row, B col, C
// row/col) mapping was independently re-derived from, and its results.md
// records the correctness check against a torch reference (rel L2 ~3e-7)
// that validates this mapping. This file does not re-derive that mapping;
// it reuses it unmodified.
//
// ---------------------------------------------------------------------
// Inner loop: K-templated (ported from tile_gemm_v3.cu), plus scaffold
// ---------------------------------------------------------------------
// Two inner loops coexist:
//
//  - The scaffold loop (gemm16x16_fp4_accumulate, unchanged since v1):
//    single output column-tile per call, K and num_k_waves are *runtime*
//    kernel arguments, so it keeps handling any K with K%128==0 at call
//    time. Used as the fallback for any (hidden_size, intermediate_size)
//    this model's two known shapes don't cover.
//  - The templated loop (gemm_wide_tpl, new): K, NWAVES, WIDE, and UDEPTH
//    are template parameters, ported from the concurrent optimization
//    campaign's scratchpad/loop/moe-fused-tile/hip/tile_gemm_v3.cu. Making
//    K/NWAVES compile-time makes NLOADS (the K-loop trip count) a
//    compile-time constant, so the loop fully unrolls and the
//    load-prefetch ring (UDEPTH slots) does not spill to scratch memory
//    the way the v2-derived GEMM_WIDE=2 path's runtime-NLOADS attempt at
//    prefetch did (see tile_gemm_v3.cu's header and
//    scratchpad/loop/moe-fused-tile/results-v3/results.md). WIDE output
//    column-tiles share one A-fragment load per K-step; BATCH_SCALE
//    (always on here) preloads the e8m0 scale bytes for all NLOADS steps
//    up front.
//
// This grouped kernel only instantiates gemm_wide_tpl at the shapes this
// model actually uses (see moe_gemm_w4a16_stage{1,2}'s host dispatch
// below): stage2/down at K=256 (intermediate_size_per_partition),
// NWAVES=2, WIDE=4, UDEPTH=1 (results-v3's best down config, used
// unconditionally -- stage2 has no per-M regression), and stage1/gate_up
// at K=4096 (hidden_size) with TWO templated instantiations compiled in
// (see "Stage1 per-launch dispatch" below, which picks per launch among
// these and the scaffold kernel): a "big" one (NWAVES=8, WIDE=4, UDEPTH=2,
// results-v3's best gate_up config) and a "small" one (NWAVES=4, WIDE=2,
// UDEPTH=1, half the launch). Any other K falls back to the runtime
// scaffold kernel, keeping this file's original K%128==0 generality for
// shapes outside this model.
//
// Stage1's WIDE is not WIDE independent output column-tiles of the same
// GEMM the way stage2's is: at WIDE=4 it is [gate_subtile0, gate_subtile1,
// up_subtile0, up_subtile1] (WIDE=2: [gate_subtile0, up_subtile0]) --
// adjacent 16-column gate tile(s) and the paired up tile(s) at the same
// output position, all sharing ONE A-fragment read per K-step. This
// targets the register-pressure regression an earlier attempt hit (see
// git history: applying GEMM_WIDE=2 independently to a separate gate pass
// and a separate up pass kept 4 live f32x4 accumulators *without* cutting
// the number of full-K A reads from 2 to 1, since each pass still walked K
// on its own -- that regressed stage1 8-31% despite GEMM_WIDE=2 helping
// stage2). Interleaving gate and up into one pass keeps the same
// accumulator-per-column-tile-pair footprint but halves the A-read count
// (one shared K-walk instead of two).
//
// ---------------------------------------------------------------------
// Stage1 per-launch dispatch: big templated vs. small templated vs.
// scaffold, by sorted-block count
// ---------------------------------------------------------------------
// GPU measurement (job 632489, real layer-30 weights) found the "big"
// gate_up instantiation above regresses 36% vs. the scaffold loop at
// decode-sized launches (M=16, top_k=10: ~138 sorted 16-row blocks --
// 0.404ms vs. 0.296ms). A follow-up sweep (job 632904, this change) at M
// in {16, 32, 48, 64} (160-521 sorted blocks) found the scaffold loop
// *still* fastest through all four points, not just M=16 -- the naive
// assumption that "big" would already win by M=64 did not hold; a
// separate, earlier measurement (see git history) established "big" wins
// by 13% at M=1024's block count (thousands), so the actual crossover is
// somewhere in the unmeasured 521-to-thousands range. moe_gemm_w4a16_
// stage1's host dispatch below picks among the stage1 kernels using
// `expert_ids.size(0)` (the sorted 16-row block count / kernel's grid.x):
// this is a host-side integer, known from the already-computed
// `moe_align_block_size` output before this launch, so the choice is a
// plain host branch made once per call and is CUDA-graph safe -- each
// captured decode batch size gets its own graph, and the branch outcome
// for a given block count is the same on every replay. See PR description
// for the measured M in {16, 32, 48, 64, 256, 1024} table and
// STAGE1_SCAFFOLD_BLOCK_THRESHOLD's comment for how the threshold was
// picked from it.
//
// MTILES (tile_gemm_v3.cu's M=32 axis, processing two 16-row tiles per
// block to share one decoded B-fragment across both) is NOT ported here.
// Doing so in this grouped kernel would need two consecutive 16-row
// sorted-token blocks to belong to the same expert (or padding
// moe_align_block_size's BLOCK_M to 32) -- a real change to the
// block/grid contract, not a drop-in template axis, and one that needs
// its own correctness argument and measurement. Left as follow-up.
//
// ---------------------------------------------------------------------
// Known v1 limitations (see the design note's "open questions" section)
// ---------------------------------------------------------------------
//  - K (both stage1's hidden_size and stage2's intermediate_size) must be a
//    multiple of 128: the scaffold's 4-wave/4-blk K-partition (Kwave=K/4,
//    WBLOCK=Kwave/4, NLOADS=WBLOCK/32 must be a positive integer) needs K
//    a multiple of 512 for the full 4-way wave split; compute_num_k_waves()
//    falls back to a 2-way or 1-way wave split (K a multiple of 256 or 128
//    respectively) otherwise, at the cost of wasting 1/2 or 3/4 of each
//    block's waves on redundant, unsummed duplicate work (see
//    compute_num_k_waves's and gemm16x16_fp4_accumulate's comments). This
//    only affects the scaffold fallback path -- the templated path's two
//    instantiations use a NWAVES that exactly divides their K with no
//    redundant-wave waste (K=4096/NWAVES=8 and K=256/NWAVES=2 both give an
//    exact Kwave%128==0 split).
//  - No expert-parallel (EP) support beyond honoring an already-computed
//    expert_ids[-1] sentinel by zero-writing that block's outputs; this
//    kernel does not implement moe_align_block_size's ignore_invalid_expert
//    path itself.
//  - apply_router_weight_on_input and activations other than silu are not
//    implemented (checked in the Python wrapper, not here).

#include <ATen/cuda/CUDAContext.h>
#include <c10/cuda/CUDAException.h>
#include <c10/util/BFloat16.h>
#include <torch/extension.h>

#include <cmath>
#include <cstdint>
#include <cstdlib>
#include <cstring>

namespace {

using bf16x4 = unsigned short __attribute__((ext_vector_type(4)));
using f32x4 = float __attribute__((ext_vector_type(4)));

__constant__ float MXFP4_LUT[16] = {
    0.0f, 0.5f, 1.0f, 1.5f, 2.0f, 3.0f, 4.0f, 6.0f,
    -0.0f, -0.5f, -1.0f, -1.5f, -2.0f, -3.0f, -4.0f, -6.0f,
};
constexpr int E8M0_BIAS = 127;
constexpr int BLOCK_M = 16;

__device__ __forceinline__ unsigned short to_bf16_bits(float v) { return c10::BFloat16(v).x; }

__device__ __forceinline__ f32x4 mfma_1k(bf16x4 a, bf16x4 b, f32x4 c) {
  return __builtin_amdgcn_mfma_f32_16x16x16bf16_1k(a, b, c, 0, 0, 0);
}

// decode16: unpack 16 bytes (32 packed e2m1 nibbles) plus one shared e8m0
// block exponent into 32 bf16-bit values.
//
// Three compile-time paths, all bit-identical by construction (same
// `ldexpf` + `to_bf16_bits` inputs feeding every element, no new
// rounding/edge-case logic anywhere):
//
// MXFP4_DECODE_LEGACY (v1): per element, `MXFP4_LUT[nibble]` against a
// `__constant__` array, `ldexpf` for the block scale, general IEEE
// round-to-nearest-even fp32->bf16 cast. Per scratchpad/loop/moe-isa/
// results.md (ISA audit, job 634070): despite `__constant__`, the
// compiler cannot prove the per-lane nibble index is wave-uniform, so
// this compiles to one real `global_load_dword` plus one `s_nop` PER
// WEIGHT ELEMENT. Measured 8.81 VALU/elem (stage1: 6.31), ~1.17-1.19
// loads/elem.
//
// MXFP4_DECODE_SELECT (iteration 1, job 634076): eliminates the memory
// gather by building a 16-entry (one per e2m1 nibble) bf16-bits table
// ONCE per call via a compile-time-constant table index `n`, then
// selecting per element with a plain C array index `tbl[byte & 0xF]`.
// Loads/elem dropped 6-7x as designed, but the compiler lowered the
// runtime-indexed 16-entry register array as a linear compare/select
// expansion rather than GPR-indexing or `v_perm_b32`: VALU/elem roughly
// TRIPLED instead (stage1 6.31->17.6, stage2 8.27->19.6), VGPRs jumped
// (82->113 / 78->119), stage1 gained a 22-instruction SGPR spill, and
// occupancy dropped a wave at both kernels. Net effect: every M measured
// 22-54% SLOWER than legacy. Kept for A/B/C ISA and perf comparison only
// -- not the default, not recommended for use.
//
// Default (variant P, this iteration): same building-block functions as
// above (`ldexpf` + `to_bf16_bits`), but the once-per-call table is only
// 8 entries (one per *magnitude* index k = nibble & 7 -- e2m1's sign bit
// is nibble bit 3, magnitude is bits 2:0, magnitudes {0,0.5,1,1.5,2,3,4,6}
// per MXFP4_LUT[0..7]) and the per-element selection is done with
// `__builtin_amdgcn_perm` (`v_perm_b32`) byte-permutes instead of a
// compiler-synthesized compare/select chain. This relies on two
// mathematical facts, both empirically verified in
// test_decode_exhaustive.hip before being relied on here (not merely
// assumed): (1) sign-collapse -- both `ldexpf` and round-to-nearest-even
// are odd-symmetric in sign for every finite/infinite input in this
// domain (no NaN is ever produced), so
// `to_bf16_bits(ldexpf(-MXFP4_LUT[k], e)) == to_bf16_bits(ldexpf(MXFP4_LUT[k], e)) | 0x8000`
// for every (block_exp, k) pair -- so only the 8 positive magnitudes
// need a table entry; sign is reattached separately. (2) the magnitude
// table's high byte (bits 15:8 of the bf16 pattern) never has its top
// bit (bit 15, the sign bit) set for any of the 8 positive magnitudes at
// any block_exp (0..8x2, and every ldexpf overflow saturates to +inf,
// whose bf16 high byte is 0x7F, still MSB-clear) -- so the real sign bit
// can be OR'd into the *unsigned* high-byte plane before the interleave
// permute, instead of needing a separate post-interleave bit-scatter.
//
// Once per call: build the 8 magnitude bf16-bit words via the exact same
// `to_bf16_bits(ldexpf(MXFP4_LUT[k], block_exp))` used by both other
// paths (compile-time-constant `k`, folds to an immediate), split each
// into its low/high byte, and pack the low bytes of k=0..3 into `lo0`,
// k=4..7 into `lo1` (analogously `hi0`/`hi1` for high bytes).
//
// Per raw 32-bit word `w` of 4 packed nibble-bytes (= 4 of the 16 input
// bytes = 8 elements), two passes (low nibbles of the 4 bytes, then high
// nibbles -- `(w >> 4) & mask` isolates each byte's high nibble despite
// crossing byte boundaries on the shift, because the contaminating bits
// from the neighboring byte land above bit 2/bit 3 of each byte lane and
// are masked away):
//   sel      = w_or_wshr4 & 0x07070707        (per-lane magnitude index)
//   sign_nib = w_or_wshr4 & 0x08080808         (per-lane sign bit, at bit 3)
//   lo_bytes = perm(lo1, lo0, sel)             (low byte of each element's bf16 word)
//   hi_bytes = perm(hi1, hi0, sel)             (high byte, sign bit still 0)
//   signed_hi = hi_bytes | (sign_nib << 4)     (move bit 3 -> bit 7 = the sign bit)
//   dwordAB  = perm(signed_hi, lo_bytes, 0x05010400) | -- elements 0,1: bytes [lo0,hi0,lo1,hi1]
//   dwordCD  = perm(signed_hi, lo_bytes, 0x07030602) | -- elements 2,3: bytes [lo2,hi2,lo3,hi3]
// `dwordAB`/`dwordCD` are the finished bf16x2 pairs for that pass's 4
// elements, written directly into `vals[]` at the same indices the other
// two paths use (`vals[2b]` = low nibble of byte b, `vals[2b+1]` = high
// nibble), so callers are byte-for-byte unchanged. Cost: ~2.1 VALU/elem
// (2 `v_perm_b32` per element for lookup + 2 more per 4 elements for the
// interleave, plus the AND/OR/SHIFT bookkeeping), zero loads, table-build
// amortized over 32 elements/call (~1 op/elem, 8 ldexpf + 8 conversions +
// ~8 pack ops instead of the 16-entry version's ~16+16+~12).
__device__ __forceinline__ void decode16(const uint8_t bytes[16], int block_exp, unsigned short vals[32]) {
#if defined(MXFP4_DECODE_LEGACY)
#pragma unroll
  for (int b = 0; b < 16; ++b) {
    const uint8_t byte = bytes[b];
    const float lo = ldexpf(MXFP4_LUT[byte & 0xF], block_exp);
    const float hi = ldexpf(MXFP4_LUT[(byte >> 4) & 0xF], block_exp);
    vals[2 * b] = to_bf16_bits(lo);
    vals[2 * b + 1] = to_bf16_bits(hi);
  }
#elif defined(MXFP4_DECODE_SELECT)
  unsigned short tbl[16];
#pragma unroll
  for (int n = 0; n < 16; ++n) {
    tbl[n] = to_bf16_bits(ldexpf(MXFP4_LUT[n], block_exp));
  }
#pragma unroll
  for (int b = 0; b < 16; ++b) {
    const uint8_t byte = bytes[b];
    vals[2 * b] = tbl[byte & 0xF];
    vals[2 * b + 1] = tbl[(byte >> 4) & 0xF];
  }
#else
  // Variant P: register-only, __builtin_amdgcn_perm-based lookup. See
  // the block comment above for the full derivation.
  unsigned int lo_byte_tbl[8];
  unsigned int hi_byte_tbl[8];
#pragma unroll
  for (int k = 0; k < 8; ++k) {
    const unsigned short m = to_bf16_bits(ldexpf(MXFP4_LUT[k], block_exp));
    lo_byte_tbl[k] = m & 0xFFu;
    hi_byte_tbl[k] = (m >> 8) & 0xFFu;
  }
  const unsigned int lo0 = lo_byte_tbl[0] | (lo_byte_tbl[1] << 8) | (lo_byte_tbl[2] << 16) | (lo_byte_tbl[3] << 24);
  const unsigned int lo1 = lo_byte_tbl[4] | (lo_byte_tbl[5] << 8) | (lo_byte_tbl[6] << 16) | (lo_byte_tbl[7] << 24);
  const unsigned int hi0 = hi_byte_tbl[0] | (hi_byte_tbl[1] << 8) | (hi_byte_tbl[2] << 16) | (hi_byte_tbl[3] << 24);
  const unsigned int hi1 = hi_byte_tbl[4] | (hi_byte_tbl[5] << 8) | (hi_byte_tbl[6] << 16) | (hi_byte_tbl[7] << 24);

  constexpr unsigned int SEL_AB = 0x05010400u;  // out bytes [lo.b0, hi.b0, lo.b1, hi.b1]
  constexpr unsigned int SEL_CD = 0x07030602u;  // out bytes [lo.b2, hi.b2, lo.b3, hi.b3]

#pragma unroll
  for (int g = 0; g < 4; ++g) {
    unsigned int w;
    memcpy(&w, bytes + 4 * g, sizeof(w));
#pragma unroll
    for (int pass = 0; pass < 2; ++pass) {
      const unsigned int wv = pass == 0 ? w : (w >> 4);
      const unsigned int sel = wv & 0x07070707u;
      const unsigned int sign_nib = wv & 0x08080808u;
      const unsigned int lo_bytes = __builtin_amdgcn_perm(lo1, lo0, sel);
      const unsigned int hi_bytes = __builtin_amdgcn_perm(hi1, hi0, sel);
      const unsigned int signed_hi = hi_bytes | (sign_nib << 4);
      const unsigned int dwordAB = __builtin_amdgcn_perm(signed_hi, lo_bytes, SEL_AB);
      const unsigned int dwordCD = __builtin_amdgcn_perm(signed_hi, lo_bytes, SEL_CD);
      unsigned short e[4];
      memcpy(&e[0], &dwordAB, sizeof(dwordAB));
      memcpy(&e[2], &dwordCD, sizeof(dwordCD));
#pragma unroll
      for (int i = 0; i < 4; ++i) {
        const int b_idx = 4 * g + i;
        vals[2 * b_idx + pass] = e[i];
      }
    }
  }
#endif
}

// Dword-wide weight load: both call sites below already issue a single
// `global_load_dwordx4` for the 16-byte weight block (via
// `*reinterpret_cast<const uint4*>(...)`), but decode16 above takes a
// `uint8_t bytes[16]`, so the compiler was routing that uint4 through a
// `memcpy(bytes, &raw, 16)` round trip and then re-grouping the bytes
// back into dwords inside decode16's own body -- which gave it license
// to split the load itself at byte granularity: the compiled scaffold
// loop showed eight `global_load_ushort` (0.41 loads/elem) instead of the
// one wide load already available (0.19 loads/elem). This function is
// the same variant-P decode math (identical `ldexpf`/`to_bf16_bits`
// inputs, identical `__builtin_amdgcn_perm` sequence, same `vals[32]`
// output layout) but takes the four raw dwords directly, no byte-array
// intermediate -- a load-shape-only change, bit-exact by construction.
__device__ __forceinline__ void decode16_from_dwords(unsigned int w0, unsigned int w1, unsigned int w2,
                                                       unsigned int w3, int block_exp, unsigned short vals[32]) {
  unsigned int lo_byte_tbl[8];
  unsigned int hi_byte_tbl[8];
#pragma unroll
  for (int k = 0; k < 8; ++k) {
    const unsigned short m = to_bf16_bits(ldexpf(MXFP4_LUT[k], block_exp));
    lo_byte_tbl[k] = m & 0xFFu;
    hi_byte_tbl[k] = (m >> 8) & 0xFFu;
  }
  const unsigned int lo0 = lo_byte_tbl[0] | (lo_byte_tbl[1] << 8) | (lo_byte_tbl[2] << 16) | (lo_byte_tbl[3] << 24);
  const unsigned int lo1 = lo_byte_tbl[4] | (lo_byte_tbl[5] << 8) | (lo_byte_tbl[6] << 16) | (lo_byte_tbl[7] << 24);
  const unsigned int hi0 = hi_byte_tbl[0] | (hi_byte_tbl[1] << 8) | (hi_byte_tbl[2] << 16) | (hi_byte_tbl[3] << 24);
  const unsigned int hi1 = hi_byte_tbl[4] | (hi_byte_tbl[5] << 8) | (hi_byte_tbl[6] << 16) | (hi_byte_tbl[7] << 24);

  constexpr unsigned int SEL_AB = 0x05010400u;
  constexpr unsigned int SEL_CD = 0x07030602u;

  const unsigned int ws[4] = {w0, w1, w2, w3};
#pragma unroll
  for (int g = 0; g < 4; ++g) {
    const unsigned int w = ws[g];
#pragma unroll
    for (int pass = 0; pass < 2; ++pass) {
      const unsigned int wv = pass == 0 ? w : (w >> 4);
      const unsigned int sel = wv & 0x07070707u;
      const unsigned int sign_nib = wv & 0x08080808u;
      const unsigned int lo_bytes = __builtin_amdgcn_perm(lo1, lo0, sel);
      const unsigned int hi_bytes = __builtin_amdgcn_perm(hi1, hi0, sel);
      const unsigned int signed_hi = hi_bytes | (sign_nib << 4);
      const unsigned int dwordAB = __builtin_amdgcn_perm(signed_hi, lo_bytes, SEL_AB);
      const unsigned int dwordCD = __builtin_amdgcn_perm(signed_hi, lo_bytes, SEL_CD);
      unsigned short e[4];
      memcpy(&e[0], &dwordAB, sizeof(dwordAB));
      memcpy(&e[2], &dwordCD, sizeof(dwordCD));
#pragma unroll
      for (int i = 0; i < 4; ++i) {
        const int b_idx = 4 * g + i;
        vals[2 * b_idx + pass] = e[i];
      }
    }
  }
}

// ---------------------------------------------------------------------
// Scaffold inner loop (runtime K / num_k_waves) -- unchanged from v1
// apart from the dword-wide load (see decode16_from_dwords above).
// ---------------------------------------------------------------------
__device__ __forceinline__ void produce_b_fragments_32k(const uint8_t* __restrict__ w_fp4_col,
                                                          const uint8_t* __restrict__ w_scale_col,
                                                          bf16x4 out_frags[8]) {
  const uint4 raw = *reinterpret_cast<const uint4*>(w_fp4_col);
  const int block_exp = static_cast<int>(*w_scale_col) - E8M0_BIAS;
  unsigned short vals[32];
  decode16_from_dwords(raw.x, raw.y, raw.z, raw.w, block_exp, vals);
#pragma unroll
  for (int i = 0; i < 8; ++i) {
    out_frags[i] = bf16x4{vals[4 * i], vals[4 * i + 1], vals[4 * i + 2], vals[4 * i + 3]};
  }
}

// One wave's contribution to a single 16x16 output tile. `num_k_waves` (1,
// 2, or 4; only waves < num_k_waves do distinct work -- see
// compute_num_k_waves) replaces a hardcoded 4-way wave split: Kwave = K /
// num_k_waves, wave_k0 = (wave % num_k_waves) * Kwave. blk's role (the
// inner 4-way split of Kwave into WBLOCK) is untouched. Waves >=
// num_k_waves alias (via wave % num_k_waves) onto an earlier wave's
// K-range and compute a redundant duplicate; the caller's reduce_waves
// only sums the first num_k_waves LDS slots, so the duplicates are
// computed but never read -- correct, merely wasteful.
__device__ __forceinline__ f32x4 gemm16x16_fp4_accumulate(const c10::BFloat16* __restrict__ a_row_ptr,
                                                            const uint8_t* __restrict__ w_fp4_col_base,
                                                            const uint8_t* __restrict__ w_scale_col_base, int K,
                                                            int num_k_waves, int wave, int blk) {
  const int Kwave = K / num_k_waves;
  const int WBLOCK = Kwave >> 2;
  const int NLOADS = WBLOCK >> 5;
  const int wave_k0 = (wave % num_k_waves) * Kwave;
  const int blk_k0 = wave_k0 + blk * WBLOCK;

  f32x4 c = {0.f, 0.f, 0.f, 0.f};
  for (int j = 0; j < NLOADS; ++j) {
    const int k_load0 = blk_k0 + j * 32;
    bf16x4 b_frags[8];
    produce_b_fragments_32k(w_fp4_col_base + (k_load0 >> 1), w_scale_col_base + (k_load0 >> 5), b_frags);
#pragma unroll
    for (int i = 0; i < 8; ++i) {
      const int k_abs = k_load0 + i * 4;
      const bf16x4 a_frag =
          *reinterpret_cast<const bf16x4*>(reinterpret_cast<const unsigned short*>(a_row_ptr) + k_abs);
      c = mfma_1k(a_frag, b_frags[i], c);
    }
  }
  return c;
}

// Largest wave-count in {4, 2, 1} for which K / num_k_waves is a multiple
// of 128 (so WBLOCK = (K/num_k_waves)/4 is a multiple of 32, the MX scale
// block size produce_b_fragments_32k requires). Only used by the scaffold
// fallback path -- the templated path's NWAVES is fixed per instantiation.
__host__ __device__ __forceinline__ int compute_num_k_waves(int K) {
  if (K % (4 * 128) == 0) return 4;
  if (K % (2 * 128) == 0) return 2;
  return 1;
}

// Reduces one wave's f32x4 accumulator (this thread's (blk*4+r, rc) piece)
// across the active waves (see compute_num_k_waves) via LDS and returns
// this thread's own (tid>>4, tid&15) cell of the fully-summed 16x16 tile.
// Only used by the scaffold kernels below, which always launch 256
// threads (4 waves) regardless of num_k_waves<=4. Safe to call twice in
// the same kernel invocation (e.g. once for gate, once for up): the
// trailing __syncthreads() guards the second call's writes against lanes
// that have not yet finished reading the first call's results.
__device__ __forceinline__ float reduce_waves(float c0, float c1, float c2, float c3, int num_k_waves, int wave,
                                                int blk, int rc) {
  __shared__ float lds[4][16][16];
  lds[wave][blk * 4 + 0][rc] = c0;
  lds[wave][blk * 4 + 1][rc] = c1;
  lds[wave][blk * 4 + 2][rc] = c2;
  lds[wave][blk * 4 + 3][rc] = c3;
  __syncthreads();
  const int tid = threadIdx.x;
  const int m = tid >> 4;
  const int n = tid & 15;
  float sum = 0.f;
#pragma unroll
  for (int w = 0; w < num_k_waves; ++w) sum += lds[w][m][n];
  __syncthreads();
  return sum;
}

// ---------------------------------------------------------------------
// Templated inner loop (ported from tile_gemm_v3.cu) -- see file header.
// ---------------------------------------------------------------------
// Computes WIDE output column-tiles' f32x4 accumulators for one
// (wave, blk) slot, sharing one A-fragment load per K-step across all
// WIDE tiles. `w_fp4_cols[t]`/`w_scale_cols[t]` must already be offset to
// column t's fp4-byte-0/scale-byte-0 base (K-offset 0); this function adds
// the (blk_k0, j) K-offset itself, mirroring tile_gemm_v3.cu's
// `kernel_v3` body (minus the MTILES and A_LDS axes, not ported here -- see
// file header).
template <int K, int NWAVES, int WIDE, int UDEPTH>
__device__ __forceinline__ void gemm_wide_tpl(const c10::BFloat16* __restrict__ a_row_ptr,
                                                const uint8_t* const* __restrict__ w_fp4_cols,
                                                const uint8_t* const* __restrict__ w_scale_cols, int wave, int blk,
                                                f32x4 (&c)[WIDE]) {
  constexpr int Kwave = K / NWAVES;
  constexpr int WBLOCK = Kwave / 4;
  constexpr int NLOADS = WBLOCK / 32;
  static_assert(Kwave * NWAVES == K, "K must be an exact multiple of NWAVES for the templated path");
  static_assert(WBLOCK * 4 == Kwave, "Kwave must be a multiple of 4 (blk split)");
  static_assert(NLOADS >= 1, "K/NWAVES/128 must be >= 1 -- shape too small for this NWAVES");
  // Ring depth clamped to NLOADS: UDEPTH>=NLOADS just prefetches
  // everything up front instead of leaving unused ring slots.
  constexpr int UD = (UDEPTH < NLOADS) ? UDEPTH : NLOADS;
  const int wave_k0 = wave * Kwave;
  const int blk_k0 = wave_k0 + blk * WBLOCK;

#pragma unroll
  for (int t = 0; t < WIDE; ++t) c[t] = {0.f, 0.f, 0.f, 0.f};

  // Batched scale (tile_gemm_v3.cu's BATCH_SCALE, always on here): NLOADS
  // is a template constant, so this is exactly-sized.
  uint8_t scale_cache[WIDE][NLOADS];
#pragma unroll
  for (int t = 0; t < WIDE; ++t) {
#pragma unroll
    for (int j = 0; j < NLOADS; ++j) scale_cache[t][j] = w_scale_cols[t][(blk_k0 >> 5) + j];
  }

  // Load-prefetch ring, fixed per tile_gemm_v3.cu: with K and NWAVES
  // template parameters, `cur`/`next` below are compile-time constants at
  // every unrolled j, so this does not spill to scratch memory the way an
  // equivalent runtime-NLOADS ring did (see file header).
  uint4 raw_buf[WIDE][UD];
#pragma unroll
  for (int p = 0; p < UD - 1; ++p) {
    const int k_load0 = blk_k0 + p * 32;
#pragma unroll
    for (int t = 0; t < WIDE; ++t) {
      raw_buf[t][p] = *reinterpret_cast<const uint4*>(w_fp4_cols[t] + (k_load0 >> 1));
    }
  }

#pragma unroll
  for (int j = 0; j < NLOADS; ++j) {
    const int cur = j % UD;
    if (j + UD - 1 < NLOADS) {
      const int next = (j + UD - 1) % UD;
      const int k_loadn = blk_k0 + (j + UD - 1) * 32;
#pragma unroll
      for (int t = 0; t < WIDE; ++t) {
        raw_buf[t][next] = *reinterpret_cast<const uint4*>(w_fp4_cols[t] + (k_loadn >> 1));
      }
    }

    const int k_load0 = blk_k0 + j * 32;
    unsigned short vals[WIDE][32];
#pragma unroll
    for (int t = 0; t < WIDE; ++t) {
      const int block_exp = static_cast<int>(scale_cache[t][j]) - E8M0_BIAS;
      decode16_from_dwords(raw_buf[t][cur].x, raw_buf[t][cur].y, raw_buf[t][cur].z, raw_buf[t][cur].w, block_exp,
                            vals[t]);
    }

#pragma unroll
    for (int i = 0; i < 8; ++i) {
      const int k_abs = k_load0 + i * 4;
      const bf16x4 a_frag =
          *reinterpret_cast<const bf16x4*>(reinterpret_cast<const unsigned short*>(a_row_ptr) + k_abs);
#pragma unroll
      for (int t = 0; t < WIDE; ++t) {
        const bf16x4 b_frag = {vals[t][4 * i], vals[t][4 * i + 1], vals[t][4 * i + 2], vals[t][4 * i + 3]};
        c[t] = mfma_1k(a_frag, b_frag, c[t]);
      }
    }
  }
}

// Writes one wave's WIDE partial f32x4 accumulators into `lds` at this
// (wave, blk, rc) slot's 4 rows. Caller must __syncthreads() before
// reading `lds` back.
template <int NWAVES, int WIDE>
__device__ __forceinline__ void store_wave_partials(const f32x4 (&c)[WIDE], float lds[NWAVES][WIDE][16][16],
                                                      int wave, int blk, int rc) {
#pragma unroll
  for (int t = 0; t < WIDE; ++t) {
    lds[wave][t][blk * 4 + 0][rc] = c[t][0];
    lds[wave][t][blk * 4 + 1][rc] = c[t][1];
    lds[wave][t][blk * 4 + 2][rc] = c[t][2];
    lds[wave][t][blk * 4 + 3][rc] = c[t][3];
  }
}

// ---------------------------------------------------------------------
// Stage 1 production shapes/config.
// ---------------------------------------------------------------------
constexpr int STAGE1_TPL_K = 4096;  // hidden_size

// "big": results-v3's best gate_up config -- wins at prefill-sized (many
// sorted-block) launches.
constexpr int STAGE1_BIG_NWAVES = 8;
constexpr int STAGE1_BIG_WIDE = 4;  // [gate0, gate1, up0, up1], one shared A read
constexpr int STAGE1_BIG_UDEPTH = 2;
constexpr int STAGE1_BIG_SUBTILES = STAGE1_BIG_WIDE / 2;                // 2 (gate/up column tiles each)
constexpr int STAGE1_BIG_SUBTILE_COLS = BLOCK_M * STAGE1_BIG_SUBTILES;  // 32 inter_size columns/block

// "small": half the launch (fewer threads/block, fewer live accumulators)
// -- wins at decode-sized (few sorted-block) launches, where the big
// instantiation's fixed per-block overhead is not amortized by enough
// K-loop work. See "Stage1 per-launch dispatch" above.
constexpr int STAGE1_SMALL_NWAVES = 4;
constexpr int STAGE1_SMALL_WIDE = 2;  // [gate0, up0], one shared A read
constexpr int STAGE1_SMALL_UDEPTH = 1;
constexpr int STAGE1_SMALL_SUBTILES = STAGE1_SMALL_WIDE / 2;                // 1
constexpr int STAGE1_SMALL_SUBTILE_COLS = BLOCK_M * STAGE1_SMALL_SUBTILES;  // 16 inter_size columns/block

// Sorted 16-row block count (expert_ids.size(0)) below which the scaffold
// kernel beats both templated instantiations. GPU measurement (M in {16,
// 32, 48, 64}, real layer-30 weights, top_k=10) found the scaffold loop
// fastest at every measured point (160-521 blocks), "small" a consistent
// second (beats "big" by ~20-25% there but never catches scaffold), and
// "big" fastest at M=1024's block count (thousands; -13.2% vs. scaffold,
// a fact already established by an earlier session's measurement -- see
// git history). No measurement was taken between 521 and that thousands-
// scale range, so this threshold (1024) is chosen to keep every measured
// "scaffold wins" point on the scaffold side while staying well clear of
// the known "big wins" point -- not itself a measured crossover. See PR
// description for the full table. "small" is not used by the shipped
// auto rule (scaffold dominates it in the only range measured) but is
// kept as a compiled instantiation, selectable via
// SGLANG_MXFP4_FUSED_FORCE_SMALL_STAGE1, in case future measurement at
// intermediate block counts finds it useful there.
// With the dword-wide loads above, the templated kernel wins from 160 blocks
// up; the measured crossover this file used to describe no longer exists in range.
constexpr int STAGE1_SCAFFOLD_BLOCK_THRESHOLD = 160;

// ---------------------------------------------------------------------
// Stage 1 (scaffold): gate_up GEMM + silu(gate)*up epilogue.
// ---------------------------------------------------------------------
// A: hidden_states [num_tokens, K] bf16. Row for lane rc is
//    sorted_token_ids[pid_m*16+rc] // top_k (the top_k-strided gather: a
//    token appearing in top_k expert slots reads the same hidden_states row
//    top_k times, once per slot).
// B: w13_fp4 [E, N13=2*inter_size, K/2] u8, w13_scale [E, N13, K/32] u8.
//    Column n_tile*16+rc is the gate half, inter_size+n_tile*16+rc the up
//    half of the SAME output tile -- both GEMM passes share the same A rows
//    and grid position, only the B column base differs.
// Out: intermediate [num_valid_tokens, inter_size] bf16, one row per flat
//    (token, top_k-slot) index (== sorted_token_ids' value, matching
//    production's intermediate_cache1/2 indexing so stage2 and the final
//    sum-over-top_k reduction need no extra bookkeeping).
__global__ void moe_gemm_w4a16_stage1_kernel_scaffold(
    const c10::BFloat16* __restrict__ hidden_states, const uint8_t* __restrict__ w13_fp4,
    const uint8_t* __restrict__ w13_scale, const int32_t* __restrict__ sorted_token_ids,
    const int32_t* __restrict__ expert_ids, const int32_t* __restrict__ num_tokens_post_padded,
    c10::BFloat16* __restrict__ intermediate, int K, int num_k_waves, int inter_size, int top_k, int num_tokens,
    int num_valid_tokens, int64_t fp4_estride, int64_t scale_estride) {
  const int pid_m = blockIdx.x;
  const int n_tile = blockIdx.y;
  const int tid = threadIdx.x;
  const int wave = tid >> 6;
  const int lane = tid & 63;
  const int rc = lane & 15;
  const int blk = lane >> 4;
  const int m = tid >> 4;
  const int n = tid & 15;

  if (pid_m * BLOCK_M >= *num_tokens_post_padded) return;
  const int expert = expert_ids[pid_m];

  const int64_t out_row_id = static_cast<int64_t>(pid_m) * BLOCK_M + m;
  const int32_t out_row = sorted_token_ids[out_row_id];
  const bool out_valid = out_row < num_valid_tokens;

  if (expert < 0) {
    // EP-filtered block (moe_align_block_size's ignore_invalid_expert
    // sentinel): write zeros for this block's valid rows, matching the
    // Triton kernel's write_zeros_to_output.
    if (out_valid) {
      intermediate[static_cast<int64_t>(out_row) * inter_size + n_tile * BLOCK_M + n] = c10::BFloat16(0.f);
    }
    return;
  }

  const int64_t rc_row_id = static_cast<int64_t>(pid_m) * BLOCK_M + rc;
  const int32_t rc_offs_token = sorted_token_ids[rc_row_id];
  int64_t token_row = static_cast<int64_t>(rc_offs_token) / top_k;
  // Padding lanes may compute a garbage token row up to one past the last
  // valid row; clamp so the load stays in-bounds. The result is discarded
  // below (never written) since it only ever lands in an output row this
  // thread does not own or that fails out_valid.
  if (token_row < 0) token_row = 0;
  if (token_row >= num_tokens) token_row = num_tokens - 1;
  const c10::BFloat16* a_row_ptr = hidden_states + token_row * K;

  const uint8_t* w_fp4_e = w13_fp4 + static_cast<int64_t>(expert) * fp4_estride;
  const uint8_t* w_scale_e = w13_scale + static_cast<int64_t>(expert) * scale_estride;

  const int gate_col = n_tile * BLOCK_M + rc;
  const int up_col = inter_size + n_tile * BLOCK_M + rc;

  const f32x4 c_gate =
      gemm16x16_fp4_accumulate(a_row_ptr, w_fp4_e + static_cast<int64_t>(gate_col) * (K / 2),
                                w_scale_e + static_cast<int64_t>(gate_col) * (K / 32), K, num_k_waves, wave, blk);
  const float gate_val = reduce_waves(c_gate[0], c_gate[1], c_gate[2], c_gate[3], num_k_waves, wave, blk, rc);

  const f32x4 c_up =
      gemm16x16_fp4_accumulate(a_row_ptr, w_fp4_e + static_cast<int64_t>(up_col) * (K / 2),
                                w_scale_e + static_cast<int64_t>(up_col) * (K / 32), K, num_k_waves, wave, blk);
  const float up_val = reduce_waves(c_up[0], c_up[1], c_up[2], c_up[3], num_k_waves, wave, blk, rc);

  if (out_valid) {
    const float silu_gate = gate_val / (1.f + expf(-gate_val));
    intermediate[static_cast<int64_t>(out_row) * inter_size + n_tile * BLOCK_M + n] =
        c10::BFloat16(silu_gate * up_val);
  }
}

// ---------------------------------------------------------------------
// Stage 1 (templated production path): K=4096, instantiated at the "big"
// (NWAVES=8, WIDE=4, UDEPTH=2) and "small" (NWAVES=4, WIDE=2, UDEPTH=1)
// configs below. NWAVES/WIDE/UDEPTH are template parameters (not baked
// into a single hardcoded kernel) purely so both instantiations share one
// kernel body; each is still a distinct compiled kernel, selected by the
// host dispatch in moe_gemm_w4a16_stage1 below.
// ---------------------------------------------------------------------
// Same A/B/Out contract as the scaffold kernel above, except each block
// now covers SUBTILE_COLS=BLOCK_M*(WIDE/2) inter_size columns (WIDE/2 gate
// subtiles + their paired WIDE/2 up subtiles) per shared A-fragment
// K-walk, instead of the scaffold's 1 gate + 1 up column via two separate
// K-walks. See file header for the register-pressure rationale.
template <int K, int NWAVES, int WIDE, int UDEPTH>
__global__ void moe_gemm_w4a16_stage1_kernel_tpl(
    const c10::BFloat16* __restrict__ hidden_states, const uint8_t* __restrict__ w13_fp4,
    const uint8_t* __restrict__ w13_scale, const int32_t* __restrict__ sorted_token_ids,
    const int32_t* __restrict__ expert_ids, const int32_t* __restrict__ num_tokens_post_padded,
    c10::BFloat16* __restrict__ intermediate, int inter_size, int top_k, int num_tokens, int num_valid_tokens,
    int64_t fp4_estride, int64_t scale_estride) {
  constexpr int SUBTILES = WIDE / 2;
  constexpr int SUBTILE_COLS = BLOCK_M * SUBTILES;
  constexpr int TOTAL_THREADS = NWAVES * 64;
  // TOTAL_THREADS == SUBTILES*256 for both instantiations (big: 512==2*256,
  // small: 256==1*256): every thread owns exactly one (subtile, m, n)
  // output cell, no distribute loop needed.
  static_assert(TOTAL_THREADS == SUBTILES * 256,
                "stage1 templated launch config must give one thread per output cell");

  const int pid_m = blockIdx.x;
  const int n_tile = blockIdx.y;  // covers inter_size columns [n_tile*SUBTILE_COLS, ...+SUBTILE_COLS-1]
  const int tid = threadIdx.x;
  const int wave = tid >> 6;
  const int lane = tid & 63;
  const int rc = lane & 15;
  const int blk = lane >> 4;
  const int col_base = n_tile * SUBTILE_COLS;

  if (pid_m * BLOCK_M >= *num_tokens_post_padded) return;
  const int expert = expert_ids[pid_m];

  __shared__ int32_t out_row_cache[BLOCK_M];
  if (tid < BLOCK_M) {
    out_row_cache[tid] = sorted_token_ids[static_cast<int64_t>(pid_m) * BLOCK_M + tid];
  }
  __syncthreads();

  const int subtile = tid >> 8;
  const int mn = tid & 255;
  const int m = mn >> 4;
  const int n = mn & 15;
  const int32_t out_row = out_row_cache[m];
  const bool out_valid = out_row < num_valid_tokens;

  if (expert < 0) {
    if (out_valid) {
      intermediate[static_cast<int64_t>(out_row) * inter_size + col_base + subtile * BLOCK_M + n] =
          c10::BFloat16(0.f);
    }
    return;
  }

  const int64_t rc_row_id = static_cast<int64_t>(pid_m) * BLOCK_M + rc;
  const int32_t rc_offs_token = sorted_token_ids[rc_row_id];
  int64_t token_row = static_cast<int64_t>(rc_offs_token) / top_k;
  if (token_row < 0) token_row = 0;
  if (token_row >= num_tokens) token_row = num_tokens - 1;
  const c10::BFloat16* a_row_ptr = hidden_states + token_row * K;

  const uint8_t* w_fp4_e = w13_fp4 + static_cast<int64_t>(expert) * fp4_estride;
  const uint8_t* w_scale_e = w13_scale + static_cast<int64_t>(expert) * scale_estride;

  // cols[0..SUBTILES-1] = gate subtiles, cols[SUBTILES..WIDE-1] = the
  // paired up subtiles.
  int cols[WIDE];
#pragma unroll
  for (int s = 0; s < SUBTILES; ++s) cols[s] = col_base + s * BLOCK_M + rc;
#pragma unroll
  for (int s = 0; s < SUBTILES; ++s) cols[SUBTILES + s] = inter_size + col_base + s * BLOCK_M + rc;

  const uint8_t* w_fp4_cols[WIDE];
  const uint8_t* w_scale_cols[WIDE];
#pragma unroll
  for (int t = 0; t < WIDE; ++t) {
    w_fp4_cols[t] = w_fp4_e + static_cast<int64_t>(cols[t]) * (K / 2);
    w_scale_cols[t] = w_scale_e + static_cast<int64_t>(cols[t]) * (K / 32);
  }

  f32x4 c[WIDE];
  gemm_wide_tpl<K, NWAVES, WIDE, UDEPTH>(a_row_ptr, w_fp4_cols, w_scale_cols, wave, blk, c);

  __shared__ float lds[NWAVES][WIDE][16][16];
  store_wave_partials<NWAVES, WIDE>(c, lds, wave, blk, rc);
  __syncthreads();

  if (out_valid) {
    float gate_val = 0.f, up_val = 0.f;
#pragma unroll
    for (int w = 0; w < NWAVES; ++w) {
      gate_val += lds[w][subtile][m][n];
      up_val += lds[w][SUBTILES + subtile][m][n];
    }
    const float silu_gate = gate_val / (1.f + expf(-gate_val));
    intermediate[static_cast<int64_t>(out_row) * inter_size + col_base + subtile * BLOCK_M + n] =
        c10::BFloat16(silu_gate * up_val);
  }
}

// ---------------------------------------------------------------------
// Stage 2 production shape/config.
// ---------------------------------------------------------------------
constexpr int STAGE2_TPL_K = 256;    // intermediate_size_per_partition
constexpr int STAGE2_TPL_NWAVES = 2;  // results-v3's best down NWAVES (NWAVES=4 would give NLOADS=0 for K=256)
constexpr int STAGE2_TPL_WIDE = 4;    // results-v3's best down WIDE
constexpr int STAGE2_TPL_UDEPTH = 1;  // NLOADS=1 at this shape -- no room for prefetch (UD clamps to 1)
constexpr int STAGE2_TPL_COLS = BLOCK_M * STAGE2_TPL_WIDE;  // 64 hidden_size columns/block

// ---------------------------------------------------------------------
// Stage 2 (scaffold): down GEMM + routed-weight epilogue.
// ---------------------------------------------------------------------
// A: intermediate [num_valid_tokens, inter_size] bf16. Row for lane rc is
//    sorted_token_ids[pid_m*16+rc] directly (top_k=1 striding: intermediate
//    is already indexed by flat (token, slot), one row per slot, so no
//    further division is needed -- mirrors production calling
//    invoke_fused_moe_kernel's down pass with top_k=1).
// B: w2_fp4 [E, hidden_size, inter_size/2] u8, w2_scale [E, hidden_size,
//    inter_size/32] u8.
// Out: down_out [num_valid_tokens, hidden_size] bf16, same flat-row
//    indexing as the input; the routed weight is multiplied in here
//    (mirrors production's MUL_ROUTED_WEIGHT on the down GEMM). The Python
//    wrapper finishes the reduction with
//    down_out.view(num_tokens, top_k, hidden_size).sum(dim=1), valid
//    because a token's top_k slots occupy contiguous flat rows.
__global__ void moe_gemm_w4a16_stage2_kernel_scaffold(
    const c10::BFloat16* __restrict__ intermediate, const uint8_t* __restrict__ w2_fp4,
    const uint8_t* __restrict__ w2_scale, const int32_t* __restrict__ sorted_token_ids,
    const int32_t* __restrict__ expert_ids, const int32_t* __restrict__ num_tokens_post_padded,
    const float* __restrict__ topk_weights, c10::BFloat16* __restrict__ down_out, int inter_size, int num_k_waves,
    int hidden_size, int num_valid_tokens, int64_t fp4_estride, int64_t scale_estride) {
  const int pid_m = blockIdx.x;
  const int n_tile = blockIdx.y;
  const int tid = threadIdx.x;
  const int wave = tid >> 6;
  const int lane = tid & 63;
  const int rc = lane & 15;
  const int blk = lane >> 4;
  const int m = tid >> 4;
  const int n = tid & 15;

  if (pid_m * BLOCK_M >= *num_tokens_post_padded) return;
  const int expert = expert_ids[pid_m];

  const int64_t out_row_id = static_cast<int64_t>(pid_m) * BLOCK_M + m;
  const int32_t out_row = sorted_token_ids[out_row_id];
  const bool out_valid = out_row < num_valid_tokens;

  if (expert < 0) {
    if (out_valid) {
      down_out[static_cast<int64_t>(out_row) * hidden_size + n_tile * BLOCK_M + n] = c10::BFloat16(0.f);
    }
    return;
  }

  const int64_t rc_row_id = static_cast<int64_t>(pid_m) * BLOCK_M + rc;
  const int32_t rc_offs_token = sorted_token_ids[rc_row_id];
  int64_t token_row = static_cast<int64_t>(rc_offs_token);
  if (token_row < 0) token_row = 0;
  if (token_row >= num_valid_tokens) token_row = num_valid_tokens - 1;
  const c10::BFloat16* a_row_ptr = intermediate + token_row * inter_size;

  const uint8_t* w_fp4_e = w2_fp4 + static_cast<int64_t>(expert) * fp4_estride;
  const uint8_t* w_scale_e = w2_scale + static_cast<int64_t>(expert) * scale_estride;
  const int col = n_tile * BLOCK_M + rc;

  const f32x4 c = gemm16x16_fp4_accumulate(a_row_ptr, w_fp4_e + static_cast<int64_t>(col) * (inter_size / 2),
                                            w_scale_e + static_cast<int64_t>(col) * (inter_size / 32), inter_size,
                                            num_k_waves, wave, blk);
  const float val = reduce_waves(c[0], c[1], c[2], c[3], num_k_waves, wave, blk, rc);

  if (out_valid) {
    const float weight = topk_weights[out_row];
    down_out[static_cast<int64_t>(out_row) * hidden_size + n_tile * BLOCK_M + n] = c10::BFloat16(val * weight);
  }
}

// ---------------------------------------------------------------------
// Stage 2 (templated production path): K=256, NWAVES=2, WIDE=4, UDEPTH=1.
// ---------------------------------------------------------------------
// Same A/B/Out contract as the scaffold kernel above; each block now
// covers STAGE2_TPL_COLS=64 hidden_size columns sharing one A-fragment
// K-walk, instead of the scaffold's 1 column per block.
__global__ void moe_gemm_w4a16_stage2_kernel_tpl(
    const c10::BFloat16* __restrict__ intermediate, const uint8_t* __restrict__ w2_fp4,
    const uint8_t* __restrict__ w2_scale, const int32_t* __restrict__ sorted_token_ids,
    const int32_t* __restrict__ expert_ids, const int32_t* __restrict__ num_tokens_post_padded,
    const float* __restrict__ topk_weights, c10::BFloat16* __restrict__ down_out, int hidden_size,
    int num_valid_tokens, int64_t fp4_estride, int64_t scale_estride) {
  constexpr int K = STAGE2_TPL_K;
  constexpr int NWAVES = STAGE2_TPL_NWAVES;
  constexpr int WIDE = STAGE2_TPL_WIDE;
  constexpr int UDEPTH = STAGE2_TPL_UDEPTH;
  constexpr int TOTAL_THREADS = NWAVES * 64;
  constexpr int TOTAL_ELEMS = WIDE * 256;

  const int pid_m = blockIdx.x;
  const int n_tile = blockIdx.y;
  const int tid = threadIdx.x;
  const int wave = tid >> 6;
  const int lane = tid & 63;
  const int rc = lane & 15;
  const int blk = lane >> 4;
  const int col_base = n_tile * STAGE2_TPL_COLS;

  if (pid_m * BLOCK_M >= *num_tokens_post_padded) return;
  const int expert = expert_ids[pid_m];

  __shared__ int32_t out_row_cache[BLOCK_M];
  if (tid < BLOCK_M) {
    out_row_cache[tid] = sorted_token_ids[static_cast<int64_t>(pid_m) * BLOCK_M + tid];
  }
  __syncthreads();

  if (expert < 0) {
#pragma unroll
    for (int idx = tid; idx < TOTAL_ELEMS; idx += TOTAL_THREADS) {
      const int t = idx >> 8;
      const int mn = idx & 255;
      const int m = mn >> 4;
      const int n = mn & 15;
      const int32_t out_row = out_row_cache[m];
      if (out_row < num_valid_tokens) {
        down_out[static_cast<int64_t>(out_row) * hidden_size + col_base + t * BLOCK_M + n] = c10::BFloat16(0.f);
      }
    }
    return;
  }

  const int64_t rc_row_id = static_cast<int64_t>(pid_m) * BLOCK_M + rc;
  const int32_t rc_offs_token = sorted_token_ids[rc_row_id];
  int64_t token_row = static_cast<int64_t>(rc_offs_token);
  if (token_row < 0) token_row = 0;
  if (token_row >= num_valid_tokens) token_row = num_valid_tokens - 1;
  const c10::BFloat16* a_row_ptr = intermediate + token_row * K;

  const uint8_t* w_fp4_e = w2_fp4 + static_cast<int64_t>(expert) * fp4_estride;
  const uint8_t* w_scale_e = w2_scale + static_cast<int64_t>(expert) * scale_estride;

  int cols[WIDE];
#pragma unroll
  for (int t = 0; t < WIDE; ++t) cols[t] = col_base + t * BLOCK_M + rc;

  const uint8_t* w_fp4_cols[WIDE];
  const uint8_t* w_scale_cols[WIDE];
#pragma unroll
  for (int t = 0; t < WIDE; ++t) {
    w_fp4_cols[t] = w_fp4_e + static_cast<int64_t>(cols[t]) * (K / 2);
    w_scale_cols[t] = w_scale_e + static_cast<int64_t>(cols[t]) * (K / 32);
  }

  f32x4 c[WIDE];
  gemm_wide_tpl<K, NWAVES, WIDE, UDEPTH>(a_row_ptr, w_fp4_cols, w_scale_cols, wave, blk, c);

  __shared__ float lds[NWAVES][WIDE][16][16];
  store_wave_partials<NWAVES, WIDE>(c, lds, wave, blk, rc);
  __syncthreads();

#pragma unroll
  for (int idx = tid; idx < TOTAL_ELEMS; idx += TOTAL_THREADS) {
    const int t = idx >> 8;
    const int mn = idx & 255;
    const int m = mn >> 4;
    const int n = mn & 15;
    const int32_t out_row = out_row_cache[m];
    if (out_row < num_valid_tokens) {
      float sum = 0.f;
#pragma unroll
      for (int w = 0; w < NWAVES; ++w) sum += lds[w][t][m][n];
      const float weight = topk_weights[out_row];
      down_out[static_cast<int64_t>(out_row) * hidden_size + col_base + t * BLOCK_M + n] =
          c10::BFloat16(sum * weight);
    }
  }
}

void check_common(const torch::Tensor& fp4, const torch::Tensor& scale, const char* fp4_name,
                   const char* scale_name) {
  TORCH_CHECK(fp4.is_cuda() && fp4.scalar_type() == torch::kUInt8 && fp4.is_contiguous() && fp4.dim() == 3, fp4_name,
              " must be a contiguous [E, N, K/2] uint8 tensor");
  TORCH_CHECK(scale.is_cuda() && scale.scalar_type() == torch::kUInt8 && scale.is_contiguous() && scale.dim() == 3,
              scale_name, " must be a contiguous [E, N, K/32] uint8 tensor");
}

// Bisection/measurement knobs, read fresh on every dispatch call (a plain
// host branch made before the kernel launch it selects, so re-reading each
// call costs nothing perf-relevant and lets a single already-built
// extension be swept across forced modes without rebuilding -- see
// fused-validate-2.sbatch and test_mxfp4_fused_moe.py's stage1 dispatch
// sweep). "1" enables, matching the sbatch script's env convention.
bool env_flag_set(const char* name) {
  const char* v = std::getenv(name);
  return v != nullptr && v[0] == '1';
}

}  // namespace

// hidden_states: [num_tokens, K] bf16. w13_fp4: [E, 2*inter, K/2] u8.
// w13_scale: [E, 2*inter, K/32] u8. sorted_token_ids: [EM] i32.
// expert_ids: [EM/16] i32. num_tokens_post_padded: [1] i32 (device scalar,
// read in-kernel -- no host sync, CUDA-graph safe). intermediate:
// [num_valid_tokens, inter] bf16, pre-allocated by the caller.
//
// Dispatch: K==STAGE1_TPL_K (4096, this model's hidden_size) and
// inter_size a multiple of STAGE1_BIG_SUBTILE_COLS selects between the
// scaffold and "big" templated kernels by sorted-block count
// (STAGE1_SCAFFOLD_BLOCK_THRESHOLD, see file header); any other shape
// falls back to the scaffold kernel, which keeps handling any K with
// K%128==0 (checked below) at call time. The "small" templated kernel
// exists but is not part of this default rule -- see
// STAGE1_SCAFFOLD_BLOCK_THRESHOLD's comment.
void moe_gemm_w4a16_stage1(torch::Tensor hidden_states, torch::Tensor w13_fp4, torch::Tensor w13_scale,
                            torch::Tensor sorted_token_ids, torch::Tensor expert_ids,
                            torch::Tensor num_tokens_post_padded, torch::Tensor intermediate, int64_t top_k,
                            int64_t num_valid_tokens) {
  TORCH_CHECK(hidden_states.is_cuda() && hidden_states.scalar_type() == torch::kBFloat16 &&
                  hidden_states.is_contiguous() && hidden_states.dim() == 2,
              "hidden_states must be a contiguous [num_tokens, K] bf16 tensor");
  check_common(w13_fp4, w13_scale, "w13_fp4", "w13_scale");
  TORCH_CHECK(intermediate.is_cuda() && intermediate.scalar_type() == torch::kBFloat16 &&
                  intermediate.is_contiguous() && intermediate.dim() == 2,
              "intermediate must be a contiguous [num_valid_tokens, inter] bf16 tensor");
  TORCH_CHECK(sorted_token_ids.is_cuda() && sorted_token_ids.scalar_type() == torch::kInt32 &&
                  sorted_token_ids.is_contiguous(),
              "sorted_token_ids must be a contiguous int32 tensor");
  TORCH_CHECK(expert_ids.is_cuda() && expert_ids.scalar_type() == torch::kInt32 && expert_ids.is_contiguous(),
              "expert_ids must be a contiguous int32 tensor");
  TORCH_CHECK(num_tokens_post_padded.is_cuda() && num_tokens_post_padded.scalar_type() == torch::kInt32,
              "num_tokens_post_padded must be an int32 tensor");

  const int64_t num_tokens = hidden_states.size(0);
  const int64_t K = hidden_states.size(1);
  const int64_t N13 = w13_fp4.size(1);
  const int64_t inter_size = N13 / 2;
  TORCH_CHECK(w13_fp4.size(2) == K / 2, "w13_fp4 last dim must be K/2");
  TORCH_CHECK(w13_scale.size(1) == N13 && w13_scale.size(2) == K / 32, "w13_scale shape mismatch");
  TORCH_CHECK(intermediate.size(0) == num_valid_tokens && intermediate.size(1) == inter_size,
              "intermediate shape mismatch");
  TORCH_CHECK(K % 128 == 0,
              "K (hidden_size) must be a multiple of 128 -- see file header's known limitations "
              "and compute_num_k_waves");
  TORCH_CHECK(inter_size % BLOCK_M == 0, "2*intermediate_size must be a multiple of 32");

  auto stream = at::cuda::getCurrentCUDAStream();

  // Bisection/measurement overrides (see env_flag_set's comment).
  // force_scaffold takes precedence over force_small/force_big (a request
  // to isolate the scaffold kernel should not be silently overridden by a
  // stale force-small/force-big export); the sweep script that sets
  // force_small/force_big never sets both at once.
  const bool force_scaffold = env_flag_set("SGLANG_MXFP4_FUSED_FORCE_SCAFFOLD_STAGE1");
  const bool force_small = env_flag_set("SGLANG_MXFP4_FUSED_FORCE_SMALL_STAGE1");
  const bool force_big = env_flag_set("SGLANG_MXFP4_FUSED_FORCE_BIG_STAGE1");

  const bool big_shape_ok = (K == STAGE1_TPL_K) && (inter_size % STAGE1_BIG_SUBTILE_COLS == 0);
  const bool small_shape_ok = (K == STAGE1_TPL_K) && (inter_size % STAGE1_SMALL_SUBTILE_COLS == 0);
  const int num_blocks = static_cast<int>(expert_ids.size(0));

  // Default rule (no force env set): scaffold below the measured
  // block-count threshold (it beat both templated instantiations at every
  // measured point -- see STAGE1_SCAFFOLD_BLOCK_THRESHOLD's comment), big
  // at/above it, falling back to scaffold if the big shape doesn't apply
  // (e.g. K != STAGE1_TPL_K). "small" is not part of this default rule;
  // it is only reachable via the force env below.
  bool use_small, use_big;
  if (force_scaffold) {
    use_small = use_big = false;
  } else if (force_small && small_shape_ok) {
    use_small = true;
    use_big = false;
  } else if (force_big && big_shape_ok) {
    use_small = false;
    use_big = true;
  } else {
    use_small = false;
    use_big = big_shape_ok && num_blocks >= STAGE1_SCAFFOLD_BLOCK_THRESHOLD;
  }

  if (use_small) {
    const dim3 grid(static_cast<unsigned>(num_blocks), static_cast<unsigned>(inter_size / STAGE1_SMALL_SUBTILE_COLS));
    moe_gemm_w4a16_stage1_kernel_tpl<STAGE1_TPL_K, STAGE1_SMALL_NWAVES, STAGE1_SMALL_WIDE, STAGE1_SMALL_UDEPTH>
        <<<grid, dim3(STAGE1_SMALL_NWAVES * 64), 0, stream>>>(
            hidden_states.data_ptr<c10::BFloat16>(), w13_fp4.data_ptr<uint8_t>(), w13_scale.data_ptr<uint8_t>(),
            sorted_token_ids.data_ptr<int32_t>(), expert_ids.data_ptr<int32_t>(),
            num_tokens_post_padded.data_ptr<int32_t>(), intermediate.data_ptr<c10::BFloat16>(),
            static_cast<int>(inter_size), static_cast<int>(top_k), static_cast<int>(num_tokens),
            static_cast<int>(num_valid_tokens), w13_fp4.stride(0), w13_scale.stride(0));
  } else if (use_big) {
    const dim3 grid(static_cast<unsigned>(num_blocks), static_cast<unsigned>(inter_size / STAGE1_BIG_SUBTILE_COLS));
    moe_gemm_w4a16_stage1_kernel_tpl<STAGE1_TPL_K, STAGE1_BIG_NWAVES, STAGE1_BIG_WIDE, STAGE1_BIG_UDEPTH>
        <<<grid, dim3(STAGE1_BIG_NWAVES * 64), 0, stream>>>(
            hidden_states.data_ptr<c10::BFloat16>(), w13_fp4.data_ptr<uint8_t>(), w13_scale.data_ptr<uint8_t>(),
            sorted_token_ids.data_ptr<int32_t>(), expert_ids.data_ptr<int32_t>(),
            num_tokens_post_padded.data_ptr<int32_t>(), intermediate.data_ptr<c10::BFloat16>(),
            static_cast<int>(inter_size), static_cast<int>(top_k), static_cast<int>(num_tokens),
            static_cast<int>(num_valid_tokens), w13_fp4.stride(0), w13_scale.stride(0));
  } else {
    const int num_k_waves = compute_num_k_waves(static_cast<int>(K));
    const dim3 grid(static_cast<unsigned>(num_blocks), static_cast<unsigned>(inter_size / BLOCK_M));
    moe_gemm_w4a16_stage1_kernel_scaffold<<<grid, dim3(256), 0, stream>>>(
        hidden_states.data_ptr<c10::BFloat16>(), w13_fp4.data_ptr<uint8_t>(), w13_scale.data_ptr<uint8_t>(),
        sorted_token_ids.data_ptr<int32_t>(), expert_ids.data_ptr<int32_t>(),
        num_tokens_post_padded.data_ptr<int32_t>(), intermediate.data_ptr<c10::BFloat16>(), static_cast<int>(K),
        num_k_waves, static_cast<int>(inter_size), static_cast<int>(top_k), static_cast<int>(num_tokens),
        static_cast<int>(num_valid_tokens), w13_fp4.stride(0), w13_scale.stride(0));
  }
  C10_CUDA_KERNEL_LAUNCH_CHECK();
}

// intermediate: [num_valid_tokens, inter] bf16. w2_fp4: [E, hidden, inter/2]
// u8. w2_scale: [E, hidden, inter/32] u8. topk_weights: [num_tokens, top_k]
// fp32, addressed flat (row = sorted_token_ids' value, matching production's
// topk_weights_ptr + offs_token). down_out: [num_valid_tokens, hidden] bf16.
//
// Dispatch: inter_size==STAGE2_TPL_K (256, this model's
// intermediate_size_per_partition) and hidden_size a multiple of
// STAGE2_TPL_COLS (64) use the templated production kernel; any other
// shape falls back to the scaffold kernel.
void moe_gemm_w4a16_stage2(torch::Tensor intermediate, torch::Tensor w2_fp4, torch::Tensor w2_scale,
                            torch::Tensor sorted_token_ids, torch::Tensor expert_ids,
                            torch::Tensor num_tokens_post_padded, torch::Tensor topk_weights, torch::Tensor down_out,
                            int64_t num_valid_tokens) {
  TORCH_CHECK(intermediate.is_cuda() && intermediate.scalar_type() == torch::kBFloat16 &&
                  intermediate.is_contiguous() && intermediate.dim() == 2,
              "intermediate must be a contiguous [num_valid_tokens, inter] bf16 tensor");
  check_common(w2_fp4, w2_scale, "w2_fp4", "w2_scale");
  TORCH_CHECK(topk_weights.is_cuda() && topk_weights.scalar_type() == torch::kFloat32 && topk_weights.is_contiguous(),
              "topk_weights must be a contiguous fp32 tensor");
  TORCH_CHECK(down_out.is_cuda() && down_out.scalar_type() == torch::kBFloat16 && down_out.is_contiguous() &&
                  down_out.dim() == 2,
              "down_out must be a contiguous [num_valid_tokens, hidden] bf16 tensor");
  TORCH_CHECK(sorted_token_ids.is_cuda() && sorted_token_ids.scalar_type() == torch::kInt32 &&
                  sorted_token_ids.is_contiguous(),
              "sorted_token_ids must be a contiguous int32 tensor");
  TORCH_CHECK(expert_ids.is_cuda() && expert_ids.scalar_type() == torch::kInt32 && expert_ids.is_contiguous(),
              "expert_ids must be a contiguous int32 tensor");
  TORCH_CHECK(num_tokens_post_padded.is_cuda() && num_tokens_post_padded.scalar_type() == torch::kInt32,
              "num_tokens_post_padded must be an int32 tensor");

  const int64_t inter_size = intermediate.size(1);
  const int64_t hidden_size = w2_fp4.size(1);
  TORCH_CHECK(w2_fp4.size(2) == inter_size / 2, "w2_fp4 last dim must be inter_size/2");
  TORCH_CHECK(w2_scale.size(1) == hidden_size && w2_scale.size(2) == inter_size / 32, "w2_scale shape mismatch");
  TORCH_CHECK(down_out.size(0) == num_valid_tokens && down_out.size(1) == hidden_size, "down_out shape mismatch");
  TORCH_CHECK(inter_size % 128 == 0,
              "inter_size (intermediate_size) must be a multiple of 128 -- see file header's known "
              "limitations and compute_num_k_waves");
  TORCH_CHECK(hidden_size % BLOCK_M == 0, "hidden_size must be a multiple of 16");

  auto stream = at::cuda::getCurrentCUDAStream();
  if (inter_size == STAGE2_TPL_K && hidden_size % STAGE2_TPL_COLS == 0) {
    const dim3 grid(static_cast<unsigned>(expert_ids.size(0)), static_cast<unsigned>(hidden_size / STAGE2_TPL_COLS));
    moe_gemm_w4a16_stage2_kernel_tpl<<<grid, dim3(STAGE2_TPL_NWAVES * 64), 0, stream>>>(
        intermediate.data_ptr<c10::BFloat16>(), w2_fp4.data_ptr<uint8_t>(), w2_scale.data_ptr<uint8_t>(),
        sorted_token_ids.data_ptr<int32_t>(), expert_ids.data_ptr<int32_t>(),
        num_tokens_post_padded.data_ptr<int32_t>(), topk_weights.data_ptr<float>(),
        down_out.data_ptr<c10::BFloat16>(), static_cast<int>(hidden_size), static_cast<int>(num_valid_tokens),
        w2_fp4.stride(0), w2_scale.stride(0));
  } else {
    const int num_k_waves = compute_num_k_waves(static_cast<int>(inter_size));
    const dim3 grid(static_cast<unsigned>(expert_ids.size(0)), static_cast<unsigned>(hidden_size / BLOCK_M));
    moe_gemm_w4a16_stage2_kernel_scaffold<<<grid, dim3(256), 0, stream>>>(
        intermediate.data_ptr<c10::BFloat16>(), w2_fp4.data_ptr<uint8_t>(), w2_scale.data_ptr<uint8_t>(),
        sorted_token_ids.data_ptr<int32_t>(), expert_ids.data_ptr<int32_t>(),
        num_tokens_post_padded.data_ptr<int32_t>(), topk_weights.data_ptr<float>(),
        down_out.data_ptr<c10::BFloat16>(), static_cast<int>(inter_size), num_k_waves,
        static_cast<int>(hidden_size), static_cast<int>(num_valid_tokens), w2_fp4.stride(0), w2_scale.stride(0));
  }
  C10_CUDA_KERNEL_LAUNCH_CHECK();
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
  m.def("moe_gemm_w4a16_stage1", &moe_gemm_w4a16_stage1, "MXFP4 grouped GEMM, gate_up + silu(gate)*up epilogue");
  m.def("moe_gemm_w4a16_stage2", &moe_gemm_w4a16_stage2, "MXFP4 grouped GEMM, down + routed-weight epilogue");
}
