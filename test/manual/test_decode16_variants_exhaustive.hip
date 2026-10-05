// Compile and run (plain hipcc, no torch extension build needed; the
// torch includes below are only for c10::BFloat16's rounding semantics):
//
//   TORCH_INCLUDES=$(python3 -c \
//     "from torch.utils.cpp_extension import include_paths; print(' '.join('-I'+p for p in include_paths()))")
//   hipcc -O2 --offload-arch=gfx942 -std=c++17 -D__HIP_PLATFORM_AMD__=1 -DUSE_ROCM=1 \
//     -isystem"$(python3 -c 'import sysconfig; print(sysconfig.get_paths()["include"])')" \
//     $TORCH_INCLUDES \
//     test/manual/test_decode16_variants_exhaustive.hip -o /tmp/test_decode16_variants
//   /tmp/test_decode16_variants
//
// Exits 0 and prints OVERALL_VERDICT=PASS iff all three parts (perm
// convention, sign-collapse, legacy-vs-select/legacy-vs-vperm exhaustive
// equivalence) pass; requires an actual gfx942 (MI300) device.
//
// Iteration 2: extends scratchpad/loop/moe-lut/test_decode_exhaustive.hip
// (job 634076) with a third `decode16_vperm` body (variant P, the
// __builtin_amdgcn_perm-based rewrite) and a standalone perm-convention
// unit test, for scratchpad/loop/moe-lut2/results.md.
//
// All three function bodies below are transcribed VERBATIM from
// python/sglang/srt/layers/moe/mxfp4_fused/csrc/mxfp4_fused_moe.cu's
// decode16() (the MXFP4_DECODE_LEGACY, MXFP4_DECODE_SELECT, and default
// #else branches respectively) -- kept here as a standalone plain-hipcc
// translation unit (no torch/extension/pybind dependency).
//
// Part A (perm_convention_kernel): confirms, BEFORE trusting any of
// decode16_vperm's derivation, that `__builtin_amdgcn_perm(src0, src1,
// sel)` on this target follows the assumed convention: selector byte
// value 0-3 picks byte (value) of src1, value 4-7 picks byte (value-4) of
// src0. If this check fails, decode16_vperm's byte-plane packing
// (lo0/lo1/hi0/hi1 layout, SEL_AB/SEL_CD constants) is invalid and the
// exhaustive test below is expected to fail too -- the printed verdict
// says explicitly which one broke first.
//
// Part B (sign-collapse check): confirms the claim
// `to_bf16_bits(ldexpf(-MXFP4_LUT[k], e)) == to_bf16_bits(ldexpf(MXFP4_LUT[k], e)) | 0x8000`
// empirically for all 256 e8m0 values x all 8 magnitude indices k, on
// real device rounding (not just the host-side c10::BFloat16 semantics
// assumed in scratchpad/verify_perm_math.py) -- decode16_vperm's whole
// 8-entry (vs. 16-entry) table design depends on this holding exactly.
//
// Part C (exhaustive equivalence): one GPU thread per (e8m0 0..255) x
// (seed 0..255) pair (65536 threads). Each thread builds a 16-byte block
// via `bytes[i] = (seed + i * 97) & 0xFF` (97 is odd, so this sweeps
// every byte position through a different, position-dependent value as
// `seed` ranges 0..255, exercising cross-byte-boundary behavior of
// decode16_vperm's shift-based nibble extraction -- stronger than the
// iteration-1 test's uniform-fill pattern, which cannot distinguish a
// byte-boundary bug from a per-byte bug since all 16 input bytes were
// identical). Compares decode16_legacy vs decode16_select (established
// bit-identical in iteration 1, re-checked here as a regression guard)
// and decode16_legacy vs decode16_vperm (the new claim under test) for
// all 32 elements/thread = 2,097,152 comparisons per pair.

#include <hip/hip_runtime.h>
#include <c10/util/BFloat16.h>

#include <cmath>
#include <cstdint>
#include <cstdio>
#include <cstring>

namespace {

__constant__ float MXFP4_LUT[16] = {
    0.0f, 0.5f, 1.0f, 1.5f, 2.0f, 3.0f, 4.0f, 6.0f,
    -0.0f, -0.5f, -1.0f, -1.5f, -2.0f, -3.0f, -4.0f, -6.0f,
};
constexpr int E8M0_BIAS = 127;

__device__ __forceinline__ unsigned short to_bf16_bits(float v) { return c10::BFloat16(v).x; }

// Verbatim transcription of decode16's MXFP4_DECODE_LEGACY branch.
__device__ void decode16_legacy(const uint8_t bytes[16], int block_exp, unsigned short vals[32]) {
#pragma unroll
  for (int b = 0; b < 16; ++b) {
    const uint8_t byte = bytes[b];
    const float lo = ldexpf(MXFP4_LUT[byte & 0xF], block_exp);
    const float hi = ldexpf(MXFP4_LUT[(byte >> 4) & 0xF], block_exp);
    vals[2 * b] = to_bf16_bits(lo);
    vals[2 * b + 1] = to_bf16_bits(hi);
  }
}

// Verbatim transcription of decode16's MXFP4_DECODE_SELECT branch
// (iteration 1's body, job 634076 -- the compare/select-chain regression).
__device__ void decode16_select(const uint8_t bytes[16], int block_exp, unsigned short vals[32]) {
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
}

// Verbatim transcription of decode16's default (variant P, v_perm-based)
// branch. See mxfp4_fused_moe.cu's decode16() comment for the derivation.
__device__ void decode16_vperm(const uint8_t bytes[16], int block_exp, unsigned short vals[32]) {
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
}

// ---------------------------------------------------------------------
// Part A: __builtin_amdgcn_perm operand-order convention.
// ---------------------------------------------------------------------
__global__ void perm_convention_kernel(int* fail_count, uint32_t* info) {
  if (threadIdx.x != 0 || blockIdx.x != 0) return;
  const unsigned int src0 = 0x03020100u;  // bytes (msb..lsb): 03 02 01 00
  const unsigned int src1 = 0x07060504u;  // bytes (msb..lsb): 07 06 05 04
  // Assumed convention: sel byte value v in [0,3] -> src1 byte v;
  // v in [4,7] -> src0 byte (v-4). Check every output-byte lane
  // independently by using a distinct selector per lane.
  struct Case {
    unsigned int sel;
    unsigned int expected;
  };
  // Selector byte i (i=0..3) chooses what output byte i becomes. Design
  // selector bytes (b0,b1,b2,b3) = (0,3,4,7) so: out.b0 selb=0 -> idx=0
  // (<4) -> src1.byte0=0x04; out.b1 selb=3 -> idx=3 -> src1.byte3=0x07;
  // out.b2 selb=4 -> idx=0 (4-4) -> src0.byte0=0x00; out.b3 selb=7 ->
  // idx=3 (7-4) -> src0.byte3=0x03. Packed little-endian:
  // sel = (7<<24)|(4<<16)|(3<<8)|0 = 0x07040300.
  const unsigned int sel = 0x07040300u;
  // Expected output bytes (b0,b1,b2,b3) = (0x04,0x07,0x00,0x03), packed
  // little-endian: (3<<24)|(0<<16)|(7<<8)|4 = 0x03000704.
  const unsigned int expected = 0x03000704u;
  const unsigned int got = __builtin_amdgcn_perm(src0, src1, sel);
  info[0] = got;
  info[1] = expected;
  if (got != expected) {
    atomicAdd(fail_count, 1);
  }
}

// ---------------------------------------------------------------------
// Part B: sign-collapse claim, on-device.
// ---------------------------------------------------------------------
__global__ void sign_collapse_kernel(int* mismatch_count, uint32_t* first_mismatch_info) {
  const int idx = blockIdx.x * blockDim.x + threadIdx.x;
  if (idx >= 256 * 8) return;
  const int e8m0 = idx / 8;
  const int k = idx % 8;
  const int block_exp = e8m0 - E8M0_BIAS;
  const unsigned short pos = to_bf16_bits(ldexpf(MXFP4_LUT[k], block_exp));
  const unsigned short neg = to_bf16_bits(ldexpf(MXFP4_LUT[k + 8], block_exp));
  const unsigned short expected = pos | 0x8000;
  if (neg != expected) {
    const int prev = atomicAdd(mismatch_count, 1);
    if (prev == 0) {
      first_mismatch_info[0] = static_cast<uint32_t>(e8m0);
      first_mismatch_info[1] = static_cast<uint32_t>(k);
      first_mismatch_info[2] = static_cast<uint32_t>(pos);
      first_mismatch_info[3] = static_cast<uint32_t>(neg);
      first_mismatch_info[4] = static_cast<uint32_t>(expected);
    }
  }
}

// ---------------------------------------------------------------------
// Part C: exhaustive equivalence, heterogeneous per-position bytes.
// ---------------------------------------------------------------------
__global__ void exhaustive_kernel(int* mismatch_select, uint32_t* first_mismatch_select,
                                   int* mismatch_vperm, uint32_t* first_mismatch_vperm) {
  const int idx = blockIdx.x * blockDim.x + threadIdx.x;
  if (idx >= 256 * 256) return;
  const int e8m0 = idx / 256;
  const int seed = idx % 256;
  const int block_exp = e8m0 - E8M0_BIAS;

  uint8_t bytes[16];
#pragma unroll
  for (int i = 0; i < 16; ++i) bytes[i] = static_cast<uint8_t>((seed + i * 97) & 0xFF);

  unsigned short legacy_vals[32];
  unsigned short select_vals[32];
  unsigned short vperm_vals[32];
  decode16_legacy(bytes, block_exp, legacy_vals);
  decode16_select(bytes, block_exp, select_vals);
  decode16_vperm(bytes, block_exp, vperm_vals);

#pragma unroll
  for (int i = 0; i < 32; ++i) {
    if (legacy_vals[i] != select_vals[i]) {
      const int prev = atomicAdd(mismatch_select, 1);
      if (prev == 0) {
        first_mismatch_select[0] = static_cast<uint32_t>(e8m0);
        first_mismatch_select[1] = static_cast<uint32_t>(seed);
        first_mismatch_select[2] = static_cast<uint32_t>(i);
        first_mismatch_select[3] = static_cast<uint32_t>(legacy_vals[i]);
        first_mismatch_select[4] = static_cast<uint32_t>(select_vals[i]);
      }
    }
    if (legacy_vals[i] != vperm_vals[i]) {
      const int prev = atomicAdd(mismatch_vperm, 1);
      if (prev == 0) {
        first_mismatch_vperm[0] = static_cast<uint32_t>(e8m0);
        first_mismatch_vperm[1] = static_cast<uint32_t>(seed);
        first_mismatch_vperm[2] = static_cast<uint32_t>(i);
        first_mismatch_vperm[3] = static_cast<uint32_t>(legacy_vals[i]);
        first_mismatch_vperm[4] = static_cast<uint32_t>(vperm_vals[i]);
      }
    }
  }
}

}  // namespace

namespace {
template <typename KernelPairT>
int run_pair_check(const char* label, KernelPairT&& launch, int total_elems) {
  int* d_count = nullptr;
  uint32_t* d_info = nullptr;
  hipMalloc(&d_count, sizeof(int));
  hipMalloc(&d_info, sizeof(uint32_t) * 5);
  hipMemset(d_count, 0, sizeof(int));
  hipMemset(d_info, 0, sizeof(uint32_t) * 5);
  launch(d_count, d_info);
  const hipError_t err = hipDeviceSynchronize();
  if (err != hipSuccess) {
    fprintf(stderr, "HIP error in %s: %s\n", label, hipGetErrorString(err));
    hipFree(d_count);
    hipFree(d_info);
    return 2;
  }
  int h_count = 0;
  uint32_t h_info[5] = {0, 0, 0, 0, 0};
  hipMemcpy(&h_count, d_count, sizeof(int), hipMemcpyDeviceToHost);
  hipMemcpy(h_info, d_info, sizeof(uint32_t) * 5, hipMemcpyDeviceToHost);
  hipFree(d_count);
  hipFree(d_info);
  printf("%s_TOTAL_CASES=%d\n", label, total_elems);
  printf("%s_MISMATCHES=%d\n", label, h_count);
  if (h_count > 0) {
    printf("%s_FIRST_MISMATCH a=%u b=%u elem_idx=%u ref_bits=0x%04x got_bits=0x%04x\n", label, h_info[0], h_info[1],
           h_info[2], h_info[3], h_info[4]);
    printf("%s_VERDICT=FAIL\n", label);
    return 1;
  }
  printf("%s_VERDICT=PASS\n", label);
  return 0;
}
}  // namespace

int main() {
  int overall_rc = 0;

  // Part A: perm convention.
  {
    int* d_fail = nullptr;
    uint32_t* d_info = nullptr;
    hipMalloc(&d_fail, sizeof(int));
    hipMalloc(&d_info, sizeof(uint32_t) * 2);
    hipMemset(d_fail, 0, sizeof(int));
    hipLaunchKernelGGL(perm_convention_kernel, dim3(1), dim3(1), 0, 0, d_fail, d_info);
    hipDeviceSynchronize();
    int h_fail = 0;
    uint32_t h_info[2] = {0, 0};
    hipMemcpy(&h_fail, d_fail, sizeof(int), hipMemcpyDeviceToHost);
    hipMemcpy(h_info, d_info, sizeof(uint32_t) * 2, hipMemcpyDeviceToHost);
    hipFree(d_fail);
    hipFree(d_info);
    printf("PERM_CONVENTION_GOT=0x%08x EXPECTED=0x%08x\n", h_info[0], h_info[1]);
    printf("PERM_CONVENTION_VERDICT=%s\n", h_fail == 0 ? "PASS" : "FAIL");
    if (h_fail != 0) overall_rc = 1;
  }

  // Part B: sign collapse.
  {
    const int total = 256 * 8;
    const int threads = 256;
    const int blocks = (total + threads - 1) / threads;
    int rc = run_pair_check(
        "SIGN_COLLAPSE",
        [&](int* d_count, uint32_t* d_info) {
          hipLaunchKernelGGL(sign_collapse_kernel, dim3(blocks), dim3(threads), 0, 0, d_count, d_info);
        },
        total);
    if (rc != 0) overall_rc = rc;
  }

  // Part C: exhaustive equivalence (legacy vs select, legacy vs vperm).
  {
    const int total_pairs = 256 * 256;
    const int threads = 256;
    const int blocks = (total_pairs + threads - 1) / threads;
    int* d_mismatch_select = nullptr;
    uint32_t* d_first_select = nullptr;
    int* d_mismatch_vperm = nullptr;
    uint32_t* d_first_vperm = nullptr;
    hipMalloc(&d_mismatch_select, sizeof(int));
    hipMalloc(&d_first_select, sizeof(uint32_t) * 5);
    hipMalloc(&d_mismatch_vperm, sizeof(int));
    hipMalloc(&d_first_vperm, sizeof(uint32_t) * 5);
    hipMemset(d_mismatch_select, 0, sizeof(int));
    hipMemset(d_first_select, 0, sizeof(uint32_t) * 5);
    hipMemset(d_mismatch_vperm, 0, sizeof(int));
    hipMemset(d_first_vperm, 0, sizeof(uint32_t) * 5);

    hipLaunchKernelGGL(exhaustive_kernel, dim3(blocks), dim3(threads), 0, 0, d_mismatch_select, d_first_select,
                        d_mismatch_vperm, d_first_vperm);
    const hipError_t err = hipDeviceSynchronize();
    if (err != hipSuccess) {
      fprintf(stderr, "HIP error in EXHAUSTIVE: %s\n", hipGetErrorString(err));
      return 2;
    }

    int h_mismatch_select = 0, h_mismatch_vperm = 0;
    uint32_t h_first_select[5] = {0, 0, 0, 0, 0};
    uint32_t h_first_vperm[5] = {0, 0, 0, 0, 0};
    hipMemcpy(&h_mismatch_select, d_mismatch_select, sizeof(int), hipMemcpyDeviceToHost);
    hipMemcpy(h_first_select, d_first_select, sizeof(uint32_t) * 5, hipMemcpyDeviceToHost);
    hipMemcpy(&h_mismatch_vperm, d_mismatch_vperm, sizeof(int), hipMemcpyDeviceToHost);
    hipMemcpy(h_first_vperm, d_first_vperm, sizeof(uint32_t) * 5, hipMemcpyDeviceToHost);
    hipFree(d_mismatch_select);
    hipFree(d_first_select);
    hipFree(d_mismatch_vperm);
    hipFree(d_first_vperm);

    const int total_elems = total_pairs * 32;
    printf("EXHAUSTIVE_LEGACY_VS_SELECT_TOTAL_CASES=%d\n", total_elems);
    printf("EXHAUSTIVE_LEGACY_VS_SELECT_MISMATCHES=%d\n", h_mismatch_select);
    if (h_mismatch_select > 0) {
      printf("EXHAUSTIVE_LEGACY_VS_SELECT_FIRST_MISMATCH e8m0=%u seed=%u elem_idx=%u legacy_bits=0x%04x select_bits=0x%04x\n",
             h_first_select[0], h_first_select[1], h_first_select[2], h_first_select[3], h_first_select[4]);
      printf("EXHAUSTIVE_LEGACY_VS_SELECT_VERDICT=FAIL\n");
      overall_rc = 1;
    } else {
      printf("EXHAUSTIVE_LEGACY_VS_SELECT_VERDICT=PASS\n");
    }

    printf("EXHAUSTIVE_LEGACY_VS_VPERM_TOTAL_CASES=%d\n", total_elems);
    printf("EXHAUSTIVE_LEGACY_VS_VPERM_MISMATCHES=%d\n", h_mismatch_vperm);
    if (h_mismatch_vperm > 0) {
      printf("EXHAUSTIVE_LEGACY_VS_VPERM_FIRST_MISMATCH e8m0=%u seed=%u elem_idx=%u legacy_bits=0x%04x vperm_bits=0x%04x\n",
             h_first_vperm[0], h_first_vperm[1], h_first_vperm[2], h_first_vperm[3], h_first_vperm[4]);
      printf("EXHAUSTIVE_LEGACY_VS_VPERM_VERDICT=FAIL\n");
      overall_rc = 1;
    } else {
      printf("EXHAUSTIVE_LEGACY_VS_VPERM_VERDICT=PASS\n");
    }
  }

  printf("OVERALL_VERDICT=%s\n", overall_rc == 0 ? "PASS" : "FAIL");
  return overall_rc;
}
