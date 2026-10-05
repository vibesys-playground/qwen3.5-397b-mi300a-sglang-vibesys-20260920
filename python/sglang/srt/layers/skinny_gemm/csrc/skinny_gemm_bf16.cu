// Skinny bf16 GEMM for small-M dense projections (decode-step qkv/o/gate
// projections at M <= 32) on gfx942 (MI300A).
//
// y[M, N] = x[M, K] (bf16, row-major) times w[N, K]^T (bf16, row-major, i.e.
// torch.nn.Linear weight layout: y = F.linear(x, w)), no bias here (the
// Python wrapper in skinny_gemm.py adds bias as a separate elementwise op).
//
// Design: every CU streams a share of w exactly once at full HBM bandwidth.
// Each 256-thread workgroup (4 waves x 64 lanes) owns ROWS_PER_WG contiguous
// rows of w (ROWS_PER_WAVE = ROWS_PER_WG / 4 rows per wave); lanes stride
// over K with 16-byte (8 x bf16) loads, accumulate in fp32 for all M rows of
// x simultaneously, then wave-reduce (shuffle butterfly across the 64-lane
// wavefront) and write out. x is small (M <= 32, so at most 32 * K * 2
// bytes) and is read directly from global memory: it is reused by every
// column-block's workgroup, so it stays resident in L2 rather than needing
// an explicit LDS staging step.
//
// Instantiation strategy (see resources/skills or the PR description for the
// campaign prototype this was ported from, which built 20 instantiations --
// 5 M values x 4 rows_per_wg values -- at ~67s cold-compile): this version
// templates on two axes only, to keep the compiled instantiation count at 4
// instead of 20:
//
//   MBUCKET (16 or 32): the compile-time unroll width for the M loop. A call
//   with actual M <= MBUCKET runs the same MBUCKET-wide unrolled loop, but
//   loads/accumulates/stores are masked off (`m < m_actual`) for the unused
//   rows, so a call never reads or writes past the true M rows of x/y. This
//   trades some wasted ALU/register work for far fewer compiled kernels: the
//   production gate (SGLANG_SKINNY_GEMM_MAX_M, default 16) only ever
//   instantiates the M=16 bucket at runtime; the M=32 bucket exists for
//   experimentation with a higher cap without recompiling for every M.
//
//   ROWS_PER_WAVE (1 or 2, i.e. rows_per_wg 4 or 8): the campaign's measured
//   result was that rows_per_wg=8 beats rows_per_wg=4 for every large-N
//   shape it swept, so 8 is the compiled default and 4 is kept only for
//   shapes where the occupancy heuristic (skinny_gemm.py's choose_config)
//   picks it to hit the target workgroup count. rows_per_wg values of 16 and
//   32 (in the prototype's sweep) are dropped: the measured sweep never
//   picked them as best on any shape in the six-shape decode benchmark.
//
// Split-K: for small-N shapes (e.g. N=32, N=512) there are too few w-rows to
// spread across an MI300A's CUs even with rows_per_wg=4. grid.y = split_k
// additional workgroups per row-block, each owning a K/split_k contiguous
// chunk; partial sums are accumulated into an fp32 scratch buffer (plain
// store when split_k == 1, atomicAdd otherwise).
//
// Reference read before writing this: aiter's wvSplitK_hf_/wvSplitK_hf_sml_
// family (csrc/kernels/custom_kernels.cu, ROCm/aiter) uses the same
// 64-lane/16-byte-load/fp32-accumulate/wave-shuffle-reduce shape for its
// skinny GEMV/GEMM kernels (their "N" template parameter is our M, their "M"
// is our N -- opposite naming convention). vLLM's csrc/rocm/skinny_gemms.cu
// follows the same pattern (shared ROCm lineage). This kernel does not reuse
// their code (both use LDS-staged activation and an inline-asm butterfly
// reduction tuned to their own tile shapes); it borrows only the general
// strategy, and uses portable HIP builtins (__shfl_xor, atomicAdd) instead
// of the inline row_shr/wave_shr asm.

#include <torch/extension.h>

#include <ATen/cuda/CUDAContext.h>
#include <c10/cuda/CUDAException.h>
#include <c10/cuda/CUDAGuard.h>

#include <cuda_runtime.h>

#include <cstdint>

namespace {

constexpr int WAVES = 4;    // 4 waves per workgroup -> 256 threads
constexpr int LANES = 64;   // wavefront size on CDNA (gfx942)
constexpr int A_CHUNK = 8;  // 8 x bf16 = 16 bytes per vectorized load
constexpr int STEP = LANES * A_CHUNK;  // elements of K advanced per loop iter

// bf16 (stored as raw uint16 bit pattern) -> fp32, exact (bf16 is the top 16
// bits of an fp32).
__device__ __forceinline__ float bf16_bits_to_f32(uint16_t bits) {
  return __uint_as_float(static_cast<uint32_t>(bits) << 16);
}

// 16-byte load unit: 8 packed bf16 values as 4 uint32 words (2 bf16 each).
struct Vec16B {
  uint32_t p[A_CHUNK / 2];
};

__device__ __forceinline__ Vec16B load16B(const uint16_t* addr) {
  return *reinterpret_cast<const Vec16B*>(addr);
}

__device__ __forceinline__ Vec16B zero16B() {
  Vec16B z;
#pragma unroll
  for (int p = 0; p < A_CHUNK / 2; ++p) z.p[p] = 0u;
  return z;
}

template <int MBUCKET, int ROWS_PER_WAVE>
__global__ __launch_bounds__(WAVES* LANES) void skinny_gemm_bf16_kernel(
    const uint16_t* __restrict__ x,  // [m_actual, K]
    const uint16_t* __restrict__ w,  // [N, K]
    float* __restrict__ acc_out,     // [m_actual, N] fp32 scratch/output accumulator
    int K,
    int k_len,
    int m_actual,
    bool use_atomic) {
  constexpr int ROWS_PER_WG = ROWS_PER_WAVE * WAVES;

  const int wave = threadIdx.y;
  const int lane = threadIdx.x;
  const int row0 = blockIdx.x * ROWS_PER_WG + wave * ROWS_PER_WAVE;
  const int k_start = blockIdx.y * k_len;
  const int k_end = k_start + k_len;

  float acc[ROWS_PER_WAVE][MBUCKET];
#pragma unroll
  for (int r = 0; r < ROWS_PER_WAVE; ++r) {
#pragma unroll
    for (int m = 0; m < MBUCKET; ++m) acc[r][m] = 0.0f;
  }

  for (int k = k_start + lane * A_CHUNK; k < k_end; k += STEP) {
    Vec16B wv[ROWS_PER_WAVE];
#pragma unroll
    for (int r = 0; r < ROWS_PER_WAVE; ++r) {
      wv[r] = load16B(w + static_cast<int64_t>(row0 + r) * K + k);
    }

    // Masked load: rows m >= m_actual read nothing (zero contribution)
    // rather than indexing past x's true [m_actual, K] extent.
    Vec16B xv[MBUCKET];
#pragma unroll
    for (int m = 0; m < MBUCKET; ++m) {
      xv[m] = (m < m_actual) ? load16B(x + static_cast<int64_t>(m) * K + k) : zero16B();
    }

#pragma unroll
    for (int r = 0; r < ROWS_PER_WAVE; ++r) {
#pragma unroll
      for (int m = 0; m < MBUCKET; ++m) {
        float partial = 0.0f;
#pragma unroll
        for (int p = 0; p < A_CHUNK / 2; ++p) {
          const uint32_t wp = wv[r].p[p];
          const uint32_t xp = xv[m].p[p];
          const float w0 = bf16_bits_to_f32(static_cast<uint16_t>(wp & 0xFFFFu));
          const float w1 = bf16_bits_to_f32(static_cast<uint16_t>(wp >> 16));
          const float x0 = bf16_bits_to_f32(static_cast<uint16_t>(xp & 0xFFFFu));
          const float x1 = bf16_bits_to_f32(static_cast<uint16_t>(xp >> 16));
          partial += w0 * x0 + w1 * x1;
        }
        acc[r][m] += partial;
      }
    }
  }

  // Butterfly reduction across the 64-lane wavefront.
#pragma unroll
  for (int r = 0; r < ROWS_PER_WAVE; ++r) {
#pragma unroll
    for (int m = 0; m < MBUCKET; ++m) {
      float v = acc[r][m];
#pragma unroll
      for (int off = LANES / 2; off > 0; off >>= 1) {
        v += __shfl_xor(v, off, LANES);
      }
      acc[r][m] = v;
    }
  }

  if (lane == 0) {
    // N is guaranteed to be a multiple of every supported ROWS_PER_WG value
    // (32 is a multiple of 4 and 8), so row0 + r never runs past N.
    const int N = gridDim.x * ROWS_PER_WG;
#pragma unroll
    for (int r = 0; r < ROWS_PER_WAVE; ++r) {
      const int col = row0 + r;
#pragma unroll
      for (int m = 0; m < MBUCKET; ++m) {
        // Masked store: never write past acc_out's true [m_actual, N] extent.
        if (m >= m_actual) continue;
        float* dst = acc_out + static_cast<int64_t>(m) * N + col;
        if (use_atomic) {
          atomicAdd(dst, acc[r][m]);
        } else {
          *dst = acc[r][m];
        }
      }
    }
  }
}

}  // namespace

void skinny_gemm_bf16(torch::Tensor x, torch::Tensor w, torch::Tensor scratch,
                       int64_t rows_per_wg, int64_t split_k) {
  TORCH_CHECK(x.is_cuda() && w.is_cuda() && scratch.is_cuda(),
              "x, w, scratch must all be on the GPU");
  TORCH_CHECK(x.scalar_type() == torch::kBFloat16, "x must be bf16");
  TORCH_CHECK(w.scalar_type() == torch::kBFloat16, "w must be bf16");
  TORCH_CHECK(scratch.scalar_type() == torch::kFloat32, "scratch must be fp32");
  TORCH_CHECK(x.dim() == 2 && w.dim() == 2 && scratch.dim() == 2,
              "x, w, scratch must be 2-D");
  TORCH_CHECK(x.is_contiguous() && w.is_contiguous() && scratch.is_contiguous(),
              "x, w, scratch must be contiguous");

  const int64_t M = x.size(0);
  const int64_t K = x.size(1);
  const int64_t N = w.size(0);
  TORCH_CHECK(w.size(1) == K, "K mismatch between x and w");
  TORCH_CHECK(scratch.size(0) == M && scratch.size(1) == N,
              "scratch shape must be [M, N]");
  TORCH_CHECK(M >= 1 && M <= 32, "M must be in [1, 32]");
  TORCH_CHECK(N % 32 == 0, "N must be a multiple of 32");
  TORCH_CHECK(K % 64 == 0, "K must be a multiple of 64");
  TORCH_CHECK(rows_per_wg == 4 || rows_per_wg == 8,
              "rows_per_wg must be 4 or 8, got ", rows_per_wg);
  TORCH_CHECK(N % rows_per_wg == 0,
              "N must be a multiple of rows_per_wg, got N=", N,
              " rows_per_wg=", rows_per_wg);
  TORCH_CHECK(split_k >= 1 && K % split_k == 0,
              "split_k must be >= 1 and divide K, got K=", K,
              " split_k=", split_k);
  const int64_t k_len = K / split_k;
  TORCH_CHECK(k_len % 8 == 0,
              "K / split_k must be a multiple of 8 for 16-byte loads, got ",
              k_len);

  const at::cuda::OptionalCUDAGuard device_guard(device_of(x));
  cudaStream_t stream = at::cuda::getCurrentCUDAStream();

  const bool use_atomic = split_k > 1;
  if (use_atomic) {
    // Graph-safe: a zero-fill kernel, same op sequence and buffer identity
    // on every replay.
    scratch.zero_();
  }

  const uint16_t* xp = reinterpret_cast<const uint16_t*>(x.data_ptr<at::BFloat16>());
  const uint16_t* wp = reinterpret_cast<const uint16_t*>(w.data_ptr<at::BFloat16>());
  float* accp = scratch.data_ptr<float>();

  const int rows_per_wave = static_cast<int>(rows_per_wg / WAVES);
  const int mbucket = (M <= 16) ? 16 : 32;
  dim3 block(LANES, WAVES);
  dim3 grid(static_cast<unsigned>(N / rows_per_wg), static_cast<unsigned>(split_k));

  bool launched = true;

#define SKINNY_LAUNCH(MBUCKET_, RPW_)                                        \
  skinny_gemm_bf16_kernel<MBUCKET_, RPW_><<<grid, block, 0, stream>>>(       \
      xp, wp, accp, static_cast<int>(K), static_cast<int>(k_len),            \
      static_cast<int>(M), use_atomic)

  if (mbucket == 16) {
    if (rows_per_wave == 1) {
      SKINNY_LAUNCH(16, 1);
    } else if (rows_per_wave == 2) {
      SKINNY_LAUNCH(16, 2);
    } else {
      launched = false;
    }
  } else {
    if (rows_per_wave == 1) {
      SKINNY_LAUNCH(32, 1);
    } else if (rows_per_wave == 2) {
      SKINNY_LAUNCH(32, 2);
    } else {
      launched = false;
    }
  }

#undef SKINNY_LAUNCH

  TORCH_CHECK(launched,
              "skinny_gemm_bf16: unsupported (M, rows_per_wg) combination: M=",
              M, " rows_per_wg=", rows_per_wg, " (rows_per_wg must be 4 or 8)");
  C10_CUDA_KERNEL_LAUNCH_CHECK();
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
  m.def("skinny_gemm_bf16", &skinny_gemm_bf16,
        "Skinny bf16 GEMM for small-M dense projections (HIP), writes an "
        "fp32 accumulator scratch; caller casts to the output dtype",
        py::arg("x"), py::arg("w"), py::arg("scratch"), py::arg("rows_per_wg"),
        py::arg("split_k"));
}
