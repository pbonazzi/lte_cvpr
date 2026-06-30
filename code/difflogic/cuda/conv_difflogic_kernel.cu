#include <torch/extension.h>
using Slice = torch::indexing::Slice;

#include <cuda.h>
#include <cuda_runtime.h>

#include <array>
#include <cmath>
#include <vector>

// macros
#define CHECK_CUDA(x) TORCH_CHECK(x.device().is_cuda(), #x " must be a CUDA tensor")
#define CHECK_CONTIGUOUS(x) TORCH_CHECK(x.is_contiguous(), #x " must be contiguous")
#define CHECK_INPUT(x)                                                                                                 \
    CHECK_CUDA(x);                                                                                                     \
    CHECK_CONTIGUOUS(x)

#define gpuErrchk(ans) {gpuAssert((ans), __FILE__, __LINE__);}
inline void gpuAssert(const cudaError_t code, const char *const file, const int line, const bool abort = true) {
    if (code != cudaSuccess) {
        fprintf(stderr, "GPUassert: %s %s %d\n", cudaGetErrorString(code), file, line);
        if (abort) exit(code);
    }
}

#define assertionCheck(cond, msg) {logicAssert((cond), msg, __FILE__, __LINE__);}
inline void logicAssert(bool condition, const char* errorMsg, const char* file, int line, bool abort = true) {
    if (!condition) {
        fprintf(stderr, "assertionCheck failed: %s %s %d\n", errorMsg, file, line);
        if (abort) exit(EXIT_FAILURE);
    }
}

// template <typename T> T ceil_div(const T x, const T y) {return x / y + !!(x % y);}

// thread block size
#define BLOCK_SIZE 16
#define PAD 1
#define RF 3
#define STRIDE 1
#define D 3


// forward declaration of binary operation sum
template <typename scalar_t>
__device__ __forceinline__ scalar_t bin_op_s(scalar_t a, scalar_t b, scalar_t* weights);


// forward declaration of convolutional forward kernel
template <typename scalar_t>
__global__ void conv_forward_kernel(
    torch::PackedTensorAccessor64<scalar_t, 4, torch::RestrictPtrTraits> x,
    torch::PackedTensorAccessor64<scalar_t, 3, torch::RestrictPtrTraits> weights,
    torch::PackedTensorAccessor64<int64_t, 2, torch::RestrictPtrTraits> c_m,
    torch::PackedTensorAccessor64<int64_t, 2, torch::RestrictPtrTraits> c_w,
    torch::PackedTensorAccessor64<int64_t, 2, torch::RestrictPtrTraits> c_h,
    torch::PackedTensorAccessor64<int64_t, 2, torch::RestrictPtrTraits> c_m_occ,
    int64_t pad,
    int64_t rf,
    int64_t batch_size,
    int64_t out_h,
    int64_t out_w,
    int64_t stride,
    int64_t d,
    torch::PackedTensorAccessor64<scalar_t, 4, torch::RestrictPtrTraits> out
);


template <typename scalar_t>
__device__ __forceinline__ void partial_wrt_w4(
    scalar_t* s_buff_4_w4,
    scalar_t partial_z,
    scalar_t a,
    scalar_t b,
    scalar_t* w,
    torch::PackedTensorAccessor64<scalar_t, 4, torch::RestrictPtrTraits> grad_w_4,
    int64_t thread_idx_in_block,
    int64_t t,
    int64_t block_idx,
    int64_t w_idx
);


template <typename scalar_t>
__device__ __forceinline__ scalar_t partial_wrt_in(scalar_t a, scalar_t b, scalar_t* w, bool da);


template <typename scalar_t>
__global__ void grad_x_reduction(
    torch::PackedTensorAccessor64<scalar_t, 5, torch::RestrictPtrTraits> grad_x_raw,
    torch::PackedTensorAccessor64<int64_t, 2, torch::RestrictPtrTraits> c_m_occ,
    int64_t h,
    int64_t w,
    int64_t tiles_per_side,
    torch::PackedTensorAccessor64<scalar_t, 5, torch::RestrictPtrTraits> grad_x_hw
);


// forward declaration of convolutional backward kernel w.r.t weights
template <typename scalar_t>
__global__ void conv_backward_w_kernel(
    torch::PackedTensorAccessor64<scalar_t, 4, torch::RestrictPtrTraits> x,
    torch::PackedTensorAccessor64<scalar_t, 3, torch::RestrictPtrTraits> weights,
    torch::PackedTensorAccessor64<int64_t, 2, torch::RestrictPtrTraits> c_m,
    torch::PackedTensorAccessor64<int64_t, 2, torch::RestrictPtrTraits> c_w,
    torch::PackedTensorAccessor64<int64_t, 2, torch::RestrictPtrTraits> c_h,
    torch::PackedTensorAccessor64<int64_t, 2, torch::RestrictPtrTraits> c_m_occ,
    int64_t pad,
    int64_t rf,
    int64_t batch_size,
    int64_t out_h,
    int64_t out_w,
    int64_t stride,
    int64_t d,
    torch::PackedTensorAccessor64<scalar_t, 4, torch::RestrictPtrTraits> y,
    torch::PackedTensorAccessor64<scalar_t, 4, torch::RestrictPtrTraits> grad_w_4
);


// forward declaration of convolutional backward kernel w.r.t inputs
template <typename scalar_t>
__global__ void conv_backward_x_kernel(
    torch::PackedTensorAccessor64<scalar_t, 4, torch::RestrictPtrTraits> x,
    torch::PackedTensorAccessor64<scalar_t, 3, torch::RestrictPtrTraits> weights,
    torch::PackedTensorAccessor64<int64_t, 2, torch::RestrictPtrTraits> c_m,
    torch::PackedTensorAccessor64<int64_t, 2, torch::RestrictPtrTraits> c_w,
    torch::PackedTensorAccessor64<int64_t, 2, torch::RestrictPtrTraits> c_h,
    torch::PackedTensorAccessor64<int64_t, 2, torch::RestrictPtrTraits> c_m_occ,
    int64_t pad,
    int64_t rf,
    int64_t batch_size,
    int64_t out_h,
    int64_t out_w,
    int64_t stride,
    int64_t d,
    torch::PackedTensorAccessor64<scalar_t, 4, torch::RestrictPtrTraits> grad_y,
    torch::PackedTensorAccessor64<scalar_t, 5, torch::RestrictPtrTraits> grad_x_raw
);


// forward kernel invocation
torch::Tensor conv_forward(
    torch::Tensor x,
    torch::Tensor weights,
    torch::Tensor c_m,
    torch::Tensor c_h,
    torch::Tensor c_w,
    torch::Tensor c_m_occ,
    int64_t pad,
    int64_t rf,
    int64_t stride,
    int64_t d
) {
    assertionCheck(
    (pad == PAD && rf == RF && stride == STRIDE && d == D),
    "Not implemented for pad != 1 or rf != 3x3 or stride != 1."
    );

    CHECK_INPUT(x); CHECK_INPUT(weights);
    CHECK_INPUT(c_m); CHECK_INPUT(c_h); CHECK_INPUT(c_w); CHECK_INPUT(c_m_occ);

    // # x of shape [batch_size, channels, height, width]; not padded yet
    assertionCheck(x.dim() == 4, "Incorrect input dimensions.");
    int64_t batch_size = x.size(0);
    int64_t n_chs = x.size(1);
    int64_t h = x.size(2);
    int64_t w = x.size(3);
    assertionCheck(h==w && h%4==0, "Not implemented for input h, w non multiples of 4.");

    int64_t n_trees = c_m_occ.size(0);

    int64_t out_h = h;
    int64_t out_w = w;
    torch::Tensor out = torch::empty({batch_size, n_trees, out_h, out_w}, torch::dtype(x.dtype()).device(x.device()));

    // define dimBlock and dimGrid
    dim3 dimBlock(BLOCK_SIZE, BLOCK_SIZE);

    int64_t n_blocks_per_tree =  batch_size*out_h*out_w/(BLOCK_SIZE*BLOCK_SIZE);
    // if second condition not present the following setting would be allowed:
    // batch_size, w=h, BLOCK_SIZE = 49, 3, 7
    assertionCheck(
        ((batch_size*out_h*out_w) % (BLOCK_SIZE*BLOCK_SIZE) == 0 &&
        out_h <= batch_size ? batch_size%h==0 : h%batch_size==0 &&
        n_trees <= 65535 && n_blocks_per_tree <= 65535),
        "Combination of batch size, input dimensions (h,w), block size not allowed."
    );
    dim3 dimGrid (n_trees, n_blocks_per_tree);

    // invoke kernel
    AT_DISPATCH_FLOATING_TYPES_AND_HALF(x.scalar_type(), "conv_forward", ([&] {
        size_t pow2_d = static_cast<size_t>(1) << D;
        size_t mem_c_m = (pow2_d) * sizeof(int64_t);
        size_t mem_c_h = (pow2_d) * sizeof(int64_t);
        size_t mem_c_w = (pow2_d) * sizeof(int64_t);
        size_t mem_weights = (pow2_d -1)*16 * sizeof(scalar_t);
        size_t s_mem_size = mem_c_m+mem_c_h+mem_c_w + mem_weights;

        conv_forward_kernel<scalar_t><<<dimGrid, dimBlock, s_mem_size>>>(
            x.packed_accessor64<scalar_t, 4, torch::RestrictPtrTraits>(),
            weights.packed_accessor64<scalar_t, 3, torch::RestrictPtrTraits>(),
            c_m.packed_accessor64<int64_t, 2, torch::RestrictPtrTraits>(),
            c_w.packed_accessor64<int64_t, 2, torch::RestrictPtrTraits>(),
            c_h.packed_accessor64<int64_t, 2, torch::RestrictPtrTraits>(),
            c_m_occ.packed_accessor64<int64_t, 2, torch::RestrictPtrTraits>(),
            pad,
            rf,
            batch_size,
            out_h,
            out_w,
            stride,
            d,
            out.packed_accessor64<scalar_t, 4, torch::RestrictPtrTraits>()
        );
    }));

    // check errors during kernel execution
    gpuErrchk(cudaPeekAtLastError());
    // synchronize device == wait until all computations on device are finshed
    gpuErrchk(cudaDeviceSynchronize());

    return out;
}


template <typename scalar_t>
__global__ void conv_forward_kernel(
    torch::PackedTensorAccessor64<scalar_t, 4, torch::RestrictPtrTraits> x,
    torch::PackedTensorAccessor64<scalar_t, 3, torch::RestrictPtrTraits> weights,
    torch::PackedTensorAccessor64<int64_t, 2, torch::RestrictPtrTraits> c_m,
    torch::PackedTensorAccessor64<int64_t, 2, torch::RestrictPtrTraits> c_w,
    torch::PackedTensorAccessor64<int64_t, 2, torch::RestrictPtrTraits> c_h,
    torch::PackedTensorAccessor64<int64_t, 2, torch::RestrictPtrTraits> c_m_occ,
    int64_t pad,
    int64_t rf,
    int64_t batch_size,
    int64_t out_h,
    int64_t out_w,
    int64_t stride,
    int64_t d,
    torch::PackedTensorAccessor64<scalar_t, 4, torch::RestrictPtrTraits> out
) {
    // thread task:
    // - collaborate loading shared memory
    // - load its own patch and compute output[b,t,i,j]

    // compute thread's indices (b,t,i,j)
    int64_t h = out_h;
    int64_t w = out_w;

    int64_t thread_idx_x = threadIdx.x;
    int64_t thread_idx_y = threadIdx.y;

    int64_t block_idx_x = blockIdx.x; // == tree index
    int64_t block_idx_y = blockIdx.y; // == block index for specific tree

    int64_t glob_thread_idx;
    if (h <= BLOCK_SIZE) {
        glob_thread_idx = block_idx_y*BLOCK_SIZE*BLOCK_SIZE
                        +  BLOCK_SIZE*thread_idx_y + thread_idx_x;
    }
    else {
        int64_t tiles_per_side = (w/BLOCK_SIZE);
        glob_thread_idx = (block_idx_y / tiles_per_side) * (BLOCK_SIZE*w)
                        + thread_idx_y*w + BLOCK_SIZE*(block_idx_y % tiles_per_side) + thread_idx_x;
    }

    int64_t b = glob_thread_idx / (h*w);
    int64_t t = block_idx_x;
    int64_t i = (glob_thread_idx % (h*w)) / w;
    int64_t j = (glob_thread_idx % (h*w)) % w;

    // dynamic load of shared memory [c_m_binary; c_h_; c_w_; weights_tree]
    // first 3*pow2_d bins for c_m_binary; c_h_; c_w_; (int64_t), remaining for weights_tree (scalar_t)
    extern __shared__ int64_t s_m_buffer[];
    __shared__ int64_t ch[2];

    size_t bins_c = static_cast<size_t>(1) << D;

    int64_t* c_m_binary = s_m_buffer;
    int64_t* c_h_ = (int64_t*) &c_m_binary[bins_c];
    int64_t* c_w_ = (int64_t*) &c_h_[bins_c];
    scalar_t* weights_tree = (scalar_t*) &c_w_[bins_c];

    int64_t thread_idx_in_block = thread_idx_y*BLOCK_SIZE + thread_idx_x;
    for (size_t idx = static_cast<size_t>(thread_idx_in_block); idx < 19*bins_c - 16 + 2; idx += BLOCK_SIZE*BLOCK_SIZE) {
        if (idx < bins_c) {
            c_m_binary[idx] = c_m[t][idx] == c_m_occ[t][0] ? 0 : 1;
        } else if (idx < 2*bins_c) {
            c_h_[idx-bins_c] = c_h[t][idx-bins_c];
        } else if (idx < 3*bins_c) {
            c_w_[idx-2*bins_c] = c_w[t][idx-2*bins_c];
        } else if (idx < 19*bins_c - 16) {
            weights_tree[idx-3*bins_c] = weights[t][(idx-3*bins_c)/16][(idx-3*bins_c)%16];
        }
        else {
            ch[idx-(19*bins_c - 16)] = c_m_occ[t][idx-(19*bins_c - 16)];
        }
    }
    __syncthreads();


    // load the needed values to perform BLOCK_SIZE*BLOCK_SIZE output computations
    // if h,w (w=h)
    // < BLOCK_SIZE: load different batches dimensions (BLOCK_SIZE**2 / w*h) => BLOCK_SIZE**2 *2elements
    // = BLOCK_SIZE: load all batch dimension  => BLOCK_SIZE**2 *2elements
    // > BLOCK_SIZE: load portion of batch dimension + contour (padding & values) => (BLOCK_SIZE+2)**2 *2elements

    __shared__ scalar_t x_sub[2*(BLOCK_SIZE+2)*(BLOCK_SIZE+2)];

    if (h <= BLOCK_SIZE) {
        for (int64_t idx = 0; idx < 2; ++idx) {
            x_sub[idx*BLOCK_SIZE*BLOCK_SIZE + thread_idx_in_block] = x[b][ch[idx]][i][j];
        }
    } else { // extract tile of size (BLOCK_SIZE+2)**2 from W x H (for the two channels)
        int64_t tiles_per_side = h/BLOCK_SIZE;
        int64_t tile_in_hw = block_idx_y % (tiles_per_side*tiles_per_side);
        int64_t tile_row = tile_in_hw / tiles_per_side;
        int64_t tile_col = tile_in_hw % tiles_per_side;

        // start of block indices
        int64_t start_i = tile_row*BLOCK_SIZE;
        int64_t start_j = tile_col*BLOCK_SIZE;

        for (int64_t idx = thread_idx_in_block; idx < 2*(BLOCK_SIZE+2)*(BLOCK_SIZE+2); idx += BLOCK_SIZE*BLOCK_SIZE) {
            // thread responsability: channel and local position in (tile + contour)
            int64_t ch_idx = idx / ((BLOCK_SIZE+2)*(BLOCK_SIZE+2));
            int64_t sub_ind = idx % ((BLOCK_SIZE+2)*(BLOCK_SIZE+2));
            // row and col
            int64_t loc_i = sub_ind / (BLOCK_SIZE+2);
            int64_t loc_j = sub_ind % (BLOCK_SIZE+2);

            // location in H x W (block offset+patch offset-relocation)
            int64_t glob_i = start_i + loc_i - 1;
            int64_t glob_j = start_j + loc_j - 1;

            scalar_t value = static_cast<scalar_t>(0);
            if (glob_i>=0 && glob_j>=0 && glob_i<h && glob_j<w) {
                value = x[b][ch[ch_idx]][glob_i][glob_j];
            }
            x_sub[idx] = value;
        }
    }
    __syncthreads();


    // each thread reads its input values from x_sub [2*(BLOCK_SIZE+2 x BLOCK_SIZE+2)]
    const size_t n_inps = static_cast<size_t>(1)<<D;
    scalar_t inp[n_inps];

    if (h <= BLOCK_SIZE) {
        // if h,w (w=h) <= BLOCK_SIZE: read-padding
        int64_t idx_tile_in_block = b % ((BLOCK_SIZE/h)*(BLOCK_SIZE/w));
        int64_t tiles_per_side = BLOCK_SIZE / w;
        int64_t idx_tile_row = idx_tile_in_block / tiles_per_side;
        int64_t idx_tile_col = idx_tile_in_block % tiles_per_side;
        for (size_t idx = 0; idx < n_inps; ++idx) {
            scalar_t val = static_cast<scalar_t>(0);
            int64_t i_ = idx_tile_row*h + i + c_h_[idx]-1; // c_h_ in {0,1,2}
            int64_t j_ = idx_tile_col*w + j + c_w_[idx]-1; // c_w_ in {0,1,2}
            if (i_>=idx_tile_row*h && j_>=idx_tile_col*w && i_<(idx_tile_row+1)*h  && j_<(idx_tile_col+1)*w) {
                // x_sub population:
                // [-- ch0 --, -- ch1 --]
                // each channel [BL_S x BL_S] tiled in blocks [H x W]
                // threads are distributed row-wise at image [H x W] level
                int64_t idx_thread_in_x_sub = (i_/h)*(h*BLOCK_SIZE) + (j_/w)*(h*w) + (i_%h)*w + j_%w;
                val = x_sub[c_m_binary[idx]*BLOCK_SIZE*BLOCK_SIZE + idx_thread_in_x_sub];
            }
            inp[idx] = val;
        }
    } else {
        // x_sub [2*(BLOCK_SIZE+2 x BLOCK_SIZE+2)] is already padded
        // i_, j_ indeces in [H x W]
        for (int64_t idx = 0; idx < n_inps; ++idx) {
            int64_t i_ = thread_idx_y + c_h_[idx];
            int64_t j_ = thread_idx_x + c_w_[idx];
            inp[idx] = x_sub[c_m_binary[idx]*(BLOCK_SIZE+2)*(BLOCK_SIZE+2) + i_*(BLOCK_SIZE+2) + j_];
        }
    }

    // compute output value [b,t,i,j]
    int64_t n_inter = n_inps/2;
    scalar_t intermediate[n_inps/2];
    int64_t consumed = 0;

    for (int64_t idx = 0; idx < n_inter; ++idx,++consumed) {
        scalar_t inp_a = inp[2*idx];
        scalar_t inp_b = inp[2*idx+1];
        scalar_t* inp_w = &weights_tree[16*consumed];
        intermediate[idx] = bin_op_s(inp_a,inp_b,inp_w);
    }

    n_inter /= 2;
    while (n_inter >= 1) {
        for (int64_t idx = 0; idx < n_inter; ++idx,++consumed) {
            scalar_t inp_a = intermediate[2*idx];
            scalar_t inp_b = intermediate[2*idx+1];
            scalar_t* inp_w = &weights_tree[16*consumed];
            intermediate[idx] = bin_op_s(inp_a,inp_b,inp_w);
        }
        n_inter /= 2;
    }

    // write result to global memory
    out[b][t][i][j] = intermediate[0];
}


template <typename scalar_t>
__device__ __forceinline__ scalar_t bin_op_s(scalar_t a, scalar_t b, scalar_t* w) {
    // | id | Operator             | AB=00 | AB=01 | AB=10 | AB=11 |
    // |----|----------------------|-------|-------|-------|-------|
    // | 0  | 0                    | 0     | 0     | 0     | 0     |
    // | 1  | A and B              | 0     | 0     | 0     | 1     |
    // | 2  | not(A implies B)     | 0     | 0     | 1     | 0     |
    // | 3  | A                    | 0     | 0     | 1     | 1     |
    // | 4  | not(B implies A)     | 0     | 1     | 0     | 0     |
    // | 5  | B                    | 0     | 1     | 0     | 1     |
    // | 6  | A xor B              | 0     | 1     | 1     | 0     |
    // | 7  | A or B               | 0     | 1     | 1     | 1     |
    // | 8  | not(A or B)          | 1     | 0     | 0     | 0     |
    // | 9  | not(A xor B)         | 1     | 0     | 0     | 1     |
    // | 10 | not(B)               | 1     | 0     | 1     | 0     |
    // | 11 | B implies A          | 1     | 0     | 1     | 1     |
    // | 12 | not(A)               | 1     | 1     | 0     | 0     |
    // | 13 | A implies B          | 1     | 1     | 0     | 1     |
    // | 14 | not(A and B)         | 1     | 1     | 1     | 0     |
    // | 15 | 1                    | 1     | 1     | 1     | 1     |

    return  (
          (a*b)*w[1]
        + (a-a*b)*w[2]
        + (a)*w[3]
        + (b-a*b)*w[4]
        + (b)*w[5]
        + (a+b-static_cast<scalar_t>(2)*a*b)*w[6]
        + (a+b-a*b)*w[7]
        + (static_cast<scalar_t>(1)-(a+b-a*b))*w[8]
        + (static_cast<scalar_t>(1)-(a+b-static_cast<scalar_t>(2)*a*b))*w[9]
        + (static_cast<scalar_t>(1)-b)*w[10]
        + (static_cast<scalar_t>(1)-b+a*b)*w[11]
        + (static_cast<scalar_t>(1)-a)*w[12]
        + (static_cast<scalar_t>(1)-a+a*b)*w[13]
        + (static_cast<scalar_t>(1)-a*b)*w[14]
        + w[15]
    );
}


// backward kernel w.r.t weights invocation
torch::Tensor conv_backward_w(
    torch::Tensor x,
    torch::Tensor weights,
    torch::Tensor c_m,
    torch::Tensor c_h,
    torch::Tensor c_w,
    torch::Tensor c_m_occ,
    int64_t pad,
    int64_t rf,
    int64_t stride,
    int64_t d,
    torch::Tensor grad_y
) {
    assertionCheck(
    (pad == PAD && rf == RF && stride == STRIDE && d == D),
    "Not implemented for pad != 1 or rf != 3x3 or stride != 1."
    );

    CHECK_INPUT(x); CHECK_INPUT(weights); CHECK_INPUT(grad_y);
    CHECK_INPUT(c_m); CHECK_INPUT(c_h); CHECK_INPUT(c_w); CHECK_INPUT(c_m_occ);

    // # x of shape [batch_size, channels, height, width]; not padded yet
    assertionCheck(x.dim() == 4, "Incorrect input dimensions.");
    int64_t batch_size = x.size(0);
    int64_t n_chs = x.size(1);
    int64_t h = x.size(2);
    int64_t w = x.size(3);
    assertionCheck(h==w && h%4==0, "Not implemented for input h, w non multiples of 4.");

    int64_t n_trees = c_m_occ.size(0);

    int64_t out_h = h;
    int64_t out_w = w;

    // define dimBlock and dimGrid
    dim3 dimBlock(BLOCK_SIZE, BLOCK_SIZE);

    int64_t n_blocks_per_tree =  batch_size*out_h*out_w/(BLOCK_SIZE*BLOCK_SIZE);
    // if second condition not present the following setting would be allowed:
    // batch_size, w=h, BLOCK_SIZE = 49, 3, 7
    assertionCheck(
        ((batch_size*out_h*out_w) % (BLOCK_SIZE*BLOCK_SIZE) == 0 &&
        out_h <= batch_size ? batch_size%h==0 : h%batch_size==0 &&
        n_trees <= 65535 && n_blocks_per_tree <= 65535),
        "Combination of batch size, input dimensions (h,w), block size not allowed."
    );
    dim3 dimGrid (n_trees, n_blocks_per_tree);

    // store 4 types of derivatives
    torch::Tensor grad_w_4 = torch::empty(
        {n_trees, n_blocks_per_tree, (static_cast<int64_t>(1)<<D)-1, 4},
        torch::dtype(x.dtype()).device(x.device())
    );

    // invoke kernel
    AT_DISPATCH_FLOATING_TYPES_AND_HALF(x.scalar_type(), "conv_backward_w", ([&] {
        size_t pow2_d = static_cast<size_t>(1) << D;
        size_t mem_c_m = (pow2_d) * sizeof(int64_t);
        size_t mem_c_h = (pow2_d) * sizeof(int64_t);
        size_t mem_c_w = (pow2_d) * sizeof(int64_t);
        size_t mem_weights = (pow2_d -1)*16 * sizeof(scalar_t);
        size_t s_mem_size = mem_c_m+mem_c_h+mem_c_w + mem_weights;

        conv_backward_w_kernel<scalar_t><<<dimGrid, dimBlock, s_mem_size>>>(
            x.packed_accessor64<scalar_t, 4, torch::RestrictPtrTraits>(),
            weights.packed_accessor64<scalar_t, 3, torch::RestrictPtrTraits>(),
            c_m.packed_accessor64<int64_t, 2, torch::RestrictPtrTraits>(),
            c_w.packed_accessor64<int64_t, 2, torch::RestrictPtrTraits>(),
            c_h.packed_accessor64<int64_t, 2, torch::RestrictPtrTraits>(),
            c_m_occ.packed_accessor64<int64_t, 2, torch::RestrictPtrTraits>(),
            pad,
            rf,
            batch_size,
            out_h,
            out_w,
            stride,
            d,
            grad_y.packed_accessor64<scalar_t, 4, torch::RestrictPtrTraits>(),
            grad_w_4.packed_accessor64<scalar_t, 4, torch::RestrictPtrTraits>()
        );
    }));

    // check errors during kernel execution
    gpuErrchk(cudaPeekAtLastError());
    // synchronize device == wait until all computations on device are finshed
    gpuErrchk(cudaDeviceSynchronize());

    // sum over the batch size the components of grad_w_4
    torch::Tensor grad_w_comp = grad_w_4.sum(1);

    torch::Tensor grad_w_ab = grad_w_comp.index({Slice(), Slice(), 0});
    torch::Tensor grad_w_a = grad_w_comp.index({Slice(), Slice(), 1});
    torch::Tensor grad_w_b = grad_w_comp.index({Slice(), Slice(), 2});
    torch::Tensor grad_w_1 = grad_w_comp.index({Slice(), Slice(), 3});

    return torch::stack({
        torch::zeros(
            {n_trees, (static_cast<int64_t>(1)<<D)-1},
            torch::dtype(x.dtype()).device(x.device())
        ),
        grad_w_ab,
        grad_w_a-grad_w_ab,
        grad_w_a,
        grad_w_b-grad_w_ab,
        grad_w_b,
        grad_w_a+grad_w_b-grad_w_ab-grad_w_ab,
        grad_w_a+grad_w_b-grad_w_ab,
        grad_w_1-grad_w_a-grad_w_b+grad_w_ab,
        grad_w_1-grad_w_a-grad_w_b+grad_w_ab+grad_w_ab,
        grad_w_1-grad_w_b,
        grad_w_1-grad_w_b+grad_w_ab,
        grad_w_1-grad_w_a,
        grad_w_1-grad_w_a+grad_w_ab,
        grad_w_1-grad_w_ab,
        grad_w_1

    }, 2);
}


// backward kernel w.r.t weights
template <typename scalar_t>
__global__ void conv_backward_w_kernel(
    torch::PackedTensorAccessor64<scalar_t, 4, torch::RestrictPtrTraits> x,
    torch::PackedTensorAccessor64<scalar_t, 3, torch::RestrictPtrTraits> weights,
    torch::PackedTensorAccessor64<int64_t, 2, torch::RestrictPtrTraits> c_m,
    torch::PackedTensorAccessor64<int64_t, 2, torch::RestrictPtrTraits> c_w,
    torch::PackedTensorAccessor64<int64_t, 2, torch::RestrictPtrTraits> c_h,
    torch::PackedTensorAccessor64<int64_t, 2, torch::RestrictPtrTraits> c_m_occ,
    int64_t pad,
    int64_t rf,
    int64_t batch_size,
    int64_t out_h,
    int64_t out_w,
    int64_t stride,
    int64_t d,
    torch::PackedTensorAccessor64<scalar_t, 4, torch::RestrictPtrTraits> grad_y,
    torch::PackedTensorAccessor64<scalar_t, 4, torch::RestrictPtrTraits> grad_w_4
) {
    // ---------------- START FORWARD ----------------

    // thread task:
    // - collaborate loading shared memory
    // - load its own patch and compute output[b,t,i,j]

    // compute thread's indices (b,t,i,j)
    int64_t h = out_h;
    int64_t w = out_w;

    int64_t thread_idx_x = threadIdx.x;
    int64_t thread_idx_y = threadIdx.y;

    int64_t block_idx_x = blockIdx.x; // == tree index
    int64_t block_idx_y = blockIdx.y; // == block index for specific tree

    int64_t glob_thread_idx;
    if (h <= BLOCK_SIZE) {
        glob_thread_idx = block_idx_y*BLOCK_SIZE*BLOCK_SIZE
                        +  BLOCK_SIZE*thread_idx_y + thread_idx_x;
    }
    else {
        int64_t tiles_per_side = (w/BLOCK_SIZE);
        glob_thread_idx = (block_idx_y / tiles_per_side) * (BLOCK_SIZE*w)
                        + thread_idx_y*w + BLOCK_SIZE*(block_idx_y % tiles_per_side) + thread_idx_x;
    }

    int64_t b = glob_thread_idx / (h*w);
    int64_t t = block_idx_x;
    int64_t i = (glob_thread_idx % (h*w)) / w;
    int64_t j = (glob_thread_idx % (h*w)) % w;

    // dynamic load of shared memory [c_m_binary; c_h_; c_w_; weights_tree]
    // first 3*pow2_d bins for c_m_binary; c_h_; c_w_; (int64_t), remaining for weights_tree (scalar_t)
    extern __shared__ int64_t s_m_buffer[];
    __shared__ int64_t ch[2];

    size_t bins_c = static_cast<size_t>(1) << D;

    int64_t* c_m_binary = s_m_buffer;
    int64_t* c_h_ = (int64_t*) &c_m_binary[bins_c];
    int64_t* c_w_ = (int64_t*) &c_h_[bins_c];
    scalar_t* weights_tree = (scalar_t*) &c_w_[bins_c];

    int64_t thread_idx_in_block = thread_idx_y*BLOCK_SIZE + thread_idx_x;
    for (size_t idx = static_cast<size_t>(thread_idx_in_block); idx < 19*bins_c - 16 + 2; idx += BLOCK_SIZE*BLOCK_SIZE) {
        if (idx < bins_c) {
            c_m_binary[idx] = c_m[t][idx] == c_m_occ[t][0] ? 0 : 1;
        } else if (idx < 2*bins_c) {
            c_h_[idx-bins_c] = c_h[t][idx-bins_c];
        } else if (idx < 3*bins_c) {
            c_w_[idx-2*bins_c] = c_w[t][idx-2*bins_c];
        } else if (idx < 19*bins_c - 16) {
            weights_tree[idx-3*bins_c] = weights[t][(idx-3*bins_c)/16][(idx-3*bins_c)%16];
        }
        else {
            ch[idx-(19*bins_c - 16)] = c_m_occ[t][idx-(19*bins_c - 16)];
        }
    }
    __syncthreads();


    // load the needed values to perform BLOCK_SIZE*BLOCK_SIZE output computations
    // if h,w (w=h)
    // < BLOCK_SIZE: load different batches dimensions (BLOCK_SIZE**2 / w*h) => BLOCK_SIZE**2 *2elements
    // = BLOCK_SIZE: load all batch dimension  => BLOCK_SIZE**2 *2elements
    // > BLOCK_SIZE: load portion of batch dimension + contour (padding & values) => (BLOCK_SIZE+2)**2 *2elements

    __shared__ scalar_t x_sub[2*(BLOCK_SIZE+2)*(BLOCK_SIZE+2)];

    if (h <= BLOCK_SIZE) {
        for (int64_t idx = 0; idx < 2; ++idx) {
            x_sub[idx*BLOCK_SIZE*BLOCK_SIZE + thread_idx_in_block] = x[b][ch[idx]][i][j];
        }
    } else { // extract tile of size (BLOCK_SIZE+2)**2 from W x H (for the two channels)
        int64_t tiles_per_side = h/BLOCK_SIZE;
        int64_t tile_in_hw = block_idx_y % (tiles_per_side*tiles_per_side);
        int64_t tile_row = tile_in_hw / tiles_per_side;
        int64_t tile_col = tile_in_hw % tiles_per_side;

        // start of block indices
        int64_t start_i = tile_row*BLOCK_SIZE;
        int64_t start_j = tile_col*BLOCK_SIZE;

        for (int64_t idx = thread_idx_in_block; idx < 2*(BLOCK_SIZE+2)*(BLOCK_SIZE+2); idx += BLOCK_SIZE*BLOCK_SIZE) {
            // thread responsability: channel and local position in (tile + contour)
            int64_t ch_idx = idx / ((BLOCK_SIZE+2)*(BLOCK_SIZE+2));
            int64_t sub_ind = idx % ((BLOCK_SIZE+2)*(BLOCK_SIZE+2));
            // row and col
            int64_t loc_i = sub_ind / (BLOCK_SIZE+2);
            int64_t loc_j = sub_ind % (BLOCK_SIZE+2);

            // location in H x W (block offset+patch offset-relocation)
            int64_t glob_i = start_i + loc_i - 1;
            int64_t glob_j = start_j + loc_j - 1;

            scalar_t value = static_cast<scalar_t>(0);
            if (glob_i>=0 && glob_j>=0 && glob_i<h && glob_j<w) {
                value = x[b][ch[ch_idx]][glob_i][glob_j];
            }
            x_sub[idx] = value;
        }
    }
    __syncthreads();


    // each thread reads its input values from x_sub [2*(BLOCK_SIZE+2 x BLOCK_SIZE+2)]
    const size_t n_inps = static_cast<size_t>(1)<<D;
    scalar_t inp[n_inps];
    // TODO: inputs stored in z are only used in conv_backward_x_kernel
    scalar_t z[static_cast<size_t>(1)<<(D+1)-1];

    if (h <= BLOCK_SIZE) {
        // if h,w (w=h) <= BLOCK_SIZE: read-padding
        int64_t idx_tile_in_block = b % ((BLOCK_SIZE/h)*(BLOCK_SIZE/w));
        int64_t tiles_per_side = BLOCK_SIZE / w;
        int64_t idx_tile_row = idx_tile_in_block / tiles_per_side;
        int64_t idx_tile_col = idx_tile_in_block % tiles_per_side;
        for (size_t idx = 0; idx < n_inps; ++idx) {
            scalar_t val = static_cast<scalar_t>(0);
            int64_t i_ = idx_tile_row*h + i + c_h_[idx]-1; // c_h_ in {0,1,2}
            int64_t j_ = idx_tile_col*w + j + c_w_[idx]-1; // c_w_ in {0,1,2}
            if (i_>=idx_tile_row*h && j_>=idx_tile_col*w && i_<(idx_tile_row+1)*h  && j_<(idx_tile_col+1)*w) {
                // x_sub population:
                // [-- ch0 --, -- ch1 --]
                // each channel [BL_S x BL_S] tiled in blocks [H x W]
                // threads are distributed row-wise at image [H x W] level
                int64_t idx_thread_in_x_sub = (i_/h)*(h*BLOCK_SIZE) + (j_/w)*(h*w) + (i_%h)*w + j_%w;
                val = x_sub[c_m_binary[idx]*BLOCK_SIZE*BLOCK_SIZE + idx_thread_in_x_sub];
            }
            inp[idx] = val;
            z[idx] = inp[idx];
        }
    } else {
        // x_sub [2*(BLOCK_SIZE+2 x BLOCK_SIZE+2)] is already padded
        // i_, j_ indeces in [H x W]
        for (int64_t idx = 0; idx < n_inps; ++idx) {
            int64_t i_ = thread_idx_y + c_h_[idx];
            int64_t j_ = thread_idx_x + c_w_[idx];
            inp[idx] = x_sub[c_m_binary[idx]*(BLOCK_SIZE+2)*(BLOCK_SIZE+2) + i_*(BLOCK_SIZE+2) + j_];
            z[idx] = inp[idx];
        }
    }

    // compute output value [b,t,i,j]
    int64_t n_inter = n_inps/2;
    scalar_t intermediate[n_inps/2];
    int64_t consumed = 0;

    for (int64_t idx = 0; idx < n_inter; ++idx,++consumed) {
        scalar_t inp_a = inp[2*idx];
        scalar_t inp_b = inp[2*idx+1];
        scalar_t* inp_w = &weights_tree[16*consumed];
        intermediate[idx] = bin_op_s(inp_a,inp_b,inp_w);
        z[n_inps+consumed] = intermediate[idx];
    }

    n_inter /= 2;
    while (n_inter >= 1) {
        for (int64_t idx = 0; idx < n_inter; ++idx,++consumed) {
            scalar_t inp_a = intermediate[2*idx];
            scalar_t inp_b = intermediate[2*idx+1];
            scalar_t* inp_w = &weights_tree[16*consumed];
            intermediate[idx] = bin_op_s(inp_a,inp_b,inp_w);
            z[n_inps+consumed] = intermediate[idx];
        }
        n_inter /= 2;
    }

    // ---------------- END FORWARD ----------------
    __syncthreads();

    // each thread computes its dy/dz = [dy/dz0, ..., dy/dz6]
    const int64_t n_dy_dz = (static_cast<int64_t>(1)<<D)-1;
    scalar_t dy_dz[n_dy_dz];
    dy_dz[n_dy_dz-1] = 1;

    // z = [inp[0], ..., inp[7], z0, ..., z5, z_6 = y]
    int64_t z_idx_parent = n_inps + n_dy_dz-1;
    int64_t z_idx_child_a_offset = 2;

    for (int64_t idx = n_dy_dz-2; idx >= 0; --z_idx_parent,++z_idx_child_a_offset) {
        int64_t idx_a = z_idx_parent-z_idx_child_a_offset;
        int64_t idx_b = z_idx_parent-z_idx_child_a_offset+1;

        dy_dz[idx] = partial_wrt_in(z[idx_a], z[idx_b], &weights_tree[16*(z_idx_parent-n_inps)], false);
        dy_dz[idx] *= dy_dz[idx + z_idx_child_a_offset-1];
        --idx;

        dy_dz[idx] = partial_wrt_in(z[idx_a], z[idx_b], &weights_tree[16*(z_idx_parent-n_inps)], true);
        dy_dz[idx] *= dy_dz[idx + z_idx_child_a_offset];
        --idx;
    }

    // compute dL/dwi information components, i in {0,...,6}
    int64_t w_idx = (static_cast<int64_t>(1)<<D)-2;
    int64_t w_idx_child_a_offset = 2;
    scalar_t dL_dy = grad_y[b][t][i][j];

    __shared__ scalar_t s_buff_4_w4[BLOCK_SIZE*BLOCK_SIZE];

    while(w_idx >= 0) {
        scalar_t dL_dz = dL_dy*dy_dz[w_idx];
        int64_t idx_a = n_inps + w_idx - w_idx_child_a_offset;
        int64_t idx_b = n_inps + w_idx - w_idx_child_a_offset + 1;

        partial_wrt_w4(
            s_buff_4_w4,
            dL_dz,
            z[idx_a],
            z[idx_b],
            &weights_tree[16*w_idx],
            grad_w_4,
            thread_idx_in_block,
            t,
            block_idx_y,
            w_idx
        );
        --w_idx; ++w_idx_child_a_offset;
    }
}


template <typename scalar_t>
__device__ __forceinline__ void partial_wrt_w4(
    scalar_t* buffer,
    scalar_t partial_z,
    scalar_t a,
    scalar_t b,
    scalar_t* w,
    torch::PackedTensorAccessor64<scalar_t, 4, torch::RestrictPtrTraits> grad_w_4,
    int64_t thread_idx_in_block,
    int64_t t,
    int64_t block_idx,
    int64_t w_idx
) {
    scalar_t local;
    // 4 types of information components that constitute dL/dw
    for (int64_t idx = 0; idx < 4; ++idx) {
        switch(idx) {
            case 0: local = partial_z * (a*b); break;
            case 1: local = partial_z * (a); break;
            case 2: local = partial_z * (b); break;
            case 3: local = partial_z; break;
        }

        // each thread stores its contribution in shared memory
        buffer[thread_idx_in_block] = local;
        __syncthreads();

        // threads collaborate to sum courrent type contribution of the entire block
        for (int64_t stride = 1; stride < BLOCK_SIZE*BLOCK_SIZE; stride *= 2) {
            __syncthreads();
            if (thread_idx_in_block%(2*stride) == 0 && thread_idx_in_block+stride < BLOCK_SIZE*BLOCK_SIZE) {
                buffer[thread_idx_in_block] += buffer[thread_idx_in_block+stride];
            }
        }
        __syncthreads();

        // write courrent block type contribution to global memory
        if (thread_idx_in_block == 0) grad_w_4[t][block_idx][w_idx][idx] = buffer[0];
        __syncthreads();
    }
}


template <typename scalar_t>
__device__ __forceinline__ scalar_t partial_wrt_in(scalar_t a, scalar_t b, scalar_t* w, bool da) {
    if (da) {
        return (
              (b)*w[1]
            + (static_cast<scalar_t>(1)-b)*w[2]
            + w[3]
            + (-b)*w[4]
            + (static_cast<scalar_t>(1)-static_cast<scalar_t>(2)*b)*w[6]
            + (static_cast<scalar_t>(1)-b)*w[7]
            + (b-static_cast<scalar_t>(1))*w[8]
            + (static_cast<scalar_t>(2)*b-static_cast<scalar_t>(1))*w[9]
            + (b)*w[11]
            - w[12]
            + (b-static_cast<scalar_t>(1))*w[13]
            - (b)*w[14]
        );
    } else {
        return  (
            (a)*w[1]
            - (a)*w[2]
            + (static_cast<scalar_t>(1)-a)*w[4]
            + w[5]
            + (static_cast<scalar_t>(1)-static_cast<scalar_t>(2)*a)*w[6]
            + (static_cast<scalar_t>(1)-a)*w[7]
            + (a-static_cast<scalar_t>(1))*w[8]
            + (static_cast<scalar_t>(2)*a-static_cast<scalar_t>(1))*w[9]
            - w[10]
            + (a-static_cast<scalar_t>(1))*w[11]
            + (a)*w[13]
            - (a)*w[14]
        );
    }
}


// backward kernel w.r.t inputs invocation
torch::Tensor conv_backward_x(
    torch::Tensor x,
    torch::Tensor weights,
    torch::Tensor c_m,
    torch::Tensor c_h,
    torch::Tensor c_w,
    torch::Tensor c_m_occ,
    int64_t pad,
    int64_t rf,
    int64_t stride,
    int64_t d,
    torch::Tensor grad_y
) {
    assertionCheck(
    (pad == PAD && rf == RF && stride == STRIDE && d == D),
    "Not implemented for pad != 1 or rf != 3x3 or stride != 1."
    );

    CHECK_INPUT(x); CHECK_INPUT(weights); CHECK_INPUT(grad_y);
    CHECK_INPUT(c_m); CHECK_INPUT(c_h); CHECK_INPUT(c_w); CHECK_INPUT(c_m_occ);

    // x of shape [batch_size, channels, height, width]; not padded yet
    assertionCheck(x.dim() == 4, "Incorrect input dimensions.");
    int64_t batch_size = x.size(0);
    int64_t n_chs = x.size(1);
    int64_t h = x.size(2);
    int64_t w = x.size(3);
    assertionCheck(h==w && h%4==0, "Not implemented for input h, w non multiples of 4.");

    int64_t n_trees = c_m_occ.size(0);

    int64_t out_h = h;
    int64_t out_w = w;

    // define dimBlock and dimGrid
    dim3 dimBlock(BLOCK_SIZE, BLOCK_SIZE);

    int64_t n_blocks_per_tree =  batch_size*out_h*out_w/(BLOCK_SIZE*BLOCK_SIZE);
    // if second condition not present the following setting would be allowed:
    // batch_size, w=h, BLOCK_SIZE = 49, 3, 7
    assertionCheck(
        ((batch_size*out_h*out_w) % (BLOCK_SIZE*BLOCK_SIZE) == 0 &&
        out_h <= batch_size ? batch_size%h==0 : h%batch_size==0 &&
        n_trees <= 65535 && n_blocks_per_tree <= 65535),
        "Combination of batch size, input dimensions (h,w), block size not allowed."
    );
    dim3 dimGrid (n_trees, n_blocks_per_tree);

    torch::Tensor grad_x_raw = torch::zeros({
        n_trees,
        batch_size,
        2,
        h <= BLOCK_SIZE ? h : h+2*h/BLOCK_SIZE,
        w <= BLOCK_SIZE ? w : w+2*w/BLOCK_SIZE
        },
        torch::dtype(x.dtype()).device(x.device())
    );

    // invoke kernel
    AT_DISPATCH_FLOATING_TYPES_AND_HALF(x.scalar_type(), "conv_backward_x", ([&] {
        size_t pow2_d = static_cast<size_t>(1) << D;
        size_t mem_c_m = (pow2_d) * sizeof(int64_t);
        size_t mem_c_h = (pow2_d) * sizeof(int64_t);
        size_t mem_c_w = (pow2_d) * sizeof(int64_t);
        size_t mem_weights = (pow2_d -1)*16 * sizeof(scalar_t);
        size_t s_mem_size = mem_c_m+mem_c_h+mem_c_w + mem_weights;

        conv_backward_x_kernel<scalar_t><<<dimGrid, dimBlock, s_mem_size>>>(
            x.packed_accessor64<scalar_t, 4, torch::RestrictPtrTraits>(),
            weights.packed_accessor64<scalar_t, 3, torch::RestrictPtrTraits>(),
            c_m.packed_accessor64<int64_t, 2, torch::RestrictPtrTraits>(),
            c_w.packed_accessor64<int64_t, 2, torch::RestrictPtrTraits>(),
            c_h.packed_accessor64<int64_t, 2, torch::RestrictPtrTraits>(),
            c_m_occ.packed_accessor64<int64_t, 2, torch::RestrictPtrTraits>(),
            pad,
            rf,
            batch_size,
            out_h,
            out_w,
            stride,
            d,
            grad_y.packed_accessor64<scalar_t, 4, torch::RestrictPtrTraits>(),
            grad_x_raw.packed_accessor64<scalar_t, 5, torch::RestrictPtrTraits>()
        );
    }));

    // check errors during kernel execution
    gpuErrchk(cudaPeekAtLastError());
    // synchronize device == wait until all computations on device are finshed
    gpuErrchk(cudaDeviceSynchronize());

    // grad_x of shape [batch_size, channels, height, width];
    torch::Tensor grad_x = torch::zeros_like(x, torch::dtype(x.dtype()).device(x.device()));

    if (h > BLOCK_SIZE) {
        dim3 dimBlock_reduction(32); // TODO: this choice can be optimized
        dim3 dimGrid_reduction(n_trees, batch_size);
        torch::Tensor grad_x_hw = torch::empty({n_trees, grad_x_raw.size(1), 2, h, w}, torch::dtype(x.dtype()).device(x.device()));

        AT_DISPATCH_FLOATING_TYPES_AND_HALF(grad_x_raw.scalar_type(), "grad_x_reduction", ([&] {
            int64_t tiles_per_side = h/BLOCK_SIZE;
            grad_x_reduction<scalar_t><<<dimGrid_reduction, dimBlock_reduction>>>(
                grad_x_raw.packed_accessor64<scalar_t, 5, torch::RestrictPtrTraits>(),
                c_m_occ.packed_accessor64<int64_t, 2, torch::RestrictPtrTraits>(),
                h,
                w,
                tiles_per_side,
                grad_x_hw.packed_accessor64<scalar_t, 5, torch::RestrictPtrTraits>()
            );
        }));
        // check errors during kernel execution
        gpuErrchk(cudaPeekAtLastError());
        // synchronize device == wait until all computations on device are finshed
        gpuErrchk(cudaDeviceSynchronize());

        grad_x_raw = grad_x_hw;
    }

    // compose the gradient
    for (int64_t t = 0; t < n_trees; ++t) {
        int64_t idx_ch0 = c_m_occ[t][0].item<int64_t>();
        int64_t idx_ch1 = c_m_occ[t][1].item<int64_t>();
        // grad_x of shape                [batch_size, channels, height, width];
        // grad_x_raw of shape [n_trees, batch_size, 2,        height, width];
        grad_x.index({Slice(), idx_ch0, Slice(), Slice()}) += grad_x_raw.index({t, Slice(), 0, Slice(), Slice()});
        if (idx_ch0 != idx_ch1) {
            grad_x.index({Slice(), idx_ch1, Slice(), Slice()}) += grad_x_raw.index({t, Slice(), 1, Slice(), Slice()});
        }
    }
    
    return grad_x;
}


// backward kernel w.r.t inputs
template <typename scalar_t>
__global__ void conv_backward_x_kernel(
    torch::PackedTensorAccessor64<scalar_t, 4, torch::RestrictPtrTraits> x,
    torch::PackedTensorAccessor64<scalar_t, 3, torch::RestrictPtrTraits> weights,
    torch::PackedTensorAccessor64<int64_t, 2, torch::RestrictPtrTraits> c_m,
    torch::PackedTensorAccessor64<int64_t, 2, torch::RestrictPtrTraits> c_w,
    torch::PackedTensorAccessor64<int64_t, 2, torch::RestrictPtrTraits> c_h,
    torch::PackedTensorAccessor64<int64_t, 2, torch::RestrictPtrTraits> c_m_occ,
    int64_t pad,
    int64_t rf,
    int64_t batch_size,
    int64_t out_h,
    int64_t out_w,
    int64_t stride,
    int64_t d,
    torch::PackedTensorAccessor64<scalar_t, 4, torch::RestrictPtrTraits> grad_y,
    torch::PackedTensorAccessor64<scalar_t, 5, torch::RestrictPtrTraits> grad_x_raw
) {
    // ---------------- START FORWARD ----------------

    // thread task:
    // - collaborate loading shared memory
    // - load its own patch and compute output[b,t,i,j]

    // compute thread's indices (b,t,i,j)
    int64_t h = out_h;
    int64_t w = out_w;

    int64_t thread_idx_x = threadIdx.x;
    int64_t thread_idx_y = threadIdx.y;

    int64_t block_idx_x = blockIdx.x; // == tree index
    int64_t block_idx_y = blockIdx.y; // == block index for specific tree

    int64_t glob_thread_idx;
    if (h <= BLOCK_SIZE) {
        glob_thread_idx = block_idx_y*BLOCK_SIZE*BLOCK_SIZE
                        +  BLOCK_SIZE*thread_idx_y + thread_idx_x;
    }
    else {
        int64_t tiles_per_side = (w/BLOCK_SIZE);
        glob_thread_idx = (block_idx_y / tiles_per_side) * (BLOCK_SIZE*w)
                        + thread_idx_y*w + BLOCK_SIZE*(block_idx_y % tiles_per_side) + thread_idx_x;
    }

    int64_t b = glob_thread_idx / (h*w);
    int64_t t = block_idx_x;
    int64_t i = (glob_thread_idx % (h*w)) / w;
    int64_t j = (glob_thread_idx % (h*w)) % w;

    // dynamic load of shared memory [c_m_binary; c_h_; c_w_; weights_tree]
    // first 3*pow2_d bins for c_m_binary; c_h_; c_w_; (int64_t), remaining for weights_tree (scalar_t)
    extern __shared__ int64_t s_m_buffer[];
    __shared__ int64_t ch[2];

    size_t bins_c = static_cast<size_t>(1) << D;

    int64_t* c_m_binary = s_m_buffer;
    int64_t* c_h_ = (int64_t*) &c_m_binary[bins_c];
    int64_t* c_w_ = (int64_t*) &c_h_[bins_c];
    scalar_t* weights_tree = (scalar_t*) &c_w_[bins_c];

    int64_t thread_idx_in_block = thread_idx_y*BLOCK_SIZE + thread_idx_x;
    for (size_t idx = static_cast<size_t>(thread_idx_in_block); idx < 19*bins_c - 16 + 2; idx += BLOCK_SIZE*BLOCK_SIZE) {
        if (idx < bins_c) {
            c_m_binary[idx] = c_m[t][idx] == c_m_occ[t][0] ? 0 : 1;
        } else if (idx < 2*bins_c) {
            c_h_[idx-bins_c] = c_h[t][idx-bins_c];
        } else if (idx < 3*bins_c) {
            c_w_[idx-2*bins_c] = c_w[t][idx-2*bins_c];
        } else if (idx < 19*bins_c - 16) {
            weights_tree[idx-3*bins_c] = weights[t][(idx-3*bins_c)/16][(idx-3*bins_c)%16];
        }
        else {
            ch[idx-(19*bins_c - 16)] = c_m_occ[t][idx-(19*bins_c - 16)];
        }
    }
    __syncthreads();


    // load the needed values to perform BLOCK_SIZE*BLOCK_SIZE output computations
    // if h,w (w=h)
    // < BLOCK_SIZE: load different batches dimensions (BLOCK_SIZE**2 / w*h) => BLOCK_SIZE**2 *2elements
    // = BLOCK_SIZE: load all batch dimension  => BLOCK_SIZE**2 *2elements
    // > BLOCK_SIZE: load portion of batch dimension + contour (padding & values) => (BLOCK_SIZE+2)**2 *2elements

    __shared__ scalar_t x_sub[2*(BLOCK_SIZE+2)*(BLOCK_SIZE+2)];

    if (h <= BLOCK_SIZE) {
        for (int64_t idx = 0; idx < 2; ++idx) {
            x_sub[idx*BLOCK_SIZE*BLOCK_SIZE + thread_idx_in_block] = x[b][ch[idx]][i][j];
        }
    } else { // extract tile of size (BLOCK_SIZE+2)**2 from W x H (for the two channels)
        int64_t tiles_per_side = h/BLOCK_SIZE;
        int64_t tile_in_hw = block_idx_y % (tiles_per_side*tiles_per_side);
        int64_t tile_row = tile_in_hw / tiles_per_side;
        int64_t tile_col = tile_in_hw % tiles_per_side;

        // start of block indices
        int64_t start_i = tile_row*BLOCK_SIZE;
        int64_t start_j = tile_col*BLOCK_SIZE;

        for (int64_t idx = thread_idx_in_block; idx < 2*(BLOCK_SIZE+2)*(BLOCK_SIZE+2); idx += BLOCK_SIZE*BLOCK_SIZE) {
            // thread responsability: channel and local position in (tile + contour)
            int64_t ch_idx = idx / ((BLOCK_SIZE+2)*(BLOCK_SIZE+2));
            int64_t sub_ind = idx % ((BLOCK_SIZE+2)*(BLOCK_SIZE+2));
            // row and col
            int64_t loc_i = sub_ind / (BLOCK_SIZE+2);
            int64_t loc_j = sub_ind % (BLOCK_SIZE+2);

            // location in H x W (block offset+patch offset-relocation)
            int64_t glob_i = start_i + loc_i - 1;
            int64_t glob_j = start_j + loc_j - 1;

            scalar_t value = static_cast<scalar_t>(0);
            if (glob_i>=0 && glob_j>=0 && glob_i<h && glob_j<w) {
                value = x[b][ch[ch_idx]][glob_i][glob_j];
            }
            x_sub[idx] = value;
        }
    }
    __syncthreads();


    // each thread reads its input values from x_sub [2*(BLOCK_SIZE+2 x BLOCK_SIZE+2)]
    const size_t n_inps = static_cast<size_t>(1)<<D;
    scalar_t inp[n_inps];
    // TODO: inputs stored in z are only used in conv_backward_x_kernel
    scalar_t z[static_cast<size_t>(1)<<(D+1)-1];

    if (h <= BLOCK_SIZE) {
        // if h,w (w=h) <= BLOCK_SIZE: read-padding
        int64_t idx_tile_in_block = b % ((BLOCK_SIZE/h)*(BLOCK_SIZE/w));
        int64_t tiles_per_side = BLOCK_SIZE / w;
        int64_t idx_tile_row = idx_tile_in_block / tiles_per_side;
        int64_t idx_tile_col = idx_tile_in_block % tiles_per_side;
        for (size_t idx = 0; idx < n_inps; ++idx) {
            scalar_t val = static_cast<scalar_t>(0);
            int64_t i_ = idx_tile_row*h + i + c_h_[idx]-1; // c_h_ in {0,1,2}
            int64_t j_ = idx_tile_col*w + j + c_w_[idx]-1; // c_w_ in {0,1,2}
            if (i_>=idx_tile_row*h && j_>=idx_tile_col*w && i_<(idx_tile_row+1)*h  && j_<(idx_tile_col+1)*w) {
                // x_sub population:
                // [-- ch0 --, -- ch1 --]
                // each channel [BL_S x BL_S] tiled in blocks [H x W]
                // threads are distributed row-wise at image [H x W] level
                int64_t idx_thread_in_x_sub = (i_/h)*(h*BLOCK_SIZE) + (j_/w)*(h*w) + (i_%h)*w + j_%w;
                val = x_sub[c_m_binary[idx]*BLOCK_SIZE*BLOCK_SIZE + idx_thread_in_x_sub];
            }
            inp[idx] = val;
            z[idx] = inp[idx];
        }
    } else {
        // x_sub [2*(BLOCK_SIZE+2 x BLOCK_SIZE+2)] is already padded
        // i_, j_ indeces in [H x W]
        for (int64_t idx = 0; idx < n_inps; ++idx) {
            int64_t i_ = thread_idx_y + c_h_[idx];
            int64_t j_ = thread_idx_x + c_w_[idx];
            inp[idx] = x_sub[c_m_binary[idx]*(BLOCK_SIZE+2)*(BLOCK_SIZE+2) + i_*(BLOCK_SIZE+2) + j_];
            z[idx] = inp[idx];
        }
    }

    // compute output value [b,t,i,j]
    int64_t n_inter = n_inps/2;
    scalar_t intermediate[n_inps/2];
    int64_t consumed = 0;

    for (int64_t idx = 0; idx < n_inter; ++idx,++consumed) {
        scalar_t inp_a = inp[2*idx];
        scalar_t inp_b = inp[2*idx+1];
        scalar_t* inp_w = &weights_tree[16*consumed];
        intermediate[idx] = bin_op_s(inp_a,inp_b,inp_w);
        z[n_inps+consumed] = intermediate[idx];
    }

    n_inter /= 2;
    while (n_inter >= 1) {
        for (int64_t idx = 0; idx < n_inter; ++idx,++consumed) {
            scalar_t inp_a = intermediate[2*idx];
            scalar_t inp_b = intermediate[2*idx+1];
            scalar_t* inp_w = &weights_tree[16*consumed];
            intermediate[idx] = bin_op_s(inp_a,inp_b,inp_w);
            z[n_inps+consumed] = intermediate[idx];
        }
        n_inter /= 2;
    }

    // ---------------- END FORWARD ----------------
    __syncthreads();

    // allocate shared memory [2*(BLOCK_SIZE+2) x (BLOCK_SIZE+2)] to accumulate [dy/dr0, ..., dy/dr7] of all threads in block
    __shared__ scalar_t sm_grad_x[2*(BLOCK_SIZE+2)*(BLOCK_SIZE+2)];
    for (int64_t idx = thread_idx_in_block; idx < 2*(BLOCK_SIZE+2)*(BLOCK_SIZE+2); idx += BLOCK_SIZE*BLOCK_SIZE) {
        sm_grad_x[idx] = 0;
    }
    __syncthreads();

    // each thread computes intermediate derivatives of dy/dz = [dy/dr0, ..., dy/dr7, dy/dz0, ..., dy/dz6]
    // z = [inp[0]=r0, ..., inp[7]=r7, z0, ..., z5, z_6 = y]
    const int64_t n_dy_dz = (static_cast<int64_t>(1)<<(D+1))-1;
    scalar_t dy_dz[n_dy_dz];
    dy_dz[n_dy_dz-1] = 1;

    int64_t z_idx_parent = n_dy_dz-1;
    int64_t z_idx_child_a_offset = 2;

    for (int64_t idx = n_dy_dz-2; idx > (static_cast<int64_t>(1)<<D)-1; --z_idx_parent,++z_idx_child_a_offset) {
        int64_t idx_a = z_idx_parent-z_idx_child_a_offset;
        int64_t idx_b = z_idx_parent-z_idx_child_a_offset+1;

        dy_dz[idx] = partial_wrt_in(z[idx_a], z[idx_b], &weights_tree[16*(z_idx_parent-n_inps)], false);
        dy_dz[idx] *= dy_dz[idx + z_idx_child_a_offset-1];
        --idx;

        dy_dz[idx] = partial_wrt_in(z[idx_a], z[idx_b], &weights_tree[16*(z_idx_parent-n_inps)], true);
        dy_dz[idx] *= dy_dz[idx + z_idx_child_a_offset];
        --idx;
    }

    // each thread computes input derivatives and saves contribution directly into shared memory
    // note that since all trees are fixed you have no race conditions for accumulation operation

    scalar_t dL_dy = grad_y[b][t][i][j];

    int64_t tiles_per_side;
    int64_t idx_tile_row, idx_tile_col;
    int64_t tile_row, tile_col;
    if (h <= BLOCK_SIZE) {
        int64_t idx_tile_in_block = b % ((BLOCK_SIZE/h)*(BLOCK_SIZE/w));
        tiles_per_side = BLOCK_SIZE / w;
        idx_tile_row = idx_tile_in_block / tiles_per_side;
        idx_tile_col = idx_tile_in_block % tiles_per_side;
    } else {
        tiles_per_side = h/BLOCK_SIZE;
        int64_t tile_in_hw = block_idx_y % (tiles_per_side*tiles_per_side);
        tile_row = tile_in_hw / tiles_per_side;
        tile_col = tile_in_hw % tiles_per_side;
    }

    for (int64_t idx = (static_cast<int64_t>(1)<<D)-1; idx >= 0; --z_idx_parent,++z_idx_child_a_offset) {
        int64_t idx_a = z_idx_parent-z_idx_child_a_offset;
        int64_t idx_b = z_idx_parent-z_idx_child_a_offset+1;

        bool child_a = false;
        do {
            if (h <= BLOCK_SIZE) {
                int64_t i_ = idx_tile_row*h + i + c_h_[idx]-1; // c_h_ in {0,1,2}
                int64_t j_ = idx_tile_col*w + j + c_w_[idx]-1; // c_w_ in {0,1,2}
                scalar_t val = 0;
                if (i_>=idx_tile_row*h && j_>=idx_tile_col*w && i_<(idx_tile_row+1)*h  && j_<(idx_tile_col+1)*w) {
                    dy_dz[idx] = partial_wrt_in(z[idx_a], z[idx_b], &weights_tree[16*(z_idx_parent-n_inps)], child_a);
                    dy_dz[idx] *= dy_dz[idx + z_idx_child_a_offset-(child_a ? 0:1)];
                    val = dL_dy*dy_dz[idx];
                }
                int64_t idx_thread_in_sm_grad_x = (i_/h)*(h*BLOCK_SIZE) + (j_/w)*(h*w) + (i_%h)*w + j_%w;
                if (val != 0) sm_grad_x[c_m_binary[idx]*BLOCK_SIZE*BLOCK_SIZE + idx_thread_in_sm_grad_x] += val;
            } else {
                int64_t glob_i = tile_row*(BLOCK_SIZE+2) + thread_idx_y + c_h_[idx]; // c_h_ in {0,1,2}
                int64_t glob_j = tile_col*(BLOCK_SIZE+2) + thread_idx_x + c_w_[idx]; // c_w_ in {0,1,2}
                scalar_t val = 0;
                if (glob_i>0 && glob_j>0 && glob_i<tiles_per_side*(BLOCK_SIZE+2)-1 && glob_j<tiles_per_side*(BLOCK_SIZE+2)-1) {
                    dy_dz[idx] = partial_wrt_in(z[idx_a], z[idx_b], &weights_tree[16*(z_idx_parent-n_inps)], child_a);
                    dy_dz[idx] *= dy_dz[idx + z_idx_child_a_offset-(child_a ? 0:1)];
                    val = dL_dy*dy_dz[idx];
                }
                int64_t idx_thread_in_sm_grad_x = (thread_idx_y+c_h_[idx])*(BLOCK_SIZE+2) + thread_idx_x+c_w_[idx];
                if (val != 0) sm_grad_x[c_m_binary[idx]*(BLOCK_SIZE+2)*(BLOCK_SIZE+2) + idx_thread_in_sm_grad_x] += val;
            }
            __syncthreads();
            --idx;

            child_a = !child_a;
        } while (child_a);
    }

    // store in global memory
    if (h <= BLOCK_SIZE) {
        for (int64_t idx = 0; idx < 2; ++idx) {
            int64_t i_ = idx_tile_row*h + i;
            int64_t j_ = idx_tile_col*w + j;
            int64_t sm_idx = idx*BLOCK_SIZE*BLOCK_SIZE + (i_/h)*(h*BLOCK_SIZE) + (j_/w)*(h*w) + (i_%h)*w + j_%w;
            grad_x_raw[t][b][idx][i][j] = sm_grad_x[sm_idx];
        }
    } else {
        for (int64_t idx = thread_idx_in_block; idx < 2*(BLOCK_SIZE+2)*(BLOCK_SIZE+2); idx += BLOCK_SIZE*BLOCK_SIZE) {
            int64_t ch_idx = idx / ((BLOCK_SIZE+2)*(BLOCK_SIZE+2));
            int64_t sub_ind = idx % ((BLOCK_SIZE+2)*(BLOCK_SIZE+2));
            // row and col
            int64_t loc_i = sub_ind / (BLOCK_SIZE+2);
            int64_t loc_j = sub_ind % (BLOCK_SIZE+2);

            // location in H x W
            int64_t glob_i = tile_row*(BLOCK_SIZE+2) + loc_i;
            int64_t glob_j = tile_col*(BLOCK_SIZE+2) + loc_j;

            int64_t sm_idx = ch_idx*(BLOCK_SIZE+2)*(BLOCK_SIZE+2) + loc_i*(BLOCK_SIZE+2) + loc_j;
            grad_x_raw[t][b][ch_idx][glob_i][glob_j] += sm_grad_x[sm_idx];
        }
    }
}


// merge contributions of grad_x_raw
template <typename scalar_t>
__global__ void grad_x_reduction(
    torch::PackedTensorAccessor64<scalar_t, 5, torch::RestrictPtrTraits> grad_x_raw,
    torch::PackedTensorAccessor64<int64_t, 2, torch::RestrictPtrTraits> c_m_occ,
    int64_t h,
    int64_t w,
    int64_t tiles_per_side,
    torch::PackedTensorAccessor64<scalar_t, 5, torch::RestrictPtrTraits> grad_x_hw
) {
    int64_t thread_idx_in_block = threadIdx.x;
    int64_t n_threads_in_block = blockDim.x;
    int64_t t = blockIdx.x;
    int64_t b = blockIdx.y;

    int64_t n_lv_op_per_ch = 2*(tiles_per_side-1)*BLOCK_SIZE*tiles_per_side;

    // lateral sum
    for (int64_t idx = thread_idx_in_block; idx < (c_m_occ[t][0]==c_m_occ[t][1] ? 1:2)*n_lv_op_per_ch; idx += n_threads_in_block) {
        int64_t ch_idx = idx / n_lv_op_per_ch;
        int64_t sub_idx = idx % n_lv_op_per_ch;

        int64_t row_tile = sub_idx / (2*BLOCK_SIZE*(tiles_per_side-1));
        int64_t thread_idx_in_row_tile = sub_idx % (2*BLOCK_SIZE*(tiles_per_side-1));
        int64_t row_in_tile = 1 + thread_idx_in_row_tile / (2*(tiles_per_side-1));
        int64_t col_tile = 1 + (thread_idx_in_row_tile % (2*(tiles_per_side-1))) / 2;


        int64_t i = row_tile*(BLOCK_SIZE+2) + row_in_tile;
        int64_t j = col_tile*(BLOCK_SIZE+2)-1 + sub_idx%2;

        grad_x_raw[t][b][ch_idx][i][j+(sub_idx%2==0 ? 2:-2)] += grad_x_raw[t][b][ch_idx][i][j];
    }
    __syncthreads();

    // vertical sum
    for (int64_t idx = thread_idx_in_block; idx < (c_m_occ[t][0]==c_m_occ[t][1] ? 1:2)*n_lv_op_per_ch; idx += n_threads_in_block) {
        int64_t ch_idx = idx / n_lv_op_per_ch;
        int64_t sub_idx = idx % n_lv_op_per_ch;

        int64_t col_tile = sub_idx / (2*BLOCK_SIZE*(tiles_per_side-1));
        int64_t thread_idx_in_col_tile = sub_idx % (2*BLOCK_SIZE*(tiles_per_side-1));
        int64_t col_in_tile = 1 + thread_idx_in_col_tile / (2*(tiles_per_side-1));
        int64_t row_tile = 1 + (thread_idx_in_col_tile % (2*(tiles_per_side-1))) / 2;


        int64_t j = col_tile*(BLOCK_SIZE+2) + col_in_tile;
        int64_t i = row_tile*(BLOCK_SIZE+2)-1 + sub_idx%2;

        grad_x_raw[t][b][ch_idx][i+(sub_idx%2==0 ? 2:-2)][j] += grad_x_raw[t][b][ch_idx][i][j];
    }
    __syncthreads();

    // diagional sum
    int64_t n_d_op_per_ch =  4*(tiles_per_side-1)*(tiles_per_side-1);

    for (int64_t idx = thread_idx_in_block; idx < (c_m_occ[t][0]==c_m_occ[t][1] ? 1:2)*n_d_op_per_ch; idx += n_threads_in_block) {
        int64_t ch_idx = idx / n_d_op_per_ch;
        int64_t sub_idx = idx % n_d_op_per_ch;

        int64_t row_node = sub_idx / (4*(tiles_per_side-1));
        int64_t col_idx_in_node_row = (sub_idx % (4*(tiles_per_side-1))) / 2;

        int64_t i = (1+row_node)*(BLOCK_SIZE+2) + (col_idx_in_node_row<(tiles_per_side-1) ? -1:0);
        int64_t j = (1+ col_idx_in_node_row%(tiles_per_side-1))*(BLOCK_SIZE+2) + (sub_idx%2==0 ? -1:0);

        int64_t i_ = i + (col_idx_in_node_row<(tiles_per_side-1) ? 2:-2);
        int64_t j_ = j + (sub_idx%2==0 ? 2:-2);

        grad_x_raw[t][b][ch_idx][i_][j_] += grad_x_raw[t][b][ch_idx][i][j];   
    }
    __syncthreads();    

    // select region H x W from (n_el x n_el))
    int64_t n_el = tiles_per_side*(BLOCK_SIZE+2);
    for (int64_t idx = thread_idx_in_block; idx < 2*n_el*n_el; idx += n_threads_in_block) {
        int64_t ch_idx = idx / (n_el*n_el);
        int64_t sub_idx = idx % (n_el*n_el);
        
        int64_t i = sub_idx/n_el;
        int64_t j = sub_idx%n_el;

        if (
            i % (BLOCK_SIZE+2) == 0 ||
            i % (BLOCK_SIZE+2) == (BLOCK_SIZE+1) ||
            j % (BLOCK_SIZE+2) == 0 ||
            j % (BLOCK_SIZE+2) == (BLOCK_SIZE+1)
        ) continue;

        int64_t row_tile = i / (BLOCK_SIZE+2);
        int64_t col_tile = j / (BLOCK_SIZE+2);
        grad_x_hw[t][b][ch_idx][i-(2*row_tile)-1][j-(2*col_tile)-1] = grad_x_raw[t][b][ch_idx][i][j];
    }
}