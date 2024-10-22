import torch
import tabulate
import triton
import triton.language as tl
from triton.runtime import driver

#########################################################################################
@triton.jit
def add_kernel(x_ptr, y_ptr, output_ptr, n_elements, BLOCK_SIZE: tl.constexpr):
    pid = tl.program_id(axis=0)
    block_start = pid * BLOCK_SIZE
    offsets = block_start + tl.arange(0, BLOCK_SIZE)
    mask = offsets < n_elements
    x = tl.load(x_ptr + offsets, mask=mask)
    y = tl.load(y_ptr + offsets, mask=mask)
    output = x + y
    tl.store(output_ptr + offsets, output, mask=mask)


def add(x: torch.Tensor, y: torch.Tensor):
    output = torch.empty_like(x)
    assert x.is_cuda and y.is_cuda and output.is_cuda
    n_elements = output.numel()
    grid = lambda meta: (triton.cdiv(n_elements, meta['BLOCK_SIZE']), )
    add_kernel[grid](x, y, output, n_elements, BLOCK_SIZE=1024)
    return output

#########################################################################################
@torch.jit.script
def naive_softmax(x: torch.Tensor, dim: int = -1):
    # read  MN elements ; write M  elements
    x_max = x.max(dim=dim, keepdim=True)[0]
    # read MN + M elements ; write MN elements
    z = x - x_max
    # read  MN elements ; write MN elements
    numerator = torch.exp(z)
    # read  MN elements ; write M  elements
    denominator = numerator.sum(dim=dim, keepdim=True)
    # read MN + M elements ; write MN elements
    ret = numerator / denominator
    # in total: read 5MN + 2M elements ; wrote 3MN + 2M elements
    return ret


@triton.jit
def softmax_kernel(output_ptr, input_ptr, input_row_stride, output_row_stride, n_rows, n_cols, BLOCK_SIZE: tl.constexpr, num_stages: tl.constexpr):
    row_start = tl.program_id(0)
    row_step = tl.num_programs(0)
    for row_idx in tl.range(row_start, n_rows, row_step, num_stages=num_stages):
        row_start_ptr = input_ptr + row_idx * input_row_stride
        col_offsets = tl.arange(0, BLOCK_SIZE)
        input_ptrs = row_start_ptr + col_offsets
        mask = col_offsets < n_cols
        row = tl.load(input_ptrs, mask=mask, other=-float('inf'))
        row_minus_max = row - tl.max(row, axis=0)
        numerator = tl.exp(row_minus_max)
        denominator = tl.sum(numerator, axis=0)
        softmax_output = numerator / denominator
        output_row_start_ptr = output_ptr + row_idx * output_row_stride
        output_ptrs = output_row_start_ptr + col_offsets
        tl.store(output_ptrs, softmax_output, mask=mask)


kernels = {}

def softmax(x):
    n_rows, n_cols = x.shape
    BLOCK_SIZE = triton.next_power_of_2(n_cols)
    num_warps = 8
    num_stages = 2


    y = torch.empty_like(x)
    # pre-compile kernel to get register usage and compute thread occupancy.
    kernel, num_programs = kernels.get(BLOCK_SIZE, (None, 0))
    if kernel is None:
        device = torch.cuda.current_device()
        properties = driver.active.utils.get_device_properties(device)
        NUM_SM = properties["multiprocessor_count"]
        NUM_REGS = properties["max_num_regs"]
        SIZE_SMEM = properties["max_shared_mem"]
        WARP_SIZE = properties["warpSize"]
        kernel = softmax_kernel.warmup(y, x, x.stride(0), y.stride(0), n_rows, n_cols, BLOCK_SIZE=BLOCK_SIZE, num_stages=num_stages, num_warps=num_warps, grid=(1, ))
        kernel._init_handles()
        n_regs = kernel.n_regs
        size_smem = kernel.metadata.shared
        occupancy = NUM_REGS // (n_regs * WARP_SIZE * num_warps)
        occupancy = min(occupancy, SIZE_SMEM // size_smem)
        num_programs = NUM_SM * occupancy
        kernels[BLOCK_SIZE] = (kernel, num_programs)

    num_programs = min(num_programs, n_rows)

    # Create a number of persistent programs.
    kernel[(num_programs, 1, 1)](y, x, x.stride(0), y.stride(0), n_rows, n_cols)
    return y

#########################################################################################
def get_cuda_autotune_config():
    return [
        triton.Config({'BLOCK_SIZE_M': 128, 'BLOCK_SIZE_N': 256,    'BLOCK_SIZE_K': 64,     'GROUP_SIZE_M': 8}, num_stages=3,   num_warps=8),
        triton.Config({'BLOCK_SIZE_M': 64,  'BLOCK_SIZE_N': 256,    'BLOCK_SIZE_K': 32,     'GROUP_SIZE_M': 8}, num_stages=4,   num_warps=4),
        triton.Config({'BLOCK_SIZE_M': 128, 'BLOCK_SIZE_N': 128,    'BLOCK_SIZE_K': 32,     'GROUP_SIZE_M': 8}, num_stages=4,   num_warps=4),
        triton.Config({'BLOCK_SIZE_M': 128, 'BLOCK_SIZE_N': 64,     'BLOCK_SIZE_K': 32,     'GROUP_SIZE_M': 8}, num_stages=4,   num_warps=4),
        triton.Config({'BLOCK_SIZE_M': 64,  'BLOCK_SIZE_N': 128,    'BLOCK_SIZE_K': 32,     'GROUP_SIZE_M': 8}, num_stages=4,   num_warps=4),
        triton.Config({'BLOCK_SIZE_M': 128, 'BLOCK_SIZE_N': 32,     'BLOCK_SIZE_K': 32,     'GROUP_SIZE_M': 8}, num_stages=4,   num_warps=4),
        triton.Config({'BLOCK_SIZE_M': 64,  'BLOCK_SIZE_N': 32,     'BLOCK_SIZE_K': 32,     'GROUP_SIZE_M': 8}, num_stages=5,   num_warps=2),
        triton.Config({'BLOCK_SIZE_M': 32,  'BLOCK_SIZE_N': 64,     'BLOCK_SIZE_K': 32,     'GROUP_SIZE_M': 8}, num_stages=5,   num_warps=2),
        # Good config for fp8 inputs.
        triton.Config({'BLOCK_SIZE_M': 128, 'BLOCK_SIZE_N': 256,    'BLOCK_SIZE_K': 128,    'GROUP_SIZE_M': 8}, num_stages=3,   num_warps=8),
        triton.Config({'BLOCK_SIZE_M': 256, 'BLOCK_SIZE_N': 128,    'BLOCK_SIZE_K': 128,    'GROUP_SIZE_M': 8}, num_stages=3,   num_warps=8),
        triton.Config({'BLOCK_SIZE_M': 256, 'BLOCK_SIZE_N': 64,     'BLOCK_SIZE_K': 128,    'GROUP_SIZE_M': 8}, num_stages=4,   num_warps=4),
        triton.Config({'BLOCK_SIZE_M': 64,  'BLOCK_SIZE_N': 256,    'BLOCK_SIZE_K': 128,    'GROUP_SIZE_M': 8}, num_stages=4,   num_warps=4),
        triton.Config({'BLOCK_SIZE_M': 128, 'BLOCK_SIZE_N': 128,    'BLOCK_SIZE_K': 128,    'GROUP_SIZE_M': 8}, num_stages=4,   num_warps=4),
        triton.Config({'BLOCK_SIZE_M': 128, 'BLOCK_SIZE_N': 64,     'BLOCK_SIZE_K': 64,     'GROUP_SIZE_M': 8}, num_stages=4,   num_warps=4),
        triton.Config({'BLOCK_SIZE_M': 64,  'BLOCK_SIZE_N': 128,    'BLOCK_SIZE_K': 64,     'GROUP_SIZE_M': 8}, num_stages=4,   num_warps=4),
        triton.Config({'BLOCK_SIZE_M': 128, 'BLOCK_SIZE_N': 32,     'BLOCK_SIZE_K': 64,     'GROUP_SIZE_M': 8}, num_stages=4,   num_warps=4)
    ]


@triton.jit
def leaky_relu(x):
    return tl.where(x >= 0, x, 0.01 * x)


@triton.autotune(configs=get_cuda_autotune_config(), key=['M', 'N', 'K'])
@triton.jit
def matmul_kernel(a_ptr, b_ptr, c_ptr, M, N, K, stride_am, stride_ak, stride_bk, stride_bn, stride_cm, stride_cn, BLOCK_SIZE_M: tl.constexpr, BLOCK_SIZE_N: tl.constexpr, BLOCK_SIZE_K: tl.constexpr, GROUP_SIZE_M: tl.constexpr, ACTIVATION: tl.constexpr):
    pid = tl.program_id(axis=0)
    num_pid_m = tl.cdiv(M, BLOCK_SIZE_M)
    num_pid_n = tl.cdiv(N, BLOCK_SIZE_N)
    num_pid_in_group = GROUP_SIZE_M * num_pid_n
    group_id = pid // num_pid_in_group
    first_pid_m = group_id * GROUP_SIZE_M
    group_size_m = min(num_pid_m - first_pid_m, GROUP_SIZE_M)

    # -----------------------------------------------------------
    # Block Idx
    pid_m = first_pid_m + ((pid % num_pid_in_group) % group_size_m)
    pid_n = (pid % num_pid_in_group) // group_size_m

    # -----------------------------------------------------------
    # Thread Idx
    offs_am = (pid_m * BLOCK_SIZE_M + tl.arange(0, BLOCK_SIZE_M)) % M
    offs_bn = (pid_n * BLOCK_SIZE_N + tl.arange(0, BLOCK_SIZE_N)) % N
    offs_k = tl.arange(0, BLOCK_SIZE_K)
    a_ptrs = a_ptr + (offs_am[:, None] * stride_am + offs_k[None, :] * stride_ak)
    b_ptrs = b_ptr + (offs_k[:, None] * stride_bk + offs_bn[None, :] * stride_bn)

    # -----------------------------------------------------------
    # Iterate to compute a block of the C matrix.
    # We accumulate into a `[BLOCK_SIZE_M, BLOCK_SIZE_N]` block
    # of fp32 values for higher accuracy.
    # `accumulator` will be converted back to fp16 after the loop.
    accumulator = tl.zeros((BLOCK_SIZE_M, BLOCK_SIZE_N), dtype=tl.float32)
    for k in range(0, tl.cdiv(K, BLOCK_SIZE_K)):
        # Load the next block of A and B, generate a mask by checking the K dimension.
        # If it is out of bounds, set it to 0.
        a = tl.load(a_ptrs, mask=offs_k[None, :] < K - k * BLOCK_SIZE_K, other=0.0)
        b = tl.load(b_ptrs, mask=offs_k[:, None] < K - k * BLOCK_SIZE_K, other=0.0)
        # We accumulate along the K dimension.
        accumulator = tl.dot(a, b, accumulator)
        # Advance the ptrs to the next K block.
        a_ptrs += BLOCK_SIZE_K * stride_ak
        b_ptrs += BLOCK_SIZE_K * stride_bk
    if ACTIVATION == "leaky_relu":
        accumulator = leaky_relu(accumulator)
    c = accumulator.to(tl.float16)

    # -----------------------------------------------------------
    # Write back the block of the output matrix C with masks.
    offs_cm = pid_m * BLOCK_SIZE_M + tl.arange(0, BLOCK_SIZE_M)
    offs_cn = pid_n * BLOCK_SIZE_N + tl.arange(0, BLOCK_SIZE_N)
    c_ptrs = c_ptr + stride_cm * offs_cm[:, None] + stride_cn * offs_cn[None, :]
    c_mask = (offs_cm[:, None] < M) & (offs_cn[None, :] < N)
    tl.store(c_ptrs, c, mask=c_mask)


def matmul(a, b, activation=""):
    # Check constraints.
    assert a.shape[1] == b.shape[0], "Incompatible dimensions"
    assert a.is_contiguous(), "Matrix A must be contiguous"
    M, K = a.shape
    K, N = b.shape
    # Allocates output.
    c = torch.empty((M, N), device=a.device, dtype=torch.float16)
    # 1D launch kernel where each block gets its own program.
    grid = lambda META: (triton.cdiv(M, META['BLOCK_SIZE_M']) * triton.cdiv(N, META['BLOCK_SIZE_N']), )
    matmul_kernel[grid](
        a, b, c,  #
        M, N, K,  #
        a.stride(0), a.stride(1),  #
        b.stride(0), b.stride(1),  #
        c.stride(0), c.stride(1),  #
        ACTIVATION=activation  #
    )
    return c

#########################################################################################
@triton.jit
def _dropout_1d(x_ptr, x_keep_ptr, output_ptr, n_elements, p, BLOCK_SIZE: tl.constexpr):
    pid = tl.program_id(axis=0)
    block_start = pid * BLOCK_SIZE
    offsets = block_start + tl.arange(0, BLOCK_SIZE)
    mask = offsets < n_elements
    x = tl.load(x_ptr + offsets, mask=mask)
    x_keep = tl.load(x_keep_ptr + offsets, mask=mask)
    output = tl.where(x_keep, x / (1 - p), 0.0)
    tl.store(output_ptr + offsets, output, mask=mask)


def dropout_1d(x, x_keep, p):
    output = torch.empty_like(x)
    assert x.is_contiguous()
    n_elements = x.numel()
    grid = lambda meta: (triton.cdiv(n_elements, meta['BLOCK_SIZE']), )
    _dropout_1d[grid](x, x_keep, output, n_elements, p, BLOCK_SIZE=1024)
    return output

@triton.jit
def _seeded_dropout_1d(x_ptr, output_ptr, n_elements, p, seed, BLOCK_SIZE: tl.constexpr):
    pid = tl.program_id(axis=0)
    block_start = pid * BLOCK_SIZE
    offsets = block_start + tl.arange(0, BLOCK_SIZE)
    mask = offsets < n_elements
    x = tl.load(x_ptr + offsets, mask=mask)
    random = tl.rand(seed, offsets)
    x_keep = random > p
    output = tl.where(x_keep, x / (1 - p), 0.0)
    tl.store(output_ptr + offsets, output, mask=mask)


def seeded_dropout_1d(x, p, seed):
    output = torch.empty_like(x)
    assert x.is_contiguous()
    n_elements = x.numel()
    grid = lambda meta: (triton.cdiv(n_elements, meta['BLOCK_SIZE']), )
    _seeded_dropout_1d[grid](x, output, n_elements, p, seed, BLOCK_SIZE=1024)
    return output

#########################################################################################
@triton.jit
def _seeded_dropout_2d(x_ptr, output_ptr, seed_ptr, M, N, p, x_row_stride, output_row_stride, BLOCK_SIZE: tl.constexpr):
    row_start = tl.program_id(0)
    row_step = tl.num_programs(0)
    for row_idx in tl.range(row_start, M, row_step):
        row_start_ptr = x_ptr + row_idx * x_row_stride
        col_offsets = tl.arange(0, BLOCK_SIZE)
        seed_offsets = row_idx + col_offsets

        input_ptrs = row_start_ptr + col_offsets
        seed_ptrs = seed_ptr + seed_offsets
        random = tl.rand(seed_ptrs, col_offsets)
        x_keep = random > p

        mask = col_offsets < N
        row = tl.load(input_ptrs, mask=mask, other=-float('inf'))

        output = tl.where(x_keep, row / (1 - p), 0.0)
        output_row_start_ptr = output_ptr + row_idx * output_row_stride
        output_ptrs = output_row_start_ptr + col_offsets
        tl.store(output_ptrs, output, mask=mask)


dropout_kernels = {}
def seeded_dropout_2d(x, p, seed):
    n_rows, n_cols = x.shape
    BLOCK_SIZE = triton.next_power_of_2(n_cols)
    y = torch.empty_like(x)
    num_warps = 1

    kernel, num_programs = dropout_kernels.get(BLOCK_SIZE, (None, 0))
    if kernel is None:
        device = torch.cuda.current_device()
        properties = driver.active.utils.get_device_properties(device)
        NUM_SM = properties["multiprocessor_count"]
        NUM_REGS = properties["max_num_regs"]
        SIZE_SMEM = properties["max_shared_mem"]
        WARP_SIZE = properties["warpSize"]
        kernel = _seeded_dropout_2d.warmup(x, y, seed, n_rows, n_cols, p, x.stride(0), y.stride(0), BLOCK_SIZE=BLOCK_SIZE, num_warps=num_warps, grid=(1, ))
        kernel._init_handles()
        n_regs = kernel.n_regs
        size_smem = kernel.metadata.shared
        occupancy = NUM_REGS // (n_regs * WARP_SIZE * num_warps)
        if size_smem > 0:
            occupancy = min(occupancy, SIZE_SMEM // size_smem)
        num_programs = NUM_SM * occupancy
        dropout_kernels[BLOCK_SIZE] = (kernel, num_programs)

    num_programs = min(num_programs, n_rows)
    kernel[(num_programs, 1, 1)](x, y, seed, n_rows, n_cols, p, x.stride(0), y.stride(0))
    return y
#########################################################################################


try:
    # This is https://github.com/NVIDIA/apex, NOT the apex on PyPi, so it
    # should not be added to extras_require in setup.py.
    import apex
    HAS_APEX = True
except ModuleNotFoundError:
    HAS_APEX = False


@triton.jit
def _layer_norm_fwd_fused(
    X,  # pointer to the input
    Y,  # pointer to the output
    W,  # pointer to the weights
    B,  # pointer to the biases
    Mean,  # pointer to the mean
    Rstd,  # pointer to the 1/std
    stride,  # how much to increase the pointer when moving by 1 row
    N,  # number of columns in X
    eps,  # epsilon to avoid division by zero
    BLOCK_SIZE: tl.constexpr,
):
    # Map the program id to the row of X and Y it should compute.
    row = tl.program_id(0)
    Y += row * stride
    X += row * stride
    # Compute mean
    mean = 0
    _mean = tl.zeros([BLOCK_SIZE], dtype=tl.float32)
    for off in range(0, N, BLOCK_SIZE):
        cols = off + tl.arange(0, BLOCK_SIZE)
        a = tl.load(X + cols, mask=cols < N, other=0.).to(tl.float32)
        _mean += a
    mean = tl.sum(_mean, axis=0) / N
    # Compute variance
    _var = tl.zeros([BLOCK_SIZE], dtype=tl.float32)
    for off in range(0, N, BLOCK_SIZE):
        cols = off + tl.arange(0, BLOCK_SIZE)
        x = tl.load(X + cols, mask=cols < N, other=0.).to(tl.float32)
        x = tl.where(cols < N, x - mean, 0.)
        _var += x * x
    var = tl.sum(_var, axis=0) / N
    rstd = 1 / tl.sqrt(var + eps)
    # Write mean / rstd
    tl.store(Mean + row, mean)
    tl.store(Rstd + row, rstd)
    # Normalize and apply linear transformation
    for off in range(0, N, BLOCK_SIZE):
        cols = off + tl.arange(0, BLOCK_SIZE)
        mask = cols < N
        w = tl.load(W + cols, mask=mask)
        b = tl.load(B + cols, mask=mask)
        x = tl.load(X + cols, mask=mask, other=0.).to(tl.float32)
        x_hat = (x - mean) * rstd
        y = x_hat * w + b
        # Write output
        tl.store(Y + cols, y, mask=mask)


@triton.jit
def _layer_norm_bwd_dx_fused(DX,  # pointer to the input gradient
                             DY,  # pointer to the output gradient
                             DW,  # pointer to the partial sum of weights gradient
                             DB,  # pointer to the partial sum of biases gradient
                             X,  # pointer to the input
                             W,  # pointer to the weights
                             Mean,  # pointer to the mean
                             Rstd,  # pointer to the 1/std
                             Lock,  # pointer to the lock
                             stride,  # how much to increase the pointer when moving by 1 row
                             N,  # number of columns in X
                             GROUP_SIZE_M: tl.constexpr, BLOCK_SIZE_N: tl.constexpr):
    # Map the program id to the elements of X, DX, and DY it should compute.
    row = tl.program_id(0)
    cols = tl.arange(0, BLOCK_SIZE_N)
    mask = cols < N
    X += row * stride
    DY += row * stride
    DX += row * stride
    # Offset locks and weights/biases gradient pointer for parallel reduction
    lock_id = row % GROUP_SIZE_M
    Lock += lock_id
    Count = Lock + GROUP_SIZE_M
    DW = DW + lock_id * N + cols
    DB = DB + lock_id * N + cols
    # Load data to SRAM
    x = tl.load(X + cols, mask=mask, other=0).to(tl.float32)
    dy = tl.load(DY + cols, mask=mask, other=0).to(tl.float32)
    w = tl.load(W + cols, mask=mask).to(tl.float32)
    mean = tl.load(Mean + row)
    rstd = tl.load(Rstd + row)
    # Compute dx
    xhat = (x - mean) * rstd
    wdy = w * dy
    xhat = tl.where(mask, xhat, 0.)
    wdy = tl.where(mask, wdy, 0.)
    c1 = tl.sum(xhat * wdy, axis=0) / N
    c2 = tl.sum(wdy, axis=0) / N
    dx = (wdy - (xhat * c1 + c2)) * rstd
    # Write dx
    tl.store(DX + cols, dx, mask=mask)
    # Accumulate partial sums for dw/db
    partial_dw = (dy * xhat).to(w.dtype)
    partial_db = (dy).to(w.dtype)
    while tl.atomic_cas(Lock, 0, 1) == 1:
        pass
    count = tl.load(Count)
    # First store doesn't accumulate
    if count == 0:
        tl.atomic_xchg(Count, 1)
    else:
        partial_dw += tl.load(DW, mask=mask)
        partial_db += tl.load(DB, mask=mask)
    tl.store(DW, partial_dw, mask=mask)
    tl.store(DB, partial_db, mask=mask)
    # Release the lock
    tl.atomic_xchg(Lock, 0)


@triton.jit
def _layer_norm_bwd_dwdb(DW,  # pointer to the partial sum of weights gradient
                         DB,  # pointer to the partial sum of biases gradient
                         FINAL_DW,  # pointer to the weights gradient
                         FINAL_DB,  # pointer to the biases gradient
                         M,  # GROUP_SIZE_M
                         N,  # number of columns
                         BLOCK_SIZE_M: tl.constexpr, BLOCK_SIZE_N: tl.constexpr):
    # Map the program id to the elements of DW and DB it should compute.
    pid = tl.program_id(0)
    cols = pid * BLOCK_SIZE_N + tl.arange(0, BLOCK_SIZE_N)
    dw = tl.zeros((BLOCK_SIZE_M, BLOCK_SIZE_N), dtype=tl.float32)
    db = tl.zeros((BLOCK_SIZE_M, BLOCK_SIZE_N), dtype=tl.float32)
    # Iterate through the rows of DW and DB to sum the partial sums.
    for i in range(0, M, BLOCK_SIZE_M):
        rows = i + tl.arange(0, BLOCK_SIZE_M)
        mask = (rows[:, None] < M) & (cols[None, :] < N)
        offs = rows[:, None] * N + cols[None, :]
        dw += tl.load(DW + offs, mask=mask, other=0.)
        db += tl.load(DB + offs, mask=mask, other=0.)
    # Write the final sum to the output.
    sum_dw = tl.sum(dw, axis=0)
    sum_db = tl.sum(db, axis=0)
    tl.store(FINAL_DW + cols, sum_dw, mask=cols < N)
    tl.store(FINAL_DB + cols, sum_db, mask=cols < N)


class LayerNorm(torch.autograd.Function):

    @staticmethod
    def forward(ctx, x, normalized_shape, weight, bias, eps):
        # allocate output
        y = torch.empty_like(x)
        # reshape input data into 2D tensor
        x_arg = x.reshape(-1, x.shape[-1])
        M, N = x_arg.shape
        mean = torch.empty((M, ), dtype=torch.float32, device=x.device)
        rstd = torch.empty((M, ), dtype=torch.float32, device=x.device)
        # Less than 64KB per feature: enqueue fused kernel
        MAX_FUSED_SIZE = 65536 // x.element_size()
        BLOCK_SIZE = min(MAX_FUSED_SIZE, triton.next_power_of_2(N))
        if N > BLOCK_SIZE:
            raise RuntimeError("This layer norm doesn't support feature dim >= 64KB.")
        # heuristics for number of warps
        num_warps = min(max(BLOCK_SIZE // 256, 1), 8)
        # enqueue kernel
        _layer_norm_fwd_fused[(M, )](  #
            x_arg, y, weight, bias, mean, rstd,  #
            x_arg.stride(0), N, eps,  #
            BLOCK_SIZE=BLOCK_SIZE, num_warps=num_warps, num_ctas=1)
        ctx.save_for_backward(x, weight, bias, mean, rstd)
        ctx.BLOCK_SIZE = BLOCK_SIZE
        ctx.num_warps = num_warps
        ctx.eps = eps
        return y

    @staticmethod
    def backward(ctx, dy):
        x, w, b, m, v = ctx.saved_tensors
        # heuristics for amount of parallel reduction stream for DW/DB
        N = w.shape[0]
        GROUP_SIZE_M = 64
        if N <= 8192: GROUP_SIZE_M = 96
        if N <= 4096: GROUP_SIZE_M = 128
        if N <= 1024: GROUP_SIZE_M = 256
        # allocate output
        locks = torch.zeros(2 * GROUP_SIZE_M, dtype=torch.int32, device=w.device)
        _dw = torch.zeros((GROUP_SIZE_M, N), dtype=x.dtype, device=w.device)
        _db = torch.zeros((GROUP_SIZE_M, N), dtype=x.dtype, device=w.device)
        dw = torch.empty((N, ), dtype=w.dtype, device=w.device)
        db = torch.empty((N, ), dtype=w.dtype, device=w.device)
        dx = torch.empty_like(dy)
        # enqueue kernel using forward pass heuristics
        # also compute partial sums for DW and DB
        x_arg = x.reshape(-1, x.shape[-1])
        M, N = x_arg.shape
        _layer_norm_bwd_dx_fused[(M, )](  #
            dx, dy, _dw, _db, x, w, m, v, locks,  #
            x_arg.stride(0), N,  #
            BLOCK_SIZE_N=ctx.BLOCK_SIZE,  #
            GROUP_SIZE_M=GROUP_SIZE_M,  #
            num_warps=ctx.num_warps)
        grid = lambda meta: [triton.cdiv(N, meta['BLOCK_SIZE_N'])]
        # accumulate partial sums in separate kernel
        _layer_norm_bwd_dwdb[grid](
            _dw, _db, dw, db, min(GROUP_SIZE_M, M), N,  #
            BLOCK_SIZE_M=32,  #
            BLOCK_SIZE_N=128, num_ctas=1)
        return dx, None, dw, db, None


layer_norm = LayerNorm.apply


def test_layer_norm(M, N, dtype, eps=1e-5, device='cuda'):
    # create data
    x_shape = (M, N)
    w_shape = (x_shape[-1], )
    weight = torch.rand(w_shape, dtype=dtype, device=device, requires_grad=True)
    bias = torch.rand(w_shape, dtype=dtype, device=device, requires_grad=True)
    x = -2.3 + 0.5 * torch.randn(x_shape, dtype=dtype, device=device)
    dy = .1 * torch.randn_like(x)
    x.requires_grad_(True)
    # forward pass
    y_tri = layer_norm(x, w_shape, weight, bias, eps)
    y_ref = torch.nn.functional.layer_norm(x, w_shape, weight, bias, eps).to(dtype)
    # backward pass (triton)
    y_tri.backward(dy, retain_graph=True)
    dx_tri, dw_tri, db_tri = [_.grad.clone() for _ in [x, weight, bias]]
    x.grad, weight.grad, bias.grad = None, None, None
    # backward pass (torch)
    y_ref.backward(dy, retain_graph=True)
    dx_ref, dw_ref, db_ref = [_.grad.clone() for _ in [x, weight, bias]]
    # compare
    assert torch.allclose(y_tri, y_ref, atol=1e-2, rtol=0)
    assert torch.allclose(dx_tri, dx_ref, atol=1e-2, rtol=0)
    assert torch.allclose(db_tri, db_ref, atol=1e-2, rtol=0)
    assert torch.allclose(dw_tri, dw_ref, atol=1e-2, rtol=0)


if __name__ == "__main__":
    torch.manual_seed(0)
    # target = driver.active.get_current_target()
    # print(f"Target: {target}")

    # # Addition
    # size = 98432
    # x, y = torch.rand(size, device='cuda'), torch.rand(size, device='cuda')
    # output_torch, output_triton = x + y, add(x, y)
    # # print(output_torch)
    # # print(output_triton)
    # print(f'Addition: the maximum difference between torch and triton is {torch.max(torch.abs(output_torch - output_triton))}')

    # # Softmax
    # x = torch.randn(1823, 781, device='cuda')
    # y_triton = softmax(x)
    # y_torch = torch.softmax(x, axis=1)
    # # assert torch.allclose(y_triton, y_torch), (y_triton, y_torch)
    # print(f'Softmax: the maximum difference between torch and triton is {torch.max(torch.abs(y_torch - y_triton))}')

    # # Matmul
    # a = torch.randn((512, 512), device='cuda', dtype=torch.float16)
    # b = torch.randn((512, 512), device='cuda', dtype=torch.float16)
    # triton_output = matmul(a, b)
    # torch_output = torch.matmul(a, b)
    # rtol = 0
    # if torch.allclose(triton_output, torch_output, atol=1e-2, rtol=rtol):
    #     print("✅ Triton and Torch match")
    # else:
    #     print(f"triton_output_with_fp16_inputs={triton_output}")
    #     print(f"torch_output_with_fp16_inputs={torch_output}")
    #     print("❌ Triton and Torch differ")


    # TORCH_HAS_FP8 = hasattr(torch, "float8_e5m2")
    # if TORCH_HAS_FP8:
    #     torch.manual_seed(0)
    #     a = torch.randn((512, 512), device="cuda", dtype=torch.float16)
    #     b = torch.randn((512, 512), device="cuda", dtype=torch.float16)
    #     a = a.to(torch.float8_e5m2)
    #     # pre-transpose b for efficiency.
    #     b = b.T
    #     b = b.to(torch.float8_e5m2)
    #     triton_output = matmul(a, b)
    #     torch_output = torch.matmul(a.to(torch.float16), b.to(torch.float16))
    #     if torch.allclose(triton_output, torch_output, atol=0.125, rtol=0):
    #         print("✅ Triton and Torch match")
    #     else:
    #         print("❌ Triton and Torch differ")  
    #         print(f"triton_output_with_fp8_inputs={triton_output}")
    #         print(f"torch_output_with_fp8_inputs={torch_output}")  

    # # Dropout 1D
    # x = torch.randn(size=(10, )).cuda()
    # p = 0.5
    # x_keep = (torch.rand(size=(10, )) > p).to(torch.int32).cuda()
    # output = dropout_1d(x, x_keep=x_keep, p=p)
    # print(tabulate.tabulate([["input"] + x.tolist(), ["keep mask"] + x_keep.tolist(), ["output"] + output.tolist()]))
    
    # # Compare this to the baseline - dropout mask is never instantiated!
    # output = seeded_dropout_1d(x, p=0.5, seed=123)
    # output2 = seeded_dropout_1d(x, p=0.5, seed=123)
    # output3 = seeded_dropout_1d(x, p=0.5, seed=512)

    # print(
    #     tabulate.tabulate([
    #         ["input"] + x.tolist(),
    #         ["output (seed = 123)"] + output.tolist(),
    #         ["output (seed = 123)"] + output2.tolist(),
    #         ["output (seed = 512)"] + output3.tolist(),
    #     ]))


    # # Dropout 2D
    # x = torch.randn(size=(128, 32)).cuda()
    # p = 0.5
    # seed = torch.randint(0, 2**16, size=(128, ), device=x.device, dtype=torch.int32)
    # output = seeded_dropout_2d(x, p=0.5, seed=seed)

    # print(
    #     tabulate.tabulate([
    #         ["input (row == 0)"] + x[-1].tolist(),
    #         ["output (row == 0)"] + output[-1].tolist(),
    #     ]))

    
    # LayerNorm
    test_layer_norm(1151, 8192, torch.float16)