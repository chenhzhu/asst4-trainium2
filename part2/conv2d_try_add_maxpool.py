import numpy as np
import math

import neuronxcc.nki as nki
import neuronxcc.nki.language as nl
import neuronxcc.nki.isa as nisa
from neuronxcc.nki import baremetal

@nki.compiler.skip_middle_end_transformations
@nki.jit
def _transpose_weight_slice(
    W_slice_hbm,
    W_transposed_sbuf,
    i,
    j,
    n_tiles_c_out,
    n_tiles_c_in,
    c_tile,
    filter_width,
    dtype,
):
    """
    Transposes a weight slice in tiles and stores in output buffer.
    
    Args:
        W_slice_hbm: Weight slice of shape (out_channels, in_channels) in HBM
        W_transposed_sbuf: Output buffer laid out in SBUF with height fixed to `c_tile`
        i, j: Filter position indices
        n_tiles_c_out: Number of tiles in output channel dimension
        n_tiles_c_in: Number of tiles in input channel dimension
        c_tile: Tile size (128)
        filter_width: Convolution filter width
        dtype: Data type
    """
    # Transpose in tiles
    out_channels = n_tiles_c_out * c_tile
    filter_idx = i * filter_width + j
    for c_out_idx in nl.affine_range(n_tiles_c_out):
        for c_in_idx in nl.affine_range(n_tiles_c_in):
            c_out_start = c_out_idx * c_tile
            c_in_start = c_in_idx * c_tile
            
            # Load, transpose, and store tile
            W_tile_sbuf = nl.ndarray((c_tile, c_tile), dtype=dtype, buffer=nl.sbuf)
            nisa.dma_copy(
                dst=W_tile_sbuf,
                src=W_slice_hbm[c_out_start:c_out_start+c_tile, c_in_start:c_in_start+c_tile]
            )
            W_tile_T_psum = nisa.nc_transpose(W_tile_sbuf)
            W_tile_T_sbuf = nl.copy(W_tile_T_psum, dtype=dtype)
            
            # Store to transposed weight matrix
            # nisa.dma_copy(
            #     src=W_tile_T_sbuf,
            #     dst=W_transposed_sbuf[
            #         :,
            #         (
            #             (filter_idx * n_tiles_c_in + c_in_idx) * out_channels
            #             + c_out_start
            #         ) : (
            #             (filter_idx * n_tiles_c_in + c_in_idx) * out_channels
            #             + c_out_start
            #             + c_tile
            #         ),
            #     ],
            # )

            W_transposed_sbuf[
                    :,
                    (
                        (filter_idx * n_tiles_c_in + c_in_idx) * out_channels
                        + c_out_start
                    ) : (
                        (filter_idx * n_tiles_c_in + c_in_idx) * out_channels
                        + c_out_start
                        + c_tile
                    )
                ] = nisa.tensor_copy(W_tile_T_sbuf)

"""
A fused convolution - maxpool kernel that you need to implement for Part 2.

Parameters:
    X: the input tensor
    W: the weights of the convolution filters.
    bias: the biases of the convolution filters.
    pool_size: the size of the pool filter and pool stride.

expect: X.shape == [batch_size, in_channels, input_height, input_width]
expect: W.shape == [out_channels, in_channels, filter_height, filter_width]
expect: bias.shape == [out_channels]
expect: filter_height == filter_width
expect: pool_size == 1 || pool_size == 2
expect: input_channels % 128 == 0
expect: output_channels % 128 == 0

out_height = input_height - filter_height + 1
out_width = input_width - filter_width + 1

out_pool_height = out_height // pool_size
out_pool_width = out_width // pool_size

The shape of the output should be [batch_size, out_channels, out_pool_height, out_pool_width]

"""


@nki.compiler.skip_middle_end_transformations
@nki.jit
def fused_conv2d_maxpool(X, W, bias, pool_size=1):
    """
    X - A batch of input images. X has shape (Batch Size, Input Channels, Input Height, Input Width). You are guaranteed that Input Channels will be a multiple of 128.
    W - The convolution filter weights. W has shape (Output Channels, Input Channels, Filter Height, Filter Width). You are guaranteed that Filter Height == Filter Width. You are also guaranteed that Output Channels is a multiple of 128. Moreover, you can assume that the size of the weights would always be such that it can completely fit inside SBUF.
    bias - The convolution filter biases. bias has shape (Output Channels)
    pool_size - The size of the max pooling filter and pooling stride. You are guaranteed that the size of the input, the size of the filter, and the pool_size would be such that everything is nicely divisible. More concretely, (Input Height - Filter Height + 1) % Pool Size == 0. Notice that if the value of pool_size is 1, then the fused kernel operates as a normal convolution kernel. This gives us the flexibility to choose whether we want max pooling or not.

    The shape of the output should be [batch_size, out_channels, out_pool_height, out_pool_width]
    """

    batch_size, in_channels, input_height, input_width = X.shape
    out_channels, in_channels_, filter_height, filter_width = W.shape
    out_channels_ = bias.shape[0]
    type_ = X.dtype

    float_size = 4 if type_ is np.float32 else 2
    SBUF_size = 200 * 1024 # 224k
    input_height_chunk = min(math.ceil(SBUF_size / float_size / (in_channels * input_width)), input_height)
    n_total_chunk = math.ceil(input_height / input_height_chunk)

    # print("X.shape: ", X.shape)

    assert (
        in_channels_ == in_channels and out_channels_ == out_channels
    ), f"Shape mismatch. {in_channels}, {in_channels_}, {out_channels}, {out_channels_}"

    out_height = input_height - filter_height + 1
    out_width = input_width - filter_width + 1

    out_pool_height = out_height // pool_size
    out_pool_width = out_width // pool_size

    # Can assume multiple of 128 to avoid using mask
    assert in_channels % 128 == out_channels % 128 == 0

    # Can assume one PSUM bank can at least fit one row of the pixels
    assert nl.tile_size.gemm_moving_fmax >= out_width

    # Initialize output array
    X_out = nl.ndarray(
        shape=(batch_size, out_channels, out_pool_height, out_pool_width),
        dtype=type_,
        buffer=nl.hbm,
    )

    c_tile = nl.tile_size.pmax  # 128
    n_tiles_c_out = out_channels // c_tile
    n_tiles_c_in = in_channels // c_tile

    # print(f"Debug: out_height={out_height}, out_width={out_width}, out_pool_height={out_pool_height}, out_pool_width={out_pool_width}")
    # print(f"Debug: n_tiles_c_out={n_tiles_c_out}, n_tiles_c_in={n_tiles_c_in}")

    c_out_tile = c_tile
    c_in_tile = c_tile

    # Pre-transpose all weight slices to avoid repeated transposes in batch loop
    # W_transposed_sbuf shape:
    #   (c_tile, filter_height * filter_width * n_tiles_c_in * out_channels)
    W_transposed_sbuf = nl.ndarray(
        (
            c_tile,
            filter_height * filter_width * n_tiles_c_in * out_channels,
        ),
        dtype=type_,
        buffer=nl.sbuf,
    )

    # flatten_zeros_hbm = _allocate_zero_flatten(out_channels, out_height, out_width, type_)
    # bias_hbm = nl.ndarray((out_channels,), dtype=type_, buffer=nl.hbm)
    # nisa.dma_copy(dst=bias_hbm, src=bias)

    # TODO enable bias
    # flatten bias here
    bisa_flatten_sbuf = nl.ndarray((128, n_tiles_c_out), dtype=type_, buffer=nl.sbuf)
    for i in nl.affine_range(n_tiles_c_out):
        nisa.dma_copy(
            dst = bisa_flatten_sbuf[:, i: (i + 1)],
            src = bias[i * 128: (i + 1) * 128]
        )

    x_tile_width = out_width
    total_tile_j = math.ceil(out_height * out_width / x_tile_width) # over cout
    
    # Transpose each filter position's weight matrix
    for i in nl.affine_range(filter_height):
        for j in nl.affine_range(filter_width):
            W_slice_hbm = W[:, :, i, j]  # (out_channels, in_channels) - slice from HBM
            _transpose_weight_slice(
                W_slice_hbm,
                W_transposed_sbuf,
                i,
                j,
                n_tiles_c_out,
                n_tiles_c_in,
                c_tile,
                filter_width,
                type_,
            )

    

    result_flatten_hbm = nl.ndarray((out_channels, out_height * out_width), dtype=type_, buffer=nl.hbm)

    for b in nl.affine_range(batch_size):
        # Initialize accumulation buffers for max pooling (one per tile_i)
        # Must be defined outside tile_j loop to allow accumulation across tile_j iterations
        if pool_size == 2:
            out_pool_height_chunk = pool_size * 1
            # Buffer array for all tile_i: (n_tiles_c_out, 128, out_pool_height_chunk, out_width)
            out_chunk_pool_height_sbuf = nl.ndarray((n_tiles_c_out, nl.par_dim(c_out_tile), out_pool_height_chunk, out_width), dtype=type_, buffer=nl.sbuf)
        
        for tile_j in nl.affine_range(total_tile_j): # loop over the output height size
            # nl.device_print("tile_j", tile_j)

            # Prepare only the needed input heights for this output row tile (cache in SBUF)
            X_sbuf = nl.zeros(
                        shape=(
                            n_tiles_c_in,
                            nl.par_dim(c_tile),
                            filter_height,
                            input_width,
                        ),
                        dtype = type_,
                        buffer = nl.sbuf
                    )
            for c_index in nl.affine_range(in_channels//128):
                nisa.dma_copy(
                    dst=X_sbuf[c_index, :, :, :],
                    src=X[b, c_index * c_tile: (c_index + 1) * c_tile, tile_j: tile_j + filter_height, :]
                )

            # 128: n_tiles_c_in * f_width * f_height
            x_shifted_sbuf_single = nl.zeros((128, x_tile_width * n_tiles_c_in), dtype=type_, buffer=nl.sbuf)
            num_rows = x_tile_width // out_width

            for tile_i in nl.affine_range(n_tiles_c_out):

                # tile_i
                # R_ij_sbuf = nl.ndarray((128, x_tile_width), dtype=type_, buffer=nl.sbuf) # C_in x HW
                # psum_sbuf here
                # Accumulate in FP32 PSUM as required by TRN2 (even for FP16 inputs)
                # TODO: need to consider 2 filter height of X_sbuf
                psum = nl.zeros((128, x_tile_width), dtype=np.float32, buffer=nl.psum)

                # In this \sum_i \sum_j \sum_k loop we calculate R_ij
                for filter_i in nl.affine_range(filter_height):
                    for filter_j in nl.affine_range(filter_width):

                        # nisa.dma_copy()
                        # print("sbuf.shape", X_sbuf.shape)
                        # nl.device_print("filter_i", filter_i)
                        # nl.device_print("filter_j", filter_j)
                        # print("out_height", out_height)
                        # print("out_width", out_width)
                        for k in nl.affine_range(n_tiles_c_in):
                            filter_idx = filter_i * filter_width + filter_j
                            width_start = ((filter_idx * n_tiles_c_in + k) * out_channels) + tile_i * 128
                            # print("X_fi_fj.shape", X_fi_fj.shape)
                            # nl.device_print("k", k)
                            # nl.device_print("chunk_offset", chunk_offset)
                            # print("Hello")
                            # Slice X from SBUF cache for this (k, filter_i, filter_j)
                            # TODO: need to consider 2 filter height of X_sbuf
                            psum += nisa.nc_matmul(W_transposed_sbuf[:, width_start : width_start + 128], X_sbuf[k, :, filter_i, filter_j : filter_j + out_width])


                R_ij_sbuf = nisa.tensor_copy(psum, dtype=type_)
                
                # TODO enable bias
                R_ij_sbuf[...] = nisa.tensor_tensor(R_ij_sbuf, bisa_flatten_sbuf[:, tile_i : (tile_i + 1)], op=nl.add, dtype=type_)

                # max pool
                # TODO add max horizontally on R_ij, then store it to sbuf of R_j for maxpooling

                # flattened
                # nisa.dma_copy(dst=result_flatten_hbm[tile_i * x_tile_width : (tile_i + 1) * x_tile_width, tile_j], src=R_ij_sbuf)
                if pool_size == 1:
                    nisa.dma_copy(
                        dst = X_out[b, tile_i * 128 : (tile_i + 1) * 128, tile_j, :],
                        src = R_ij_sbuf
                    )
                elif pool_size == 2:
                    # Store current row to accumulation buffer for this tile_i
                    out_chunk_pool_height_sbuf[tile_i, :, tile_j % out_pool_height_chunk, :] = nisa.tensor_copy(R_ij_sbuf, dtype=type_)
                    
                    # When we have accumulated pool_size rows, perform max pooling
                    if tile_j % out_pool_height_chunk == out_pool_height_chunk - 1:
                        # Calculate pooled dimensions
                        pooled_width = out_width // pool_size
                        pooled_height = out_pool_height_chunk // pool_size  # We're processing one chunk at a time
                        
                        # Reshape for max pooling: (128, pool_size, out_width) -> (128, pooled_height, pool_size, pooled_width, pool_size)
                        # Note: We need to reshape to enable 2x2 max pooling
                        # Extract the buffer for this tile_i: (128, out_pool_height_chunk, out_width)
                        tile_buffer = out_chunk_pool_height_sbuf[tile_i, :, :, :]
                        out_reshaped = tile_buffer.reshape((
                            nl.par_dim(128),
                            pooled_height,
                            pool_size,
                            pooled_width,
                            pool_size
                        ))
                        
                        # Max pool: find maximum across the 2x2 pooling window
                        # Compare elements at (0,0) and (1,1) positions
                        pooled_a = nisa.tensor_tensor(
                            out_reshaped[:, :, 0, :, 0],
                            out_reshaped[:, :, 1, :, 1],
                            op=nl.maximum,
                            dtype=type_
                        )
                        # Compare elements at (1,0) and (0,1) positions
                        pooled_b = nisa.tensor_tensor(
                            out_reshaped[:, :, 1, :, 0],
                            out_reshaped[:, :, 0, :, 1],
                            op=nl.maximum,
                            dtype=type_
                        )
                        # Final max: maximum of pooled_a and pooled_b
                        pooled = nisa.tensor_tensor(
                            pooled_a,
                            pooled_b,
                            op=nl.maximum,
                            dtype=type_
                        )
                        
                        # Calculate the output row index after pooling
                        pool_row_idx = tile_j // pool_size
                        
                        # Copy pooled result to HBM output
                        nisa.dma_copy(
                            dst = X_out[b, tile_i * 128 : (tile_i + 1) * 128, pool_row_idx, :],
                            src = pooled
                        )

            # Reshape the flattened result back to 2D spatial dimensions
            # result_2d_hbm = result_flatten_hbm.reshape((out_channels, out_height, out_width))
            
            # # TODO if pool size == 2
            # nisa.dma_copy(
            #     src=result_2d_hbm[:, :out_pool_height, :out_pool_width],
            #     dst=X_out[b]
            # )
    
    return X_out
