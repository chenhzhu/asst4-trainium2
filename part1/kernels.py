"""
CS 149: Parallel Computing, Assigment 4 Part 1

This file contains the kernel implementations for the vector addition benchmark.

For Step 1 & 2, you should look at these kernels:
    - vector_add_naive
    - vector_add_tiled
    - vector_add_stream
For Step 3, you should look at this kernel:
    - matrix_transpose

It's highly recommended to carefully read the code of each kernel and understand how
they work. For NKI functions, you can refer to the NKI documentation at:
https://awsdocs-neuron-staging.readthedocs-hosted.com/en/nki_docs_2.21_beta_class/
"""

import numpy as np
import math

import neuronxcc.nki as nki
import neuronxcc.nki.language as nl
import neuronxcc.nki.isa as nisa


"""
This is the naive implementation of a vector add kernel. 
Due to the 128 partition size limit, this kernel only works for vector sizes <=128.
"""
@nki.compiler.skip_middle_end_transformations
@nki.jit
def vector_add_naive(a_vec, b_vec):
    
    # Allocate space for the output vector in HBM
    out = nl.ndarray(shape=a_vec.shape, dtype=a_vec.dtype, buffer=nl.hbm)

    # Allocate space for the input vectors in SBUF and copy them from HBM
    a_sbuf = nl.ndarray(shape=(a_vec.shape[0], 1), dtype=a_vec.dtype, buffer=nl.sbuf)
    b_sbuf = nl.ndarray(shape=(b_vec.shape[0], 1), dtype=b_vec.dtype, buffer=nl.sbuf)
    
    nisa.dma_copy(src=a_vec, dst=a_sbuf)
    nisa.dma_copy(src=b_vec, dst=b_sbuf)

    # Add the input vectors
    res = nisa.tensor_scalar(a_sbuf, nl.add, b_sbuf)

    # Store the result into HBM
    nisa.dma_copy(src=res, dst=out)

    return out

"""
This is the tiled implementation of a vector add kernel.
We load the input vectors in chunks, add them, and then store the result
chunk into HBM. Therefore, this kernel works for any vector that is a
multiple of 128.
"""
@nki.compiler.skip_middle_end_transformations
@nki.jit
def vector_add_tiled(a_vec, b_vec):
    
    # Allocate space for the output vector in HBM
    out = nl.ndarray(shape=a_vec.shape, dtype=a_vec.dtype, buffer=nl.hbm)

    # Get the total number of vector rows
    M = a_vec.shape[0]
    
    # TODO: You should modify this variable for Step 1
    ROW_CHUNK = 256 # 128

    # Loop over the total number of chunks, we can use affine_range
    # because there are no loop-carried dependencies
    for m in nl.affine_range(M // ROW_CHUNK):

        # Allocate row-chunk sized tiles for the input vectors
        a_tile = nl.ndarray((ROW_CHUNK, 1), dtype=a_vec.dtype, buffer=nl.sbuf)
        b_tile = nl.ndarray((ROW_CHUNK, 1), dtype=b_vec.dtype, buffer=nl.sbuf)
        
        # Load a chunk of rows
        nisa.dma_copy(src=a_vec[m * ROW_CHUNK : (m + 1) * ROW_CHUNK], dst=a_tile)
        nisa.dma_copy(src=b_vec[m * ROW_CHUNK : (m + 1) * ROW_CHUNK], dst=b_tile)

        # Add the row chunks together
        res = nisa.tensor_scalar(a_tile, nl.add, b_tile)

        # Store the result chunk into HBM
        nisa.dma_copy(src=res, dst=out[m * ROW_CHUNK : (m + 1) * ROW_CHUNK])
    
    return out

"""
This is an extension of the vector_add_tiled kernel. Instead of loading tiles
of size (ROW_CHUNK, 1), we reshape the vectors into (PARTITION_DIM, FREE_DIM)
tiles. This allows us to amortize DMA transfer overhead and load many more
elements per DMA transfer.
"""
@nki.compiler.skip_middle_end_transformations
@nki.jit
def vector_add_stream(a_vec, b_vec):

    # Get the total number of vector rows
    M = a_vec.shape[0]

    # TODO: You should modify this variable for Step 2a
    FREE_DIM = 1000 #2000

    # The maximum size of our Partition Dimension
    PARTITION_DIM = 128

    a_vec_re = a_vec.reshape((PARTITION_DIM, M // PARTITION_DIM))
    b_vec_re = b_vec.reshape((PARTITION_DIM, M // PARTITION_DIM))
    out = nl.ndarray(shape=a_vec_re.shape, dtype=a_vec_re.dtype, buffer=nl.hbm)

    # Loop over the total number of tiles
    for m in nl.affine_range(M // (PARTITION_DIM * FREE_DIM)):

        # Allocate space for a reshaped tile
        a_tile = nl.ndarray((PARTITION_DIM, FREE_DIM), dtype=a_vec.dtype, buffer=nl.sbuf)
        b_tile = nl.ndarray((PARTITION_DIM, FREE_DIM), dtype=b_vec.dtype, buffer=nl.sbuf)

        # Load the input tiles
        nisa.dma_copy(src=a_vec_re[:, m * FREE_DIM : (m + 1) * FREE_DIM], dst=a_tile)
        nisa.dma_copy(src=b_vec_re[:, m * FREE_DIM : (m + 1) * FREE_DIM], dst=b_tile)

        # Add the tiles together. Note that we must switch to tensor_tensor instead of tensor_scalar
        res = nisa.tensor_tensor(a_tile, b_tile, op=nl.add)

        # Store the result tile into HBM
        nisa.dma_copy(src=res, dst=out[:, m * FREE_DIM : (m + 1) * FREE_DIM])

    # Reshape the output vector into its original shape
    out = out.reshape(a_vec.shape)

    return out

"""
This kernel implements a simple 2D matrix transpose.
It uses a tile-based approach along with NKI's built-in transpose kernel,
which only works on tiles of size <= 128x128.
"""
@nki.compiler.skip_middle_end_transformations
@nki.jit
def matrix_transpose(a_tensor):
    M, N = a_tensor.shape
    out = nl.ndarray((N, M), dtype=a_tensor.dtype, buffer=nl.hbm)
    tile_dim = nl.tile_size.pmax  # this should be 128

    assert M % tile_dim == N % tile_dim == 0, "Matrix dimensions not divisible by tile dimension!"

    # TODO: Your implementation here. The only compute instruction you should use is `nisa.nc_transpose`.
    
    # Loop over all tiles in the input matrix
    num_tiles_m = M // tile_dim
    num_tiles_n = N // tile_dim
    
    for i in nl.affine_range(num_tiles_m):
        for j in nl.affine_range(num_tiles_n):
            # Allocate space for input tile in SBUF
            input_tile = nl.ndarray((tile_dim, tile_dim), dtype=a_tensor.dtype, buffer=nl.sbuf)
            
            # Load input tile from HBM to SBUF
            nisa.dma_copy(src=a_tensor[i * tile_dim : (i + 1) * tile_dim, j * tile_dim : (j + 1) * tile_dim], 
                         dst=input_tile)
            
            # Transpose the tile (result stored in PSUM)
            res_psum = nisa.nc_transpose(input_tile, engine=nki.isa.constants.engine.vector)
            
            # Copy from PSUM to SBUF
            res_sbuf = nl.copy(res_psum, dtype=a_tensor.dtype)
            
            # Store transposed tile to output at position (j, i) in HBM
            nisa.dma_copy(src=res_sbuf, 
                         dst=out[j * tile_dim : (j + 1) * tile_dim, i * tile_dim : (i + 1) * tile_dim])

    return out

"""
Optimized version of matrix transpose for extra credit.
Key optimizations for memory-bound operations:
1. Optimize loop order to improve memory access patterns (better write locality)
2. Use tensor_copy for faster PSUM to SBUF transfer (ISA-level operation)
3. Minimize operations by directly using tensor_copy result
"""
@nki.compiler.skip_middle_end_transformations
@nki.jit
def matrix_transpose_optimized(a_tensor):
    M, N = a_tensor.shape
    out = nl.ndarray((N, M), dtype=a_tensor.dtype, buffer=nl.hbm)
    tile_dim = nl.tile_size.pmax  # this should be 128

    assert M % tile_dim == N % tile_dim == 0, "Matrix dimensions not divisible by tile dimension!"
    
    num_tiles_m = M // tile_dim
    num_tiles_n = N // tile_dim
    
    # Optimize loop order: process by output tile position (j, i)
    # This improves write locality - for fixed j, we write to consecutive columns
    # which is more cache-friendly than the original i-outer, j-inner pattern
    for j in nl.affine_range(num_tiles_n):
        # Key optimization: Load entire column of tiles in one DMA operation
        # This reduces DMA copy overhead significantly - from num_tiles_m copies to just 1 copy per j
        # Load the entire column (M, tile_dim) directly - no transpose, preserve original layout
        # Note: If M is large, we may need to handle SBUF size limits, but let's try direct load first
        input_column = nl.ndarray((M, tile_dim), dtype=a_tensor.dtype, buffer=nl.sbuf)
        # Load the entire column from HBM - shape (M, tile_dim)
        nisa.dma_copy(src=a_tensor[:, j * tile_dim : (j + 1) * tile_dim], 
                     dst=input_column)
        
        for i in nl.affine_range(num_tiles_m):
            # Extract 128x128 tile from the pre-loaded column
            # Slice: input_column[i*tile_dim:(i+1)*tile_dim, :] gives (tile_dim, tile_dim)
            # This preserves the correct data layout for nc_transpose
            input_tile = input_column[i * tile_dim : (i + 1) * tile_dim, :]
            
            # Transpose the tile (result stored in PSUM)
            res_psum = nisa.nc_transpose(input_tile)
            
            # Copy from PSUM to SBUF using tensor_copy (ISA-level, faster than nl.copy)
            output_tile = nisa.tensor_copy(res_psum)
            
            # Store transposed tile to output at position (j, i) in HBM
            # Issue DMA write immediately after computation for better pipelining
            nisa.dma_copy(src=output_tile, 
                         dst=out[j * tile_dim : (j + 1) * tile_dim, i * tile_dim : (i + 1) * tile_dim])

    return out






@nki.compiler.skip_middle_end_transformations
@nki.jit
def matrix_transpose_optimized(a_tensor):
    M, N = a_tensor.shape
    out = nl.ndarray((N, M), dtype=a_tensor.dtype, buffer=nl.hbm)
    tile_dim = nl.tile_size.pmax  # typically 128

    assert M % tile_dim == 0 and N % tile_dim == 0, "Matrix dimensions not divisible by tile dimension!"
    num_tiles_m = M // tile_dim
    num_tiles_n = N // tile_dim

    # Partition dimension: each row‑tile block is a partition
    assert num_tiles_m <= nl.tile_size.pmax, "num_tiles_m exceeds partition limit!"
    # Define input_column with first dim = num_tiles_m (partition)
    input_column = nl.ndarray((num_tiles_m, tile_dim, tile_dim),
                              dtype=a_tensor.dtype, buffer=nl.sbuf)

    for j in nl.affine_range(num_tiles_n):
        # Load j‑th tile‑column blocks from HBM → SBUF
        for i in nl.affine_range(num_tiles_m):
            # Source block shape (tile_dim × tile_dim)
            src_block = a_tensor[i*tile_dim:(i+1)*tile_dim,
                                 j*tile_dim:(j+1)*tile_dim]
            # Reshape to include partition dim = 1
            src = src_block.reshape((1, tile_dim, tile_dim))
            dst = input_column[i:i+1, :, :]   # Keep first dim as partition = 1
            nisa.dma_copy(src=src, dst=dst)

        for i in nl.affine_range(num_tiles_m):
            # Retrieve tile from SBUF
            tile = input_column[i:i+1, :, :].reshape((tile_dim, tile_dim))
            # Perform transpose
            res_psum = nisa.nc_transpose(tile)
            output_tile = nisa.tensor_copy(res_psum)
            # Destination in output matrix
            dst_block = out[j*tile_dim:(j+1)*tile_dim,
                            i*tile_dim:(i+1)*tile_dim]
            dst_out_tile = dst_block.reshape((1, tile_dim, tile_dim))
            nisa.dma_copy(src=output_tile.reshape((1, tile_dim, tile_dim)),
                          dst=dst_out_tile)

    return out