package io.kinference.ndarray.arrays.tiled

import io.kinference.ndarray.Strides

object TiledArraysUtils {
    internal const val MIN_BLOCK_SIZE = 512

    fun blockSizeByStrides(strides: Strides): Int {
        return when {
            strides.linearSize == 0 -> 0
            strides.shape.isEmpty() -> 1
            else -> blockSizeByLastDim(strides.shape.last())
        }
    }

    fun blockSizeByLastDim(lastDim: Int): Int {
        return if (lastDim < MIN_BLOCK_SIZE) lastDim else {
            var num = lastDim / MIN_BLOCK_SIZE
            while (lastDim % num != 0) num--
            lastDim / num
        }
    }
}
