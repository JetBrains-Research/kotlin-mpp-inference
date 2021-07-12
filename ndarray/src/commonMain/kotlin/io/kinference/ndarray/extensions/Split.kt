package io.kinference.ndarray.extensions

import io.kinference.ndarray.Strides
import io.kinference.ndarray.arrays.MutableNDArray
import io.kinference.ndarray.arrays.NDArray
import kotlin.math.ceil

fun computeSplitShape(shape: IntArray, axis: Int, split: Int, keepDims: Boolean): IntArray {
    val newShape: IntArray
    if (keepDims) {
        newShape = shape.copyOf()
        newShape[axis] = split
    } else {
        newShape = IntArray(shape.size - 1)
        shape.copyInto(newShape, 0, 0, axis)
        shape.copyInto(newShape, axis, axis + 1)
    }
    return newShape
}


fun NDArray.splitWithAxis(parts: Int, axis: Int = 0, keepDims: Boolean = true): List<MutableNDArray> {
    val actualAxis = indexAxis(axis)
    require(actualAxis in shape.indices) { "Index $axis out of shape bound: (${-rank}, ${rank - 1}" }
    val elementsByIndex = shape[actualAxis]
    val mainSplit = ceil(elementsByIndex.toDouble() / parts).toInt()
    val split = IntArray(parts) { mainSplit }

    val tail = elementsByIndex % parts
    if (tail != 0) split[parts - 1] = tail

    return splitWithAxis(split, actualAxis, keepDims)
}

fun NDArray.splitWithAxis(split: IntArray, axis: Int, keepDims: Boolean = true): List<MutableNDArray> {
    val beforeAxisDims = computeBlockSize(toDim = axis)
    val fromAxisDims = computeBlockSize(fromDim = axis)
    val afterAxisDims = if (axis + 1 == rank) 1 else computeBlockSize(fromDim = axis + 1)

    var inputOffset = 0

    return List(split.size) { i ->
        val splitSize = split[i]
        val outputDims = computeSplitShape(strides.shape, axis, split[i], keepDims)
        val outStrides = Strides(outputDims)
        val fragmentSize = splitSize * afterAxisDims

        val dst = this.splitFragment(beforeAxisDims, fromAxisDims, fragmentSize, outStrides, inputOffset)
        inputOffset += fragmentSize
        dst
    }
}

fun NDArray.splitFragment(beforeAxisDims: Int, fromAxisDims: Int, fragmentSize: Int, splitStrides: Strides, offset: Int): MutableNDArray {
    val dst = this.allocateNDArray(splitStrides)
    val len = beforeAxisDims * fragmentSize
    if (fromAxisDims == fragmentSize) {
        dst.copyFrom(0, this, 0, len)
        return dst
    }

    repeat(beforeAxisDims) {
        val start = offset + fromAxisDims * it
        dst.copyFrom(it * fragmentSize, this, start, start + fragmentSize)
    }
    return dst
}

fun NDArray.splitParts(parts: Int, strides: Strides): List<MutableNDArray> {
    require(linearSize % parts == 0)
    require(strides.linearSize == linearSize / parts)

    var offset = 0
    val partSize = strides.linearSize
    return List(parts) {
        val newArray = allocateNDArray(strides)
        newArray.copyFrom(0, this, offset, offset + partSize)
        offset += partSize
        newArray
    }
}
