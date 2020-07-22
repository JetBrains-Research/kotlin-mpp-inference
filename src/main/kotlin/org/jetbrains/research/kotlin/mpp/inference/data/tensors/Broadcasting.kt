package org.jetbrains.research.kotlin.mpp.inference.data.tensors

import org.jetbrains.research.kotlin.mpp.inference.mathExtension.createInferredTypeBuffer
import scientifik.kmath.structures.BufferNDStructure
import kotlin.math.max

fun broadcastShape(currentShape: IntArray, newShape: IntArray): IntArray {
    val totalShapeLength = max(currentShape.size, newShape.size)
    val resultShape = IntArray(totalShapeLength) { -1 }
    val revCurrentShape = currentShape.reversedArray()
    val revNewShape = newShape.reversedArray()

    for (i in 0 until totalShapeLength) {
        val currentDim = revCurrentShape.getOrNull(i) ?: 1
        val newDim = revNewShape.getOrNull(i) ?: 1

        if (currentDim != newDim && currentDim != 1 && newDim != 1) error("Cannot broadcast shapes")

        resultShape[i] = max(currentDim, newDim)
    }

    return resultShape.reversedArray()
}

fun broadcastShape(currentShape: List<Int>, newShape: List<Int>): IntArray {
    return broadcastShape(currentShape.toIntArray(), newShape.toIntArray())
}

fun broadcastMatrixElementsShape(fstShape: IntArray, sndShape: IntArray): Pair<IntArray, IntArray> {
    val base = broadcastShape(fstShape.dropLast(2), sndShape.dropLast(2))

    val fst = base + fstShape.takeLast(2)
    val snd = base + sndShape.takeLast(2)
    return fst to snd
}

private fun Tensor.innerBroadcast(newShape: IntArray, asMatrixStack: Boolean = false): Tensor {
    if (this.data.shape.contentEquals(newShape) || asMatrixStack && this.data.dimension <= 2) return this

    val castShape = newShape.copyOfRange(1, newShape.size)

    //broadcast is available only if corresponding dims are equal or at least one of them is 1
    return when (this.data.shape[0]) {
        1 -> {
            val rows = this.row(0).innerBroadcast(castShape)
            rows.reshape(intArrayOf(1, *castShape)).repeatRow(newShape[0])
        }
        newShape[0] -> this.rows.map { it.innerBroadcast(castShape) }.concatenate(axis = 0)
        else -> error("Cannot broadcast tensors")
    }
}

fun Tensor.broadcast(newShape: IntArray, asMatrixStack: Boolean = false): Tensor {
    if (this.data.shape.contentEquals(newShape)) return this

    val newDims = this.data.shape.copyOf().toMutableList()

    if (newShape.size > this.data.dimension)
        repeat(newShape.size - this.data.dimension) { newDims.add(0, 1) }

    val preResult = this.reshape(newDims.toIntArray())

    return preResult.innerBroadcast(newShape, asMatrixStack)
}


fun Tensor.elementWiseWithBroadcast(other: Tensor, op: (Any, Any) -> Any): Tensor {
    val newShape = broadcastShape(data.shape, other.data.shape)
    val castedThis = this.broadcast(newShape).data.buffer
    val castedOther = other.broadcast(newShape).data.buffer
    val strides = TensorStrides(newShape)

    val (res, type) = createInferredTypeBuffer(info.type, other.info.type, strides.linearSize) { op(castedThis[it], castedOther[it]) }
    return Tensor(this.info.name, BufferNDStructure(TensorStrides(newShape), res), type)
}
