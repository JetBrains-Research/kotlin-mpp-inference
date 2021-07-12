@file:GeneratePrimitives(DataType.ALL)
@file:Suppress("DuplicatedCode")

package io.kinference.ndarray.arrays.tiled

import io.kinference.ndarray.Strides
import io.kinference.ndarray.arrays.pointers.PrimitivePointer
import io.kinference.ndarray.arrays.pointers.accept
import io.kinference.ndarray.arrays.tiled.TiledArraysUtils.blockSizeByStrides
import io.kinference.primitives.annotations.*
import io.kinference.primitives.types.*
import kotlin.math.min

@GenerateNameFromPrimitives
class PrimitiveTiledArray {
    val size: Int
    val blockSize: Int
    val blocksNum: Int
    val blocks: Array<PrimitiveArray>

    companion object {

        operator fun invoke(strides: Strides): PrimitiveTiledArray {
            val blockSize = blockSizeByStrides(strides)
            return PrimitiveTiledArray(strides.linearSize, blockSize)
        }

        operator fun invoke(strides: Strides, init: (Int) -> PrimitiveType): PrimitiveTiledArray {
            val blockSize = blockSizeByStrides(strides)
            return PrimitiveTiledArray(strides.linearSize, blockSize, init)
        }

        operator fun invoke(shape: IntArray) = invoke(Strides(shape))

        operator fun invoke(shape: IntArray, init: (Int) -> PrimitiveType) = invoke(Strides(shape), init)
    }

    constructor(size: Int, blockSize: Int) {
        if (blockSize != 0)
            require(size % blockSize == 0) { "Size must divide blockSize" }

        this.blocksNum = if (blockSize == 0) 0 else size / blockSize
        this.blocks = Array(blocksNum) { PrimitiveArray(blockSize) }
        this.blockSize = blockSize
        this.size = size
    }

    constructor(blocks: Array<PrimitiveArray>) {
        this.blocks = blocks
        this.blockSize = if (blocks.isEmpty()) 0 else blocks.first().size
        this.blocksNum = blocks.size
        this.size = this.blocksNum * this.blockSize
    }

    constructor(size: Int, blockSize: Int, init: (Int) -> PrimitiveType) : this(size, blockSize) {
        var count = 0
        for (block in blocks) {
            for (idx in 0 until blockSize) {
                block[idx] = init(count++)
            }
        }
    }

    fun pointer(startIndex: Int = 0) = PrimitivePointer(this, startIndex)

    fun toArray(): PrimitiveArray {
        if (size == 0) {
            return PrimitiveArray(0)
        }

        val array = PrimitiveArray(size)
        var offset = 0

        for (block in blocks) {
            block.copyInto(array, offset)
            offset += blockSize
        }

        return array
    }

    fun indexFor(i: Int): Pair<Int, Int> {
        val blockIdx = i / blockSize
        val blockOff = i % blockSize
        return blockIdx to blockOff
    }

    operator fun get(i: Int): PrimitiveType {
        val (blockIdx, blockOff) = indexFor(i)
        return blocks[blockIdx][blockOff]
    }

    operator fun set(i: Int, value: PrimitiveType) {
        val (blockIdx, blockOff) = indexFor(i)
        blocks[blockIdx][blockOff] = value
    }

    fun copyOf(): PrimitiveTiledArray {
        val copyArray = PrimitiveTiledArray(size, blockSize)

        for (blockNum in 0 until blocksNum) {
            val thisBlock = this.blocks[blockNum]
            val destBlock = copyArray.blocks[blockNum]

            thisBlock.copyInto(destBlock)
        }

        return copyArray
    }

    fun copyInto(dest: PrimitiveTiledArray, destOffset: Int = 0, srcStart: Int = 0, srcEnd: Int = size) {
        if (srcStart == srcEnd)
            return

        val thisPtr = PrimitivePointer(this, srcStart)
        val destPtr = PrimitivePointer(dest, destOffset)

        destPtr.accept(thisPtr, srcEnd - srcStart) { _: PrimitiveType, src: PrimitiveType -> src }
    }

    @FilterPrimitives(exclude = [DataType.BOOLEAN])
    fun plus(other: PrimitiveTiledArray): PrimitiveTiledArray {
        val thisPtr = PrimitivePointer(this)
        val destPtr = PrimitivePointer(other)
        thisPtr.accept(destPtr, this.size) { src: PrimitiveType, dst: PrimitiveType -> (src + dst).toPrimitive() }
        return this
    }

    fun copyOfRange(fromIndex: Int, toIndex: Int): PrimitiveArray {
        val array = PrimitiveArray(toIndex - fromIndex)
        val pointer = PrimitivePointer(this, fromIndex)

        for (i in array.indices) {
            array[i] = pointer.getAndIncrement()
        }

        return array
    }

    fun fill(value: PrimitiveType, from: Int = 0, to: Int = size) {
        if (from == to)
            return

        val pointer = PrimitivePointer(this, from)

        var count = to - from

        while (count > 0) {
            val block = pointer.currentBlock
            val offset = pointer.indexInBlock
            pointer.blockIncrement()

            block.fill(value, offset, min(blockSize, count))

            count -= blockSize
        }
    }
}
