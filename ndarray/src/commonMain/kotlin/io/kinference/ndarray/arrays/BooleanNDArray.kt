package io.kinference.ndarray.arrays

import io.kinference.ndarray.Strides
import io.kinference.ndarray.arrays.pointers.BooleanPointer
import io.kinference.ndarray.arrays.tiled.*
import io.kinference.ndarray.extensions.isScalar
import io.kinference.primitives.types.DataType
import io.kinference.primitives.types.PrimitiveType
import kotlin.math.abs

interface BooleanMap : PrimitiveToPrimitiveFunction {
    fun apply(value: Boolean): Boolean
}

open class BooleanNDArray(var array: BooleanTiledArray, strides: Strides) : NDArray {
    constructor(shape: IntArray) : this(BooleanTiledArray(shape), Strides(shape))
    constructor(shape: IntArray, init: (Int) -> Boolean) : this(BooleanTiledArray(shape, init), Strides(shape))

    constructor(strides: Strides) : this(BooleanTiledArray(strides), strides)
    constructor(strides: Strides, init: (Int) -> Boolean) : this(BooleanTiledArray(strides, init), strides)

    constructor(shape: IntArray, customBlockSize: Int) : this(BooleanTiledArray(shape.reduce(Int::times), customBlockSize), Strides(shape))
    constructor(shape: IntArray, customBlockSize: Int, init: (Int) -> Boolean) :
        this(BooleanTiledArray(shape.reduce(Int::times), customBlockSize, init), Strides(shape))

    constructor(strides: Strides, customBlockSize: Int) : this(BooleanTiledArray(strides.linearSize, customBlockSize), strides)
    constructor(strides: Strides, customBlockSize: Int, init: (Int) -> Boolean) :
        this(BooleanTiledArray(strides.linearSize, customBlockSize, init), strides)

    override val type: DataType = DataType.BOOLEAN

    final override var strides: Strides = strides
        protected set

    protected val blocksInRow: Int
        get() = when {
            strides.linearSize == 0 -> 0
            strides.shape.isEmpty() -> 1
            else -> strides.shape.last() / array.blockSize
        }

    override fun view(vararg axes: Int): NDArray {
        for ((i, axis) in axes.withIndex()) {
            require(shape[i] > axis)
        }

        val offset = axes.foldIndexed(0) { index, acc, i -> acc + i * strides.strides[index] }

        val newShape = shape.copyOfRange(axes.size, shape.size)
        val newStrides = Strides(newShape)

        if (array.blockSize == 0)
            return BooleanNDArray(array, newStrides)


        val offsetBlocks = offset / array.blockSize

        val countBlocks = newStrides.linearSize / array.blockSize

        val copyBlocks = array.blocks.copyOfRange(offsetBlocks, offsetBlocks + countBlocks)
        val newArray = BooleanTiledArray(copyBlocks)

        return BooleanNDArray(newArray, newStrides)
    }

    override fun singleValue(): Boolean {
        require(isScalar() || array.size == 1) { "NDArray contains more than 1 value" }
        return array.blocks[0][0]
    }

    override fun allocateNDArray(strides: Strides): MutableNDArray {
        return MutableBooleanNDArray(BooleanTiledArray(strides), strides)
    }

    override fun reshapeView(newShape: IntArray): NDArray {
        return BooleanNDArray(array, Strides(newShape))
    }

    override fun toMutable(newStrides: Strides): MutableNDArray {
        return MutableBooleanNDArray(array.copyOf(), strides)
    }

    override fun copyIfNotMutable(): MutableNDArray {
        return MutableBooleanNDArray(array, strides)
    }

    override fun map(function: PrimitiveToPrimitiveFunction): MutableNDArray {
        function as BooleanMap
        val destination = allocateNDArray(strides) as MutableBooleanNDArray
        for (index in 0 until destination.linearSize) {
            destination.array[index] = function.apply(this.array[index])
        }

        return destination
    }

    override fun row(row: Int): MutableNDArray {
        val rowLength: Int = linearSize / shape[0]
        val start = row * rowLength
        val dims = shape.copyOfRange(1, rank)

        return MutableBooleanNDArray(BooleanTiledArray(Strides(dims)) { array[start + it] }, Strides(dims))
    }

    override fun slice(starts: IntArray, ends: IntArray, steps: IntArray): MutableNDArray {
        val newShape = IntArray(shape.size) {
            val length = abs(ends[it] - starts[it])
            val rest = length % abs(steps[it])
            (length / abs(steps[it])) + if (rest != 0) 1 else 0
        }

        val newStrides = Strides(newShape)
        val newArray = BooleanTiledArray(newStrides)

        if (newArray.size > 0) {
            slice(newArray.pointer(), this.array.pointer(), 0, 0, shape, starts, ends, steps)
        }

        return MutableBooleanNDArray(newArray, newStrides)
    }

    private fun slice(dst: BooleanPointer, src: BooleanPointer, offset: Int, axis: Int, shape: IntArray, starts: IntArray, ends: IntArray, steps: IntArray) {
        val start = starts[axis]
        val end = ends[axis]
        val step = steps[axis]

        val range = if (step > 0) (start until end step step) else (start downTo end + 1 step -step)

        if (axis == shape.size - 1) {
            for (index in range) {
                src.linearIndex = offset + index
                dst.set(src.get())
                dst.increment()
            }
        } else {
            var dim = 1
            for (ind in (axis + 1) until shape.size) dim *= shape[ind]

            for (index in range) {
                slice(dst, src, offset + index * dim, axis + 1, shape, starts, ends, steps)
            }
        }
    }

    override fun splitHorizontalByBlocks(parts: Int): Array<NDArray> {
        TODO("Not yet implemented")
    }

    override fun equals(other: Any?): Boolean {
        if (this === other) return true
        if (other !is BooleanNDArray) return false

        if (type != other.type) return false
        if (strides != other.strides) return false
        if (array != other.array) return false

        return true
    }

    companion object {
        fun scalar(value: Boolean): BooleanNDArray {
            return BooleanNDArray(BooleanTiledArray(1, 1) { value }, Strides.EMPTY)
        }

        operator fun invoke(array: BooleanTiledArray, strides: Strides, customBlockSize: Int): BooleanNDArray {
            return if (customBlockSize == array.blockSize) {
                BooleanNDArray(array, strides)
            }
            else {
                val pointer = BooleanPointer(array)
                BooleanNDArray(strides, customBlockSize) { pointer.getAndIncrement() }
            }
        }
    }
}

class MutableBooleanNDArray(array: BooleanTiledArray, strides: Strides = Strides.EMPTY): BooleanNDArray(array, strides), MutableNDArray {
    constructor(shape: IntArray) : this(BooleanTiledArray(shape), Strides(shape))

    constructor(shape: IntArray, customBlockSize: Int) : this(BooleanTiledArray(shape.reduce(Int::times), customBlockSize), Strides(shape))
    constructor(shape: IntArray, customBlockSize: Int, init: (Int) -> Boolean) :
        this(BooleanTiledArray(shape.reduce(Int::times), customBlockSize, init), Strides(shape))

    constructor(strides: Strides, customBlockSize: Int) : this(BooleanTiledArray(strides.linearSize, customBlockSize), strides)
    constructor(strides: Strides, customBlockSize: Int, init: (Int) -> Boolean) :
        this(BooleanTiledArray(strides.linearSize, customBlockSize, init), strides)

    override fun viewMutable(vararg axes: Int): MutableNDArray {
        val offset = axes.foldIndexed(0) { index, acc, i -> acc + i * strides.strides[index] }
        val offsetBlocks = offset / array.blockSize

        val newShape = shape.copyOfRange(axes.size, shape.size)
        val newStrides = Strides(newShape)

        val countBlocks = newStrides.linearSize / array.blockSize

        val copyBlocks = array.blocks.copyOfRange(offsetBlocks, offsetBlocks + countBlocks)
        val newArray = BooleanTiledArray(copyBlocks)

        return MutableBooleanNDArray(newArray, newStrides)
    }

    override fun copyIfNotMutable(): MutableNDArray {
        return MutableBooleanNDArray(array, strides)
    }

    override fun mapMutable(function: PrimitiveToPrimitiveFunction): MutableNDArray {
        function as BooleanMap
        for (index in 0 until linearSize) {
            array[index] = function.apply(array[index])
        }

        return this
    }

    override fun copyFrom(offset: Int, other: NDArray, startInOther: Int, endInOther: Int) {
        other as BooleanNDArray
        other.array.copyInto(this.array, offset, startInOther, endInOther)
    }

    override fun fill(value: Any, from: Int, to: Int) {
        array.fill(value as Boolean)
    }

    override fun fillByArrayValue(array: NDArray, index: Int, from: Int, to: Int) {
        array as BooleanNDArray
        val (blockIndex, blockOffset) = array.array.indexFor(index)
        this.array.fill(array.array.blocks[blockIndex][blockOffset], from, to)
    }

    override fun reshape(strides: Strides): MutableNDArray {
        this.strides = strides
        return this
    }

    // TODO separate from PrimitiveArray (maybe LateInitArray will help)
    private fun transposeRec(prevArray: BooleanTiledArray, newArray: BooleanTiledArray, prevStrides: Strides, newStrides: Strides, index: Int, prevOffset: Int, newOffset: Int, permutation: IntArray) {
        if (index != newStrides.shape.lastIndex) {
            val temp = prevStrides.strides[permutation[index]]
            val temp2 = newStrides.strides[index]
            for (i in 0 until newStrides.shape[index])
                transposeRec(prevArray, newArray, prevStrides, newStrides, index + 1, prevOffset + temp * i,
                    newOffset + temp2 * i, permutation)
        } else {
            val temp = prevStrides.strides[permutation[index]]
            if (temp == 1) {
                prevArray.copyInto(newArray, newOffset, prevOffset, prevOffset + newStrides.shape[index])
            } else {
                var (newArrayBlock, newArrayOffset) = newArray.indexFor(newOffset)
                var (prevArrayBlock, prevArrayOffset) = prevArray.indexFor(prevOffset)

                val (deltaBlock, deltaOffset) = prevArray.indexFor(temp)

                var tempNewBlock = newArray.blocks[newArrayBlock]
                var tempPrevBlock = prevArray.blocks[prevArrayBlock]

                for (i in 0 until newStrides.shape[index]) {
                    tempNewBlock[newArrayOffset++] = tempPrevBlock[prevArrayOffset]

                    prevArrayBlock += deltaBlock
                    prevArrayOffset += deltaOffset

                    if (prevArrayOffset >= prevArray.blockSize) {
                        prevArrayBlock += prevArrayOffset / prevArray.blockSize
                        prevArrayOffset %= prevArray.blockSize
                    }

                    if (prevArrayBlock < prevArray.blocksNum)
                        tempPrevBlock = prevArray.blocks[prevArrayBlock]

                    if (newArrayOffset >= newArray.blockSize) {
                        newArrayOffset = 0

                        if (++newArrayBlock < newArray.blocksNum)
                            tempNewBlock = newArray.blocks[newArrayBlock]
                    }
                }
            }
        }
    }

    override fun transpose(permutations: IntArray): MutableNDArray {
        val newStrides = strides.transpose(permutations)
        val newArray = BooleanTiledArray(newStrides)
        array.copyInto(newArray)

        transposeRec(array, newArray, strides, newStrides, 0, 0, 0, permutations)

        this.strides = newStrides
        this.array = newArray
        return this
    }

    override fun transpose2D(): MutableNDArray {
        TODO("Not yet implemented")
    }

    override fun clean() {
        array.fill(false)
    }

    fun not(): MutableNDArray {
        return mapMutable(object : BooleanMap {
            override fun apply(value: Boolean): Boolean = value.not()
        })
    }

    companion object {
        fun scalar(value: Boolean): MutableBooleanNDArray {
            return MutableBooleanNDArray(BooleanTiledArray(1, 1) { value }, Strides.EMPTY)
        }
    }
}
