@file:GeneratePrimitives(DataType.NUMBER)
@file:Suppress("DuplicatedCode", "unused")

package io.kinference.ndarray.arrays

import io.kinference.ndarray.*
import io.kinference.ndarray.arrays.pointers.*
import io.kinference.ndarray.arrays.tiled.*
import io.kinference.ndarray.extensions.*
import io.kinference.primitives.annotations.*
import io.kinference.primitives.types.*
import kotlinx.coroutines.*
import kotlin.math.*

@GenerateNameFromPrimitives
open class PrimitiveNDArray(array: PrimitiveTiledArray, strides: Strides) : NumberNDArray {
    constructor(shape: IntArray) : this(PrimitiveTiledArray(shape), Strides(shape))
    constructor(shape: IntArray, init: (Int) -> PrimitiveType) : this(PrimitiveTiledArray(shape, init), Strides(shape))

    constructor(strides: Strides) : this(PrimitiveTiledArray(strides), strides)
    constructor(strides: Strides, init: (Int) -> PrimitiveType) : this(PrimitiveTiledArray(strides, init), strides)

    constructor(shape: IntArray, customBlockSize: Int) : this(PrimitiveTiledArray(shape.reduce(Int::times), customBlockSize), Strides(shape))
    constructor(shape: IntArray, customBlockSize: Int, init: (Int) -> PrimitiveType) :
        this(PrimitiveTiledArray(shape.reduce(Int::times), customBlockSize, init), Strides(shape))

    constructor(strides: Strides, customBlockSize: Int) : this(PrimitiveTiledArray(strides.linearSize, customBlockSize), strides)
    constructor(strides: Strides, customBlockSize: Int, init: (Int) -> PrimitiveType) :
        this(PrimitiveTiledArray(strides.linearSize, customBlockSize, init), strides)


    var array: PrimitiveTiledArray = array
        protected set

    internal val blocksInRow: Int
        get() = when {
            strides.linearSize == 0 -> 0
            strides.shape.isEmpty() -> 1
            else -> strides.shape.last() / array.blockSize
        }

    override fun view(vararg axes: Int): PrimitiveNDArray {
        for ((i, axis) in axes.withIndex()) {
            require(shape[i] > axis)
        }

        val offset = axes.foldIndexed(0) { index, acc, i -> acc + i * strides.strides[index] }

        val newShape = shape.copyOfRange(axes.size, shape.size)
        val newStrides = Strides(newShape)

        if (array.blockSize == 0)
            return PrimitiveNDArray(array, newStrides)


        val offsetBlocks = offset / array.blockSize

        val countBlocks = newStrides.linearSize / array.blockSize

        val copyBlocks = array.blocks.copyOfRange(offsetBlocks, offsetBlocks + countBlocks)
        val newArray = PrimitiveTiledArray(copyBlocks)

        return PrimitiveNDArray(newArray, newStrides)
    }

    override val type = DataType.CurrentPrimitive

    final override var strides: Strides = strides
        protected set


    override fun singleValue(): PrimitiveType {
        require(isScalar() || array.size == 1) { "NDArray contains more than 1 value" }
        return array.blocks[0][0]
    }

    override fun allocateNDArray(strides: Strides): MutableNumberNDArray = MutablePrimitiveNDArray(PrimitiveTiledArray(strides), strides)

    override fun reshapeView(newShape: IntArray): NDArray {
        val newStrides = Strides(newShape)

        require(newStrides.linearSize == linearSize)

        return PrimitiveNDArray(array, newStrides)
    }

    override fun toMutable(newStrides: Strides): MutableNumberNDArray = MutablePrimitiveNDArray(array.copyOf(), newStrides)

    override fun map(function: PrimitiveToPrimitiveFunction): MutableNumberNDArray {
        function as PrimitiveMap
        val destination = allocateNDArray(strides) as MutablePrimitiveNDArray

        for (blockNum in 0 until array.blocksNum) {
            val thisBlock = this.array.blocks[blockNum]
            val destBlock = destination.array.blocks[blockNum]

            for (idx in thisBlock.indices) {
                destBlock[idx] = function.apply(thisBlock[idx])
            }
        }
        return destination
    }

    override fun erfFor(value: Any): PrimitiveType {
        value as PrimitiveType
        val sign = value.toDouble().sign
        val doubleValue = abs(value.toDouble())
        val t = 1 / (1 + ERF_P_VALUE * doubleValue)

        val sum = t * (ERF_COEF[0] + t * (ERF_COEF[1] + t * (ERF_COEF[2] + t * (ERF_COEF[3] + t * ERF_COEF[4]))))

        return (sign * (1.0 - sum * exp(-doubleValue * doubleValue))).toPrimitive()
    }

    override fun withZeroPoint(zeroPoint: NumberNDArray): IntNDArray {
        zeroPoint as PrimitiveNDArray

        return if (zeroPoint.linearSize == 1) {
            val zero = zeroPoint.array.blocks[0][0].toInt()
            val arr = IntTiledArray(this.strides)
            arr.pointer().accept(array.pointer(), arr.size) { _, src -> src.toInt() - zero }
            IntNDArray(arr, strides)
        } else {
            val arr = IntTiledArray(strides)
            arr.pointer().acceptWithRecursive(this.array.pointer(), zeroPoint.array.pointer(), arr.size) { _, src, zero -> src.toInt() - zero.toInt() }
            IntNDArray(arr, strides)
        }
    }

    @Suppress("CAST_NEVER_SUCCEEDS")
    override fun dequantize(zeroPoint: NDArray?, scale: NDArray, axis: Int?): NDArray {
        scale as FloatNDArray
        val zeros = (zeroPoint as? PrimitiveNDArray)?.array
        val output = MutableFloatNDArray(FloatTiledArray(this.array.size, this.array.blockSize), this.strides)

        when {
            canDequantizePerTensor(zeroPoint, scale) -> {
                val zero = if (zeros == null) 0f else zeros.blocks[0][0].toFloat()
                val sc = scale.array.blocks[0][0]

                if (type == DataType.BYTE) {
                    output.array.pointer().accept(this.array.pointer() as BytePointer, output.linearSize) { _, src ->
                        (src.toFloat() - zero) * sc
                    }
                } else {
                    output.array.pointer().accept(this.array.pointer() as UBytePointer, output.linearSize) { _, src ->
                        (src.toFloat() - zero) * sc
                    }
                }
            }
            canDequantizePerAxis(axis!!, zeroPoint, scale) -> {
                val actualAxis = indexAxis(axis)
                val blockCount = computeBlockSize(toDim = actualAxis)
                val blockSize = computeBlockSize(fromDim = actualAxis + 1)
                var outOffset = 0
                repeat(blockCount) {
                    val zeroPointer = zeros?.pointer()
                    val scalePointer = scale.array.pointer()
                    for (i in 0 until shape[actualAxis]) {
                        val zero = zeroPointer?.getAndIncrement()?.toFloat() ?: 0f
                        val sc = scalePointer.getAndIncrement()

                        if (type == DataType.BYTE) {
                            output.array.pointer(outOffset).accept(this.array.pointer(outOffset) as BytePointer, blockSize) { _, src ->
                                (src.toFloat() - zero) * sc
                            }
                        } else {
                            output.array.pointer(outOffset).accept(this.array.pointer(outOffset) as UBytePointer, blockSize) { _, src ->
                                (src.toFloat() - zero) * sc
                            }
                        }

                        outOffset += blockSize
                    }
                }
            }
            else -> error("Cannot perform dequantization. Scale and zero point tensors should be either scalars or 1D tensors containing ${shape[axis]} elements")
        }

        return output
    }

    override fun row(row: Int): MutableNumberNDArray {
        val rowLength: Int = linearSize / shape[0]
        val start = row * rowLength
        val dims = shape.copyOfRange(1, rank)

        val result = PrimitiveTiledArray(Strides(dims))
        result.pointer().accept(array.pointer(start), result.size) { _, src -> src }

        return MutablePrimitiveNDArray(result, Strides(dims))
    }

    override fun slice(starts: IntArray, ends: IntArray, steps: IntArray): MutableNumberNDArray {
        val newShape = IntArray(shape.size) {
            val length = abs(ends[it] - starts[it])
            val rest = length % abs(steps[it])
            (length / abs(steps[it])) + if (rest != 0) 1 else 0
        }

        val newStrides = Strides(newShape)
        val newArray = PrimitiveTiledArray(newStrides)

        if (newArray.size > 0) {
            slice(newArray.pointer(), this.array.pointer(), 0, 0, shape, starts, ends, steps)
        }

        return MutablePrimitiveNDArray(newArray, newStrides)
    }

    private fun slice(dst: PrimitivePointer, src: PrimitivePointer, offset: Int, axis: Int, shape: IntArray, starts: IntArray, ends: IntArray, steps: IntArray) {
        val start = starts[axis]
        val end = ends[axis]
        val step = steps[axis]

        val range = if (step > 0) (start until end step step) else (start downTo end + 1 step -step)

        if (axis == shape.size - 1) {
            for (index in range) {
                src.linearIndex = offset + index
                dst.set(src.get())
                dst.increment()

                /*
                        for (index in range) {
            array.putNext(this.array[additionalOffset + index])
        }
                 */
            }
        } else {
            var dim = 1
            for (ind in (axis + 1) until shape.size) dim *= shape[ind]

            for (index in range) {
                slice(dst, src, offset + index * dim, axis + 1, shape, starts, ends, steps)
            }
        }
    }

    override fun min(): PrimitiveType {
        var min = PrimitiveType.MAX_VALUE
        for (block in array.blocks) {
            for (idx in block.indices) {
                val tmp = block[idx]
                if (tmp < min) min = tmp
            }
        }
        return min
    }

    override fun max(): PrimitiveType {
        var max = PrimitiveType.MIN_VALUE
        for (block in array.blocks) {
            for (idx in block.indices) {
                val tmp = block[idx]
                if (tmp > max) max = tmp
            }
        }

        return max
    }

    override fun sum(): PrimitiveType {
        var sum = (0).toPrimitive()

        for (block in array.blocks) {
            for (idx in block.indices) {
                sum = (sum + block[idx]).toPrimitive()
            }
        }
        return sum
    }

    override fun cumulativeSum(axis: Int, exclusive: Boolean, reverse: Boolean): MutableNumberNDArray {
        val output = MutablePrimitiveNDArray(PrimitiveTiledArray(strides), strides)

        val actualAxis = indexAxis(axis)

        val blockSize = computeBlockSize(fromDim = actualAxis + 1)
        val batchSize = computeBlockSize(fromDim = actualAxis)
        val numBatches = computeBlockSize(toDim = actualAxis)
        val numBlocks = batchSize / blockSize
        repeat(numBatches) { batchIdx ->
            val dstOff = if (!reverse) batchIdx * batchSize else (numBatches - batchIdx) * batchSize - 1
            if (!exclusive) {
                if (!reverse)
                    output.copyFrom(dstOff, this, dstOff, dstOff + blockSize)
                else
                    output.copyFrom(dstOff - blockSize + 1, this, dstOff - blockSize + 1, dstOff + 1)
            }

            // TODO rewrite using pointers
            if (!reverse) {
                for (i in 1 until numBlocks) {
                    for (j in 0 until blockSize) {
                        val currentOff = dstOff + i * blockSize + j
                        val thisOff = if (!exclusive) currentOff else currentOff - blockSize
                        output.array[currentOff] = (output.array[currentOff - blockSize] + array[thisOff]).toPrimitive()
                    }
                }
            } else {
                for (i in 1 until numBlocks) {
                    for (j in blockSize - 1 downTo 0) {
                        val currentOff = dstOff - i * blockSize - j
                        val thisOff = if (!exclusive) currentOff else currentOff + blockSize
                        output.array[currentOff] = (output.array[currentOff + blockSize] + array[thisOff]).toPrimitive()
                    }
                }
            }
        }

        return output
    }

    override fun plus(other: NumberNDArray): MutableNumberNDArray = plus(other, MutablePrimitiveNDArray(PrimitiveTiledArray(strides), strides))

    private fun plusScalar(array: PrimitiveTiledArray, scalar: PrimitiveType, destination: PrimitiveTiledArray) {
        require(array.blocksNum == destination.blocksNum && array.blockSize == destination.blockSize)

        for (blockNum in 0 until array.blocksNum) {
            val arrayBlock = array.blocks[blockNum]
            val destBlock = destination.blocks[blockNum]

            for (idx in arrayBlock.indices) {
                destBlock[idx] = (arrayBlock[idx] + scalar).toPrimitive()
            }
        }
    }

    override fun plus(other: NumberNDArray, destination: MutableNumberNDArray): MutableNumberNDArray {
        require(other is PrimitiveNDArray && destination is MutablePrimitiveNDArray) { "Operands must have the same types" }

        when {
            this.isScalar() && other.isScalar() -> destination.array.blocks[0][0] = (array.blocks[0][0] + other.array.blocks[0][0]).toPrimitive()
            this.isScalar() -> plusScalar(other.array, this.array.blocks[0][0], destination.array)
            other.isScalar() -> plusScalar(this.array, other.array.blocks[0][0], destination.array)
            else -> this.applyWithBroadcast(other, destination, false) { left, right, dest ->
                left as PrimitiveNDArray; right as PrimitiveNDArray; dest as MutablePrimitiveNDArray
                for (blockNum in 0 until left.array.blocksNum) {
                    val leftBlock = left.array.blocks[blockNum]
                    val rightBlock = right.array.blocks[blockNum]
                    val destBlock = dest.array.blocks[blockNum]

                    for (idx in leftBlock.indices) {
                        destBlock[idx] = (leftBlock[idx] + rightBlock[idx]).toPrimitive()
                    }
                }
            }
        }

        return destination
    }

    override fun minus(other: NumberNDArray): MutableNumberNDArray = minus(other, MutablePrimitiveNDArray(PrimitiveTiledArray(strides), strides))

    override fun minus(other: NumberNDArray, destination: MutableNumberNDArray): MutableNumberNDArray {
        require(other is PrimitiveNDArray && destination is MutablePrimitiveNDArray) { "Operands must have the same types" }

        when {
            this.isScalar() && other.isScalar() -> destination.array.blocks[0][0] = (this.array.blocks[0][0] - other.array.blocks[0][0]).toPrimitive()
            other.isScalar() -> {
                require(shape.contentEquals(destination.shape))

                val scalar = other.array.blocks[0][0]

                for (blockNum in 0 until array.blocksNum) {
                    val leftBlock = this.array.blocks[blockNum]
                    val destBlock = destination.array.blocks[blockNum]

                    for (idx in leftBlock.indices) {
                        destBlock[idx] = (leftBlock[idx] - scalar).toPrimitive()
                    }
                }
            }
            this.isScalar() -> error("Subtraction of a matrix from a scalar is prohibited")
            else -> this.applyWithBroadcast(other, destination, true) { left, right, dest ->
                left as PrimitiveNDArray; right as PrimitiveNDArray; dest as MutablePrimitiveNDArray

                for (blockNum in 0 until left.array.blocksNum) {
                    val leftBlock = left.array.blocks[blockNum]
                    val rightBlock = right.array.blocks[blockNum]
                    val destBlock = dest.array.blocks[blockNum]

                    for (idx in leftBlock.indices) {
                        destBlock[idx] = (leftBlock[idx] - rightBlock[idx]).toPrimitive()
                    }
                }
            }
        }

        return destination
    }

    override fun times(other: NumberNDArray): MutableNumberNDArray = times(other, MutablePrimitiveNDArray(PrimitiveTiledArray(strides), strides))

    private fun timesScalar(array: PrimitiveTiledArray, scalar: PrimitiveType, destination: PrimitiveTiledArray) {
        require(array.blocksNum == destination.blocksNum && array.blockSize == destination.blockSize)

        for (blockNum in 0 until array.blocksNum) {
            val arrayBlock = array.blocks[blockNum]
            val destBlock = destination.blocks[blockNum]

            for (idx in arrayBlock.indices) {
                destBlock[idx] = (arrayBlock[idx] * scalar).toPrimitive()
            }
        }
    }

    override fun times(other: NumberNDArray, destination: MutableNumberNDArray): MutableNumberNDArray {
        require(other is PrimitiveNDArray && destination is MutablePrimitiveNDArray) { "Operands must have the same types" }

        when {
            this.isScalar() && other.isScalar() -> destination.array.blocks[0][0] = (this.array.blocks[0][0] * other.array.blocks[0][0]).toPrimitive()
            this.isScalar() -> timesScalar(other.array, this.array.blocks[0][0], destination.array)
            other.isScalar() -> timesScalar(this.array, other.array.blocks[0][0], destination.array)
            else -> this.applyWithBroadcast(other, destination, false) { left, right, dest ->
                left as PrimitiveNDArray; right as PrimitiveNDArray; dest as MutablePrimitiveNDArray

                for (blockNum in 0 until left.array.blocksNum) {
                    val leftBlock = left.array.blocks[blockNum]
                    val rightBlock = right.array.blocks[blockNum]
                    val destBlock = dest.array.blocks[blockNum]

                    for (idx in leftBlock.indices) {
                        destBlock[idx] = (leftBlock[idx] * rightBlock[idx]).toPrimitive()
                    }
                }
            }
        }

        return destination
    }

    override fun div(other: NumberNDArray): MutableNumberNDArray = div(other, MutablePrimitiveNDArray(PrimitiveTiledArray(strides), strides))

    override fun div(other: NumberNDArray, destination: MutableNumberNDArray): MutableNumberNDArray {
        require(other is PrimitiveNDArray && destination is MutablePrimitiveNDArray) { "Operands must have the same types" }

        when {
            this.isScalar() && other.isScalar() -> destination.array.blocks[0][0] = (this.array.blocks[0][0] / other.array.blocks[0][0]).toPrimitive()
            other.isScalar() -> {
                require(shape.contentEquals(destination.shape))

                val scalar = other.array.blocks[0][0]

                for (blockNum in 0 until array.blocksNum) {
                    val leftBlock = this.array.blocks[blockNum]
                    val destBlock = destination.array.blocks[blockNum]

                    for (idx in leftBlock.indices) {
                        destBlock[idx] = (leftBlock[idx] / scalar).toPrimitive()
                    }
                }
            }
            this.isScalar() -> error("Division of a scalar into a matrix is prohibited")
            else -> this.applyWithBroadcast(other, destination, true) { left, right, dest ->
                left as PrimitiveNDArray; right as PrimitiveNDArray; dest as MutablePrimitiveNDArray

                for (blockNum in 0 until left.array.blocksNum) {
                    val leftBlock = left.array.blocks[blockNum]
                    val rightBlock = right.array.blocks[blockNum]
                    val destBlock = dest.array.blocks[blockNum]

                    for (idx in leftBlock.indices) {
                        destBlock[idx] = (leftBlock[idx] / rightBlock[idx]).toPrimitive()
                    }
                }
            }
        }

        return destination
    }

    private fun resortBlocks(blocks: Array<PrimitiveArray>, colSize: Int, blocksInRow: Int): Array<PrimitiveArray> {
        val result = blocks.copyOf()

        for (i in 0 until blocksInRow) {
            for (j in 0 until colSize) {
                result[i * colSize + j] = blocks[j * blocksInRow + i]
            }
        }

        return result
    }

    override fun dot(other: NumberNDArray, destination: MutableNumberNDArray): MutableNumberNDArray {
        other as PrimitiveNDArray; destination as MutablePrimitiveNDArray
        require(shape.size == 2 && other.shape.size == 2)
        require(shape[1] == other.shape[0])

        val n = this.shape[0]
        val t = this.shape[1]

        val resortedLeft = resortBlocks(this.array.blocks, this.shape[0], this.blocksInRow)
        val resortedRight = resortBlocks(other.array.blocks, other.shape[0], other.blocksInRow)
        val resortedDest = resortBlocks(destination.array.blocks, destination.shape[0], destination.blocksInRow)

        val lBlockSize = this.array.blockSize
        val rdBlockSize = destination.array.blockSize

        val lBlockInRow = this.blocksInRow
        val rBlockInRow = other.blocksInRow

        fun wrapper(body: (inner: () -> Unit) -> Unit = { it() }) {
            for (rdCol in 0 until rBlockInRow) {
                val rightIdx = rdCol * t
                val destIdx = rdCol * n

                body {
                    for (i in 0 until n) {
                        val destBlock = resortedDest[destIdx + i]
                        for (lCol in 0 until lBlockInRow) {
                            val leftBlock = resortedLeft[i + lCol * n]
                            val rightIdxOffset = rightIdx + lBlockSize * lCol

                            for (k in 0 until lBlockSize) {
                                val temp = leftBlock[k]
                                val rightBlock = resortedRight[rightIdxOffset + k]

                                for (j in 0 until rdBlockSize) {
                                    destBlock[j] = (destBlock[j] + temp * rightBlock[j]).toPrimitive()
                                }
                            }
                        }
                    }
                }
            }
        }

        if (rBlockInRow > 1) {
            runBlocking(Dispatchers.Default) { wrapper { launch { it() } } }
        } else {
            wrapper()
        }

        return destination
    }

    override fun dotTransposedWithAlpha(alpha: Double, other: NumberNDArray, destination: MutableNumberNDArray): MutableNumberNDArray {
        other as PrimitiveNDArray; destination as MutablePrimitiveNDArray

        @Suppress("NAME_SHADOWING") val alpha = alpha.toPrimitive()
        val dRowsNum = destination.shape[0]

        val dBlocksInRow = destination.blocksInRow
        val lrBlocksInRow = this.blocksInRow

        val dBlockSize = destination.array.blockSize
        val lrBlockSize = array.blockSize

        val dBlocks = destination.array.blocks
        val lBlocks = this.array.blocks
        val rBlocks = other.array.blocks

        fun wrapper(body: (inner: () -> Unit) -> Unit = { it() }) {
            for (dRow in 0 until dRowsNum) {
                var rRow = 0
                val dBlockOffset = dRow * dBlocksInRow
                val lBlockOffset = dRow * lrBlocksInRow

                for (dBlockInRow in 0 until dBlocksInRow) {
                    val dBlock = dBlocks[dBlockOffset + dBlockInRow]
                    val rRowOffset = rRow
                    body {
                        var rBlockOffset = rRowOffset * lrBlocksInRow
                        for (dIdx in 0 until dBlockSize) {
                            for (lrBlockInRow in 0 until lrBlocksInRow) {
                                val lBlock = lBlocks[lBlockOffset + lrBlockInRow]
                                val rBlock = rBlocks[rBlockOffset + lrBlockInRow]

                                for (lrIdx in 0 until lrBlockSize) {
                                    dBlock[dIdx] = (dBlock[dIdx] + alpha * lBlock[lrIdx] * rBlock[lrIdx]).toPrimitive()
                                }
                            }
                            rBlockOffset += lrBlocksInRow
                        }
                    }

                    rRow += dBlockSize
                }
            }
        }

        if (destination.blocksInRow > 1) {
            runBlocking(Dispatchers.Default) { wrapper { launch { it() } } }
        } else {
            wrapper()
        }

        return destination
    }

    override fun gemm(m: Int, n: Int, k: Int, alpha: Double, lda: Int, b: NDArray, ldb: Int, beta: Double, c: MutableNDArray, ldc: Int, aOffset: Int, bOffset: Int, cOffset: Int, transposeA: Boolean, transposeB: Boolean): MutableNDArray {
        b as PrimitiveNDArray; c as MutablePrimitiveNDArray
        val betaPrimitive = beta.toPrimitive()
        val alphaPrimitive = alpha.toPrimitive()
        val aPointer = array.pointer()
        val bPointer = b.array.pointer()
        val cPointer = c.array.pointer()

        if (beta != 1.0) {
            for (i in 0 until m) {
                cPointer.linearIndex = i * ldc + cOffset
                cPointer.map(n) { (betaPrimitive * it).toPrimitive() }
            }
        }

        when {
            transposeA && transposeB -> {
                // TODO rewrite using block operations
                for (t in 0 until m) {
                    for (j in 0 until n) {
                        cPointer.linearIndex = t * ldc + j + cOffset
                        for (i in 0 until k) {
                            aPointer.linearIndex = i * lda + t + aOffset
                            bPointer.linearIndex = j * ldb + i + bOffset
                            cPointer.set((alphaPrimitive * aPointer.get() * bPointer.get() + cPointer.get()).toPrimitive())
                        }
                    }
                }
            }
            transposeA -> {
                // TODO rewrite using block operations
                for (t in 0 until m) {
                    for (j in 0 until n) {
                        cPointer.linearIndex = t * ldc + j + cOffset
                        for (i in 0 until k) {
                            aPointer.linearIndex = i * lda + t + aOffset
                            bPointer.linearIndex = i * ldb + j + bOffset
                            cPointer.set((alphaPrimitive * aPointer.get() * bPointer.get() + cPointer.get()).toPrimitive())
                        }
                    }
                }
            }
            transposeB -> {
                for (t in 0 until m) {
                    val aIdx = t * lda + aOffset
                    cPointer.linearIndex = t * ldc + cOffset
                    for (j in 0 until n) {
                        aPointer.linearIndex = aIdx
                        bPointer.linearIndex = j * ldb + bOffset

                        aPointer.combine(bPointer, k) { elementInA, elementInB ->
                            cPointer.set((alphaPrimitive * elementInA * elementInB + cPointer.get()).toPrimitive())
                        }

                        cPointer.increment()
                    }
                }
            }
            else -> {
                for (t in 0 until m) {
                    val cIdx = t * ldc + cOffset
                    aPointer.linearIndex = t * lda + aOffset
                    for (i in 0 until k) {
                        val temp = (alphaPrimitive * aPointer.getAndIncrement()).toPrimitive()

                        bPointer.linearIndex = i * ldb + bOffset
                        cPointer.linearIndex = cIdx

                        cPointer.accept(bPointer, n) { elementInC, elementInB ->
                            (temp * elementInB + elementInC).toPrimitive()
                        }
                    }
                }
            }
        }

        return c
    }

    override fun splitHorizontalByBlocks(parts: Int): Array<NDArray> {
        require(rank <= 2) { "" }
        if (blocksInRow % parts != 0) {
            logger.info { "Split works ineffectively" }
            return splitWithAxis(parts, axis = -1).toTypedArray()
        }
        require(blocksInRow % parts == 0) { "" }

        val blocksInRow = this.blocksInRow
        val partBlocksInRow = blocksInRow / parts
        val blocksInPart = if (rank == 1) partBlocksInRow else partBlocksInRow * shape[0]


        val partShape = shape.copyOf()
        partShape[partShape.lastIndex] /= parts
        val partStrides = Strides(partShape)

        return Array(parts) { part ->
            val partOffset = partBlocksInRow * part
            val partBlocks = Array(blocksInPart) { block ->
                val rowNum = block / partBlocksInRow
                val colNum = block % partBlocksInRow

                array.blocks[rowNum * blocksInRow + partOffset + colNum]
            }

            PrimitiveNDArray(PrimitiveTiledArray(partBlocks), partStrides)
        }
    }

    override fun argmax(axis: Int, keepDims: Boolean, selectLastIndex: Boolean): IntNDArray {
        val countIterations = shape.sliceArray(0 until axis).fold(1) { acc, i -> acc * i }
        val rightKek = shape.sliceArray((axis + 1) until rank).fold(1) { acc, i -> acc * i }
        val count = shape[axis]

        val outputShape = if (keepDims) shape.copyOf().apply { set(axis, 1) } else shape.sliceArray(shape.indices.minus(axis))
        val outputArray = allocateNDArray(DataType.INT, outputShape) as MutableIntNDArray
        val tempMaxValues = if (axis == shape.lastIndex) PrimitiveTiledArray(1, 1) else PrimitiveTiledArray(rightKek, outputArray.array.blockSize)

        val inputPointer = this.array.pointer()

        for (i in 0 until countIterations) {
            var maxValuesPointer = tempMaxValues.pointer()

            maxValuesPointer.accept(inputPointer, rightKek) { _, src -> src }

            for (j in 1 until count) {
                val outputPointer = outputArray.array.pointer(i * rightKek)
                maxValuesPointer = tempMaxValues.pointer()

                var end = rightKek

                if (inputPointer.isCompatibleWith(outputPointer) && inputPointer.isCompatibleWith(maxValuesPointer)) {
                    while (end > 0) {
                        val outputBlock = outputPointer.currentBlock
                        val offset = outputPointer.indexInBlock
                        outputPointer.blockIncrement()

                        val inputBlock = inputPointer.currentBlock
                        inputPointer.blockIncrement()

                        val maxValuesBlock = maxValuesPointer.currentBlock
                        maxValuesPointer.blockIncrement()

                        for (index in offset until min(outputBlock.size, offset + end)) {
                            val value = inputBlock[index]
                            val oldMaxValue = maxValuesBlock[index]
                            if (value > oldMaxValue) {
                                maxValuesBlock[index] = value
                                outputBlock[index] = j
                            } else if (selectLastIndex && value == oldMaxValue) {
                                outputBlock[index] = j
                            }
                        }

                        end -= outputBlock.size - offset
                    }
                } else {
                    while (end > 0) {
                        val value = inputPointer.getAndIncrement()
                        val oldMaxValue = maxValuesPointer.get()

                        if (value > oldMaxValue) {
                            maxValuesPointer.set(value)
                            outputPointer.set(j)
                        } else if (selectLastIndex && value == oldMaxValue) {
                            outputPointer.set(j)
                        }

                        maxValuesPointer.increment()
                        outputPointer.increment()
                        end--
                    }
                }
            }
        }

        return outputArray
    }

    override fun copyIfNotMutable(): MutableNDArray {
        return MutablePrimitiveNDArray(array.copyOf(), strides)
    }

    @FilterPrimitives(exclude = [DataType.DOUBLE, DataType.FLOAT, DataType.BOOLEAN, DataType.BOOLEAN, DataType.INT, DataType.LONG, DataType.SHORT,
        DataType.UINT, DataType.ULONG, DataType.USHORT])
    @BindPrimitives(type1 = [DataType.BYTE, DataType.UBYTE])
    fun quantizeDot(other: @BindPrimitives.Type1 PrimitiveNDArray, destination: MutableFloatNDArray, zeroPointA: Int = 0, zeroPointB: Int = 0, scale: Float = 1f): MutableFloatNDArray {
        val M = this.shape[0]

        fun wrapper(body: (inner: () -> Unit) -> Unit = { it() }) {
            for (rdBlockNum in 0 until destination.blocksInRow) {
                body {
                    for (i in 0 until M) {
                        val dBlockOffset = i * destination.blocksInRow
                        val lBlockOffset = i * this.blocksInRow

                        var k = 0
                        for (lBlockNum in 0 until this.blocksInRow) {
                            val lBlock = this.array.blocks[lBlockOffset + lBlockNum]
                            for (lInd in lBlock.indices) {
                                val temp = lBlock[lInd].toInt() - zeroPointA
                                val rBlockOffset = k * other.blocksInRow
                                val rBlock = other.array.blocks[rBlockOffset + rdBlockNum]
                                val dBlock = destination.array.blocks[dBlockOffset + rdBlockNum]
                                for (idx in rBlock.indices) {
                                    dBlock[idx] += (temp * (rBlock[idx].toInt() - zeroPointB)) * scale
                                }
                                k++
                            }
                        }
                    }
                }
            }
        }

        if (other.blocksInRow > 1) {
            runBlocking(Dispatchers.Default) { wrapper { launch { it() } } }
        } else {
            wrapper()
        }

        return destination
    }

    override fun equals(other: Any?): Boolean {
        if (this === other) return true
        if (other !is PrimitiveNDArray) return false

        if (type != other.type) return false
        if (strides != other.strides) return false
        if (array != other.array) return false

        return true
    }

    override fun hashCode(): Int {
        var result = array.hashCode()
        result = 31 * result + strides.hashCode()
        result = 31 * result + type.hashCode()
        return result
    }

    companion object {
        val logger = logger("NDArray")

        fun scalar(value: PrimitiveType): PrimitiveNDArray {
            return PrimitiveNDArray(PrimitiveTiledArray(1, 1) { value }, Strides.EMPTY)
        }

        operator fun invoke(array: PrimitiveTiledArray, strides: Strides, customBlockSize: Int): PrimitiveNDArray {
            return if (customBlockSize == array.blockSize) {
                PrimitiveNDArray(array, strides)
            }
            else {
                val pointer = PrimitivePointer(array)
                PrimitiveNDArray(strides, customBlockSize) { pointer.getAndIncrement() }
            }
        }
    }
}

@GenerateNameFromPrimitives
interface PrimitiveMap : PrimitiveToPrimitiveFunction {
    fun apply(value: PrimitiveType): PrimitiveType
}
