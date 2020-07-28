package org.jetbrains.research.kotlin.inference.extensions.primitives

fun times(left: FloatArray, right: FloatArray): FloatArray {
    require(left.size == right.size)
    val array = FloatArray(left.size)

    for (i in left.indices) array[i] = left[i] * right[i]

    return array
}

fun times(left: DoubleArray, right: DoubleArray): DoubleArray {
    require(left.size == right.size)
    val array = DoubleArray(left.size)

    for (i in left.indices) array[i] = left[i] * right[i]

    return array
}

fun times(left: IntArray, right: IntArray): IntArray {
    require(left.size == right.size)
    val array = IntArray(left.size)

    for (i in left.indices) array[i] = left[i] * right[i]

    return array
}

fun times(left: LongArray, right: LongArray): LongArray {
    require(left.size == right.size)
    val array = LongArray(left.size)

    for (i in left.indices) array[i] = left[i] * right[i]

    return array
}

fun times(left: ShortArray, right: ShortArray): ShortArray {
    require(left.size == right.size)
    val array = ShortArray(left.size)

    for (i in left.indices) array[i] = (left[i] * right[i]).toShort()

    return array
}