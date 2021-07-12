package io.kinference.data.tensors

import io.kinference.data.ONNXData
import io.kinference.data.ONNXDataType
import io.kinference.ndarray.Strides
import io.kinference.ndarray.arrays.*
import io.kinference.ndarray.arrays.tiled.*
import io.kinference.ndarray.extensions.matmul
import io.kinference.protobuf.message.TensorProto
import io.kinference.protobuf.message.TensorProto.DataType
import io.kinference.types.ValueInfo

//TODO: support segments
//TODO: support external data
class Tensor(val data: NDArray, info: ValueInfo) : ONNXData(ONNXDataType.ONNX_TENSOR, info) {
    override fun rename(name: String): ONNXData {
        return Tensor(data, ValueInfo(info.typeInfo, name))
    }

    operator fun plus(other: Tensor): Tensor {
        require(this.data is NumberNDArray && other.data is NumberNDArray)
        return (this.data + other.data).asTensor()
    }

    operator fun minus(other: Tensor): Tensor {
        require(this.data is NumberNDArray && other.data is NumberNDArray)
        return (this.data - other.data).asTensor()
    }

    operator fun times(other: Tensor): Tensor {
        require(this.data is NumberNDArray && other.data is NumberNDArray)
        return (this.data * other.data).asTensor()
    }

    infix fun matmul(other: Tensor): Tensor {
        require(this.data is NumberNDArray && other.data is NumberNDArray)
        return (this.data matmul other.data).asTensor()
    }

    companion object {
        //TODO: complex, uint32/64 tensors
        @Suppress("UNCHECKED_CAST")
        fun create(proto: TensorProto): Tensor {
            val type = proto.dataType ?: DataType.UNDEFINED
            val array = parseArray(proto)
            requireNotNull(array) { "Array value should be initialized" }

            return Tensor(array, type, proto.dims, proto.name)
        }

        fun create(proto: TensorProto, customBlockSize: Int): Tensor {
            val type = proto.dataType ?: DataType.UNDEFINED
            val array = parseArray(proto)
            requireNotNull(array) { "Array value should be initialized" }

            return Tensor(array, type, proto.dims, proto.name, customBlockSize)
        }

        private operator fun invoke(value: Any, type: DataType, dims: IntArray = IntArray(0), name: String? = ""): Tensor {
            val name = name ?: ""
            val strides = Strides(dims)
            return when (type) {
                DataType.DOUBLE -> DoubleNDArray(value as DoubleTiledArray, strides).asTensor(name)
                DataType.FLOAT -> FloatNDArray(value as FloatTiledArray, strides).asTensor(name)
                DataType.INT32 -> IntNDArray(value as IntTiledArray, strides).asTensor(name)
                DataType.INT8 -> ByteNDArray(value as ByteTiledArray, strides).asTensor(name)
                DataType.UINT8 -> UByteNDArray(value as UByteTiledArray, strides).asTensor(name)
                DataType.INT64 -> LongNDArray(value as LongTiledArray, strides).asTensor(name)
                DataType.INT16 -> ShortNDArray(value as ShortTiledArray, strides).asTensor(name)
                DataType.BOOL -> BooleanNDArray(value as BooleanTiledArray, strides).asTensor(name)
                DataType.STRING -> {
                    value as List<String>
                    StringNDArray(dims) { value[it] }.asTensor(name)
                }
                else -> error("Unsupported data type $type")
            }
        }

        private operator fun invoke(value: Any, type: DataType, dims: IntArray = IntArray(0), name: String? = "", customBlockSize: Int): Tensor {
            val name = name ?: ""
            val strides = Strides(dims)
            return when (type) {
                DataType.DOUBLE -> DoubleNDArray(value as DoubleTiledArray, strides, customBlockSize).asTensor(name)
                DataType.FLOAT -> FloatNDArray(value as FloatTiledArray, strides, customBlockSize).asTensor(name)
                DataType.INT32 -> IntNDArray(value as IntTiledArray, strides, customBlockSize).asTensor(name)
                DataType.INT8 -> ByteNDArray(value as ByteTiledArray, strides, customBlockSize).asTensor(name)
                DataType.UINT8 -> UByteNDArray(value as UByteTiledArray, strides, customBlockSize).asTensor(name)
                DataType.INT64 -> LongNDArray(value as LongTiledArray, strides, customBlockSize).asTensor(name)
                DataType.INT16 -> ShortNDArray(value as ShortTiledArray, strides, customBlockSize).asTensor(name)
                DataType.BOOL -> BooleanNDArray(value as BooleanTiledArray, strides, customBlockSize).asTensor(name)
                DataType.STRING -> {
                    value as List<String>
                    StringNDArray(dims) { value[it] }.asTensor(name)
                }
                else -> error("Unsupported data type $type")
            }
        }

        private fun parseArray(proto: TensorProto) = when {
            proto.isTiled() -> proto.tiledData
            proto.isString() -> proto.stringData
            else -> error("Unsupported data type ${proto.dataType}")
        }
    }
}
