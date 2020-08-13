package org.jetbrains.research.kotlin.inference.operators.activations

import org.jetbrains.research.kotlin.inference.attributes.Attribute
import org.jetbrains.research.kotlin.inference.data.ndarray.NDArray
import org.jetbrains.research.kotlin.inference.extensions.functional.DoubleArrayToDoubleArray
import org.jetbrains.research.kotlin.inference.extensions.functional.FloatArrayToFloatArray
import org.jetbrains.research.kotlin.inference.onnx.TensorProto
import org.jetbrains.research.kotlin.inference.operators.IOInfo
import org.jetbrains.research.kotlin.inference.operators.OperatorInfo
import kotlin.math.exp
import kotlin.math.max

class Identity(attributes: Map<String, Attribute<Any>> = emptyMap(), inputs: List<String>, outputs: List<String>) : Activation(INFO, attributes, inputs, outputs) {
    companion object {
        private val TYPE_CONSTRAINTS = ALL_DATA_TYPES

        private val INFO = OperatorInfo("Identity", emptyMap(),
            listOf(IOInfo(0, TYPE_CONSTRAINTS, "input", optional = false)),
            listOf(IOInfo(0, TYPE_CONSTRAINTS, "output", optional = false))
        )
    }

    override fun activate(input: NDArray<Any>): NDArray<Any> = input
}

class Relu(attributes: Map<String, Attribute<Any>> = emptyMap(), inputs: List<String>, outputs: List<String>) : Activation(INFO, attributes, inputs, outputs) {
    companion object {
        private val TYPE_CONSTRAINTS = FLOAT_DATA_TYPES

        private val INFO = OperatorInfo("Relu", emptyMap(),
            listOf(IOInfo(0, TYPE_CONSTRAINTS, "input", optional = false)),
            listOf(IOInfo(0, TYPE_CONSTRAINTS, "output", optional = false))
        )

        val activateFloat = FloatArrayToFloatArray { array ->
            for (i in array.indices) array[i] = max(0.0f, array[i])
            array
        }

        val activateDouble = DoubleArrayToDoubleArray { array ->
            for (i in array.indices) array[i] = max(0.0, array[i])
            array
        }
    }

    override fun activate(input: NDArray<Any>): NDArray<Any> = when (input.type) {
        TensorProto.DataType.FLOAT -> input.mapElements(activateFloat)
        TensorProto.DataType.DOUBLE -> input.mapElements(activateDouble)
        else -> error("Unsupported operation")
    }
}

class Sigmoid(attributes: Map<String, Attribute<Any>> = emptyMap(), inputs: List<String>, outputs: List<String>) : Activation(INFO, attributes, inputs, outputs) {
    companion object {
        private val TYPE_CONSTRAINTS = FLOAT_DATA_TYPES

        private val INFO = OperatorInfo("Sigmoid", emptyMap(),
            listOf(IOInfo(0, TYPE_CONSTRAINTS, "input", optional = false)),
            listOf(IOInfo(0, TYPE_CONSTRAINTS, "output", optional = false))
        )

        val activateFloat = FloatArrayToFloatArray { array ->
            for (i in array.indices) array[i] = 1.0f / (1.0f + exp(-array[i]))
            array
        }

        val activateDouble = DoubleArrayToDoubleArray { array ->
            for (i in array.indices) array[i] = 1.0 / (1.0 + exp(-array[i]))
            array
        }
    }

    override fun activate(input: NDArray<Any>): NDArray<Any> = when (input.type) {
        TensorProto.DataType.FLOAT -> input.mapElements(activateFloat)
        TensorProto.DataType.DOUBLE -> input.mapElements(activateDouble)
        else -> error("Unsupported operation")
    }
}

class Tanh(attributes: Map<String, Attribute<Any>> = emptyMap(), inputs: List<String>, outputs: List<String>) : Activation(INFO, attributes, inputs, outputs) {
    companion object {
        private val TYPE_CONSTRAINTS = FLOAT_DATA_TYPES

        private val INFO = OperatorInfo("Tanh", emptyMap(),
            listOf(IOInfo(0, TYPE_CONSTRAINTS, "input", optional = false)),
            listOf(IOInfo(0, TYPE_CONSTRAINTS, "output", optional = false))
        )

        val activateFloat = FloatArrayToFloatArray { array ->
            for (i in array.indices) {
                var temp = exp(2.0f * array[i])
                if (temp.isInfinite()) temp = Float.MAX_VALUE
                array[i] = ((temp - 1.0f) / (temp + 1.0f))
            }
            array
        }

        val activateDouble = DoubleArrayToDoubleArray { array ->
            for (i in array.indices) array[i] = ((exp(2.0 * array[i]) - 1.0) / (exp(2.0 * array[i]) + 1.0))
            array
        }
    }

    override fun activate(input: NDArray<Any>): NDArray<Any> = when (input.type) {
        TensorProto.DataType.FLOAT -> input.mapElements(activateFloat)
        TensorProto.DataType.DOUBLE -> input.mapElements(activateDouble)
        else -> error("Unsupported operation")
    }
}
