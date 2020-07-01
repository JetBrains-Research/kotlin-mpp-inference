package org.jetbrains.research.kotlin.mpp.inference.operators.tensor

import org.jetbrains.research.kotlin.mpp.inference.attributes.Attribute
import org.jetbrains.research.kotlin.mpp.inference.operators.*
import TensorProto.DataType
import org.jetbrains.research.kotlin.mpp.inference.data.tensors.*

class Constant(attributes: Map<String, Attribute<Any>>)
    : Operator<BaseTensor, BaseTensor>("Constant", attributes, ATTRIBUTES_INFO, INPUTS_INFO, OUTPUTS_INFO) {
    companion object {
        private val TYPE_CONSTRAINTS = ALL_DATA_TYPES

        private val ATTRIBUTES_INFO = listOf(
            AttributeInfo("value", setOf(AttributeProto.AttributeType.TENSOR), false),
            AttributeInfo("value_float", setOf(AttributeProto.AttributeType.FLOAT), false),
            AttributeInfo("value_floats", setOf(AttributeProto.AttributeType.FLOATS), false),
            AttributeInfo("value_int", setOf(AttributeProto.AttributeType.INT), false),
            AttributeInfo("value_ints", setOf(AttributeProto.AttributeType.INTS), false),
            AttributeInfo("value_string", setOf(AttributeProto.AttributeType.STRING), false),
            AttributeInfo("value_strings", setOf(AttributeProto.AttributeType.STRINGS), false)
            //TODO: sparse tensor values
        )

        private val INPUTS_INFO = emptyList<InputInfo>()

        private val OUTPUTS_INFO = listOf(OutputInfo(0, TYPE_CONSTRAINTS, "output"))
    }

    override fun apply(inputs: Collection<BaseTensor>, numOutputs: Int): Collection<BaseTensor> {
        //only one of all attributes is not null
        val (name, value) = ATTRIBUTES_INFO.map { it.name to getAttributeValueOrNull(it.name) }.single { it.second != null }

        val result = when(name) {
            "value" -> value
            "value_float" -> ScalarTensor("output", value!!, DataType.FLOAT)
            "value_floats" -> Tensor(value!! as List<Any>, DataType.FLOAT)
            "value_int" -> ScalarTensor("output", value!!, DataType.INT64)
            "value_ints" -> Tensor(value!! as List<Any>, DataType.INT64)
            "value_string" -> ScalarTensor("output", value!!, DataType.STRING)
            "value_strings" -> Tensor(value!! as List<Any>, DataType.STRING)
            else -> error("Unsupported data type")
        } as BaseTensor
        return listOf(result)
    }
}
