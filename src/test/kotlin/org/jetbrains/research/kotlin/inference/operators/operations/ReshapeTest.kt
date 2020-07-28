package org.jetbrains.research.kotlin.inference.operators.operations

import org.jetbrains.research.kotlin.inference.Utils
import org.junit.jupiter.api.Test

class ReshapeTest {
    private fun getTargetPath(dirName: String) = "/reshape/$dirName/"

    @Test
    fun test_reshape_extended_dims(){
        Utils.tensorTestRunner(getTargetPath("test_reshape_extended_dims"))
    }

    @Test
    fun test_reshape_negative_dim(){
        Utils.tensorTestRunner(getTargetPath("test_reshape_negative_dim"))
    }

    @Test
    fun test_reshape_negative_extended_dims(){
        Utils.tensorTestRunner(getTargetPath("test_reshape_negative_extended_dims"))
    }

    @Test
    fun test_reshape_one_dim(){
        Utils.tensorTestRunner(getTargetPath("test_reshape_one_dim"))
    }

    @Test
    fun test_reshape_reduces_dims(){
        Utils.tensorTestRunner(getTargetPath("test_reshape_reduced_dims"))
    }

    @Test
    fun test_reshape_reordered_all_dims(){
        Utils.tensorTestRunner(getTargetPath("test_reshape_reordered_all_dims"))
    }

    @Test
    fun test_reshape_reordered_last_dims(){
        Utils.tensorTestRunner(getTargetPath("test_reshape_reordered_last_dims"))
    }

    @Test
    fun test_reshape_zero_and_negative_dim(){
        Utils.tensorTestRunner(getTargetPath("test_reshape_zero_and_negative_dim"))
    }

    @Test
    fun test_reshape_zero_dim(){
        Utils.tensorTestRunner(getTargetPath("test_reshape_zero_dim"))
    }
}