package io.kinference.operators.math

import io.kinference.runners.AccuracyRunner
import io.kinference.utils.TestRunner
import kotlin.test.Test

class MatMulTest {
    private fun getTargetPath(dirName: String) = "/matmul/$dirName/"

    @Test
    fun test_matmul_2D()  = TestRunner.runTest {
        AccuracyRunner.runFromResources(getTargetPath("test_matmul_2d"))
    }

    @Test
    fun test_matmul_3D()  = TestRunner.runTest {
        AccuracyRunner.runFromResources(getTargetPath("test_matmul_3d"))
    }

    @Test
    fun test_matmul_4D()  = TestRunner.runTest {
        AccuracyRunner.runFromResources(getTargetPath("test_matmul_4d"))
    }
}
