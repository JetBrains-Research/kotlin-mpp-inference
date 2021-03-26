package io.kinference.runners

import io.kinference.data.tensors.Tensor
import io.kinference.model.Model
import io.kinference.ndarray.logger
import io.kinference.utils.*
import kotlin.time.ExperimentalTime
import kotlin.time.measureTime

object PerformanceRunner {
    val logger = logger("PerformanceRunner")

    data class PerformanceResults(val name: String, val avg: Double, val min: Long, val max: Long)

    private suspend fun runPerformanceFromS3(name: String, count: Int = 10, withProfiling: Boolean = false): List<PerformanceResults> {
        val toFolder = name.replace(":", "/")
        return runPerformanceFromFolder(S3TestDataLoader, toFolder, count, withProfiling)
    }

    private suspend fun runPerformanceFromResources(testPath: String, count: Int = 10, withProfiling: Boolean = false): List<PerformanceResults> {
        val path = "build/processedResources/${TestRunner.forPlatform("js", "jvm")}/test/${testPath}"
        return runPerformanceFromFolder(ResourcesTestDataLoader, path, count, withProfiling)
    }

    data class TensorDataWithName(val tensors: List<Tensor>, val test: String)

    @OptIn(ExperimentalTime::class)
    private suspend fun runPerformanceFromFolder(loader: TestDataLoader, path: String, count: Int = 10, withProfiling: Boolean = false): List<PerformanceResults> {
        val model = Model.load(loader.bytes(TestDataLoader.Path(path, "model.onnx")))
        val files = loader.text(TestDataLoader.Path(path, "descriptor.txt")).lines()
        val datasets = files.filter { "test" in it }.groupBy { file -> file.takeWhile { it != '/' } }.map { (group, files) ->
            val inputFiles = files.filter { file -> "input" in file }
            val inputTensors = inputFiles.map { DataLoader.getTensor(loader.bytes(TestDataLoader.Path(path, it))) }.toList()
            TensorDataWithName(inputTensors, group)
        }

        val results = ArrayList<PerformanceResults>()

        for (dataset in datasets) {
            val times = LongArray(count)
            for (i in (0 until count)) {
                val time = measureTime {
                    model.predict(dataset.tensors, withProfiling)
                }.inMilliseconds.toLong()
                times[i] = time
            }
            results.add(PerformanceResults(dataset.test, times.average(), times.minOrNull()!!, times.max()!!))

            if (withProfiling) {
                logger.info {
                    "Results for ${dataset.test}:" +
                    model.analyzeProfilingResults().getInfo()
                }

                model.resetProfiles()
            }
        }



        return results
    }

    suspend fun runFromS3(name: String, count: Int = 20, withProfiling: Boolean = false) {
        output(runPerformanceFromS3(name, count, withProfiling))
    }

    suspend fun runFromResources(testPath: String, count: Int = 20, withProfiling: Boolean = false) {
        output(runPerformanceFromResources(testPath, count, withProfiling))
    }

    private fun output(results: List<PerformanceResults>) {
        for (result in results.sortedBy { it.name }) {
            println("Test ${result.name}: avg ${result.avg}, min ${result.min}, max ${result.max}")
        }
    }
}
