package io.kinference.algorithms.completion.generation

import org.junit.jupiter.api.Test

class GenerationInfoTest {
    @Test
    fun testEmpty() {
        val gi = GenerationInfo()
        assert(gi.probs.isEmpty())
    }

    @Test
    fun testAdd() {
        val probs = listOf(0.1, 0.4)
        val gi = GenerationInfo(probs)
        assert(gi.probs.size == 2)
        gi.add(1, 0.3)
        assert(gi.probs.size == 3)
        assert(gi.probs[2] == 0.3)
    }

    @Test
    fun testTrim() {
        val probs = listOf(0.1, 0.4, 0.3)
        var gi = GenerationInfo(probs)
        assert(gi.probs.size == 3)
        gi = gi.trim(1, 2)
        assert(gi.probs.size == 1)
        assert(gi.probs[0] == 0.4)
    }
}
