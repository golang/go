// Copyright 2024 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

package testing_test

import (
	"math/rand/v2"
	"testing"
)

// ExBenchmark shows how to use b.Loop in a benchmark.
//
// (If this were a real benchmark, not an example, this would be named
// BenchmarkSomething.)
func ExBenchmark(b *testing.B) {
	// Generate a large random slice to use as an input.
	// Since this is done before the first call to b.Loop(),
	// it doesn't count toward the benchmark time.
	input := make([]int, 128<<10)
	for i := range input {
		input[i] = rand.Int()
	}

	// Perform the benchmark.
	for b.Loop() {
		// Normally, the compiler would be allowed to optimize away the call
		// to sum because it has no side effects and the result isn't used.
		// However, inside a b.Loop loop, the compiler ensures function calls
		// aren't optimized away.
		sum(input)
	}

	// Outside the loop, the timer is stopped, so we could perform
	// cleanup if necessary without affecting the result.
}

func sum(data []int) int {
	total := 0
	for _, value := range data {
		total += value
	}
	return total
}

func ExampleB_Loop() {
	testing.Benchmark(ExBenchmark)
}
