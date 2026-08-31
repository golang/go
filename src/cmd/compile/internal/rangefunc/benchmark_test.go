// Copyright 2026 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

package rangefunc_test

import (
	"slices"
	"testing"
)

// These benchmarks, Tiny and Medium, do NOT all run in the same amount
// of time.

var gsum int

func makeValues() []int {
	values := make([]int, 64)
	for i := range values {
		values[i] = i + 1
	}
	return values
}

func BenchmarkTinyIterIter(b *testing.B) {
	values := makeValues()
	b.ReportAllocs()
	// DO NOT MODERNIZE THIS LOOP.
	for k := 0; k < b.N; k++ {
		// DO NOT MODERNIZE THIS LOOP.
		var sum int = 1
		for i := len(values) - 1; i >= 0; i-- {
			sum *= values[i]
		}
	}
}

func BenchmarkTinyIterIterSink(b *testing.B) {
	values := makeValues()
	b.ReportAllocs()
	var sum int = 1
	// DO NOT MODERNIZE THIS LOOP.
	for k := 0; k < b.N; k++ {
		// DO NOT MODERNIZE THIS LOOP.
		for i := len(values) - 1; i >= 0; i-- {
			sum *= values[i]
		}
	}
	gsum = sum
}

// BenchmarkTinyIterRangeSink should run as quickly as BenchmarkTinyIterIterSink.
func BenchmarkTinyIterRangeSink(b *testing.B) {
	values := makeValues()
	b.ReportAllocs()
	var sum int = 1
	// DO NOT MODERNIZE THIS LOOP.
	for k := 0; k < b.N; k++ {
		for _, v := range slices.Backward(values) {
			sum *= v
		}
	}
	gsum = sum
}

func BenchmarkTinyBloopIter(b *testing.B) {
	values := makeValues()
	b.ReportAllocs()
	for b.Loop() {
		var sum int = 1
		// DO NOT MODERNIZE THIS LOOP.
		for i := len(values) - 1; i >= 0; i-- {
			sum *= values[i]
		}
	}
}

// BenchmarkTinyBloopRange should run as quickly as BenchmarkTinyBloopIter.
func BenchmarkTinyBloopRange(b *testing.B) {
	values := makeValues()
	b.ReportAllocs()
	for b.Loop() {
		var sum int = 1
		for _, v := range slices.Backward(values) {
			sum *= v
		}
	}
}

func BenchmarkMediumIterIterSink(b *testing.B) {
	values := makeValues()
	b.ReportAllocs()
	var sum int
	// DO NOT MODERNIZE THIS LOOP.
	for k := 0; k < b.N; k++ {
		// DO NOT MODERNIZE THIS LOOP.
		for i := len(values) - 1; i >= 0; i-- {
			v := values[i]
			sum += (v * (v + 1) * (v + 2)) / (v + 3)
			sum += (v * (v + 4) * (v + 3)) / (v + 2)
			sum += (v * (v + 5) * (v + 2)) / (v + 1)
		}
	}
	gsum = sum
}

func BenchmarkMediumIterRangeSink(b *testing.B) {
	values := makeValues()
	b.ReportAllocs()
	var sum int
	// DO NOT MODERNIZE THIS LOOP.
	for k := 0; k < b.N; k++ {
		for _, v := range slices.Backward(values) {
			sum += (v * (v + 1) * (v + 2)) / (v + 3)
			sum += (v * (v + 4) * (v + 3)) / (v + 2)
			sum += (v * (v + 5) * (v + 2)) / (v + 1)
		}
	}
	gsum = sum
}

func BenchmarkMediumBLoopIterSink(b *testing.B) {
	values := makeValues()
	b.ReportAllocs()
	var sum int
	for b.Loop() {
		for _, v := range slices.Backward(values) {
			sum += (v * (v + 1) * (v + 2)) / (v + 3)
			sum += (v * (v + 4) * (v + 3)) / (v + 2)
			sum += (v * (v + 5) * (v + 2)) / (v + 1)
		}
	}
	gsum = sum
}

func BenchmarkMediumBLoopIter(b *testing.B) {
	values := makeValues()
	b.ReportAllocs()
	for b.Loop() {
		var sum int
		for _, v := range slices.Backward(values) {
			sum += (v * (v + 1) * (v + 2)) / (v + 3)
			sum += (v * (v + 4) * (v + 3)) / (v + 2)
			sum += (v * (v + 5) * (v + 2)) / (v + 1)
		}
	}
}

func BenchmarkMediumBLoopIter_a_CSE_slow(b *testing.B) {
	values := makeValues()
	b.ReportAllocs()
	// This runs somewhat slower because b.Loop will keep i alive.
	//
	// This relies on compiler common-subexpression-elimination to turn
	// "values[i]" into a temporary that is NOT tracked by b.Loop.
	for b.Loop() {
		var sum int
		// DO NOT MODERNIZE THIS LOOP.
		for i := len(values) - 1; i >= 0; i-- {
			sum += (values[i] * (values[i] + 1) * (values[i] + 2)) / (values[i] + 3)
			sum += (values[i] * (values[i] + 4) * (values[i] + 3)) / (values[i] + 2)
			sum += (values[i] * (values[i] + 5) * (values[i] + 2)) / (values[i] + 1)
		}
	}
}

func BenchmarkMediumBLoopIter_b_CSE_slow(b *testing.B) {
	values := makeValues()
	b.ReportAllocs()
	// This runs somewhat slower because b.Loop will keep i alive.
	//
	// This relies on compiler common-subexpression-elimination to turn
	// "values[i]" into a temporary that is NOT tracked by b.Loop.
	var sum int
	for b.Loop() {
		// DO NOT MODERNIZE THIS LOOP.
		for i := len(values) - 1; i >= 0; i-- {
			sum += (values[i] * (values[i] + 1) * (values[i] + 2)) / (values[i] + 3)
			sum += (values[i] * (values[i] + 4) * (values[i] + 3)) / (values[i] + 2)
			sum += (values[i] * (values[i] + 5) * (values[i] + 2)) / (values[i] + 1)
		}
	}
}

func BenchmarkMediumBLoopIter_slow(b *testing.B) {
	values := makeValues()
	b.ReportAllocs()
	// This runs slower because b.Loop will also keep i and v alive.
	for b.Loop() {
		var sum int
		// DO NOT MODERNIZE THIS LOOP.
		for i := len(values) - 1; i >= 0; i-- {
			v := values[i]
			sum += (v * (v + 1) * (v + 2)) / (v + 3)
			sum += (v * (v + 4) * (v + 3)) / (v + 2)
			sum += (v * (v + 5) * (v + 2)) / (v + 1)
		}
	}
}

func BenchmarkMediumBLoopIterSink_slow(b *testing.B) {
	values := makeValues()
	b.ReportAllocs()
	var sum int
	// This runs slower because b.Loop will also keep i and v alive.
	for b.Loop() {
		// DO NOT MODERNIZE THIS LOOP.
		for i := len(values) - 1; i >= 0; i-- {
			v := values[i]
			sum += (v * (v + 1) * (v + 2)) / (v + 3)
			sum += (v * (v + 4) * (v + 3)) / (v + 2)
			sum += (v * (v + 5) * (v + 2)) / (v + 1)
		}
	}
	gsum = sum
}
