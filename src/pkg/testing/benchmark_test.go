// Copyright 2013 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

package testing_test

import (
	"bytes"
	"runtime"
	"sync/atomic"
	"testing"
	"text/template"
)

var roundDownTests = []struct {
	v, expected int
}{
	{1, 1},
	{9, 1},
	{10, 10},
	{11, 10},
	{100, 100},
	{101, 100},
	{999, 100},
	{1000, 1000},
	{1001, 1000},
}

func TestRoundDown10(t *testing.T) {
	for _, tt := range roundDownTests {
		actual := testing.RoundDown10(tt.v)
		if tt.expected != actual {
			t.Errorf("roundDown10(%d): expected %d, actual %d", tt.v, tt.expected, actual)
		}
	}
}

var roundUpTests = []struct {
	v, expected int
}{
	{0, 1},
	{1, 1},
	{2, 2},
	{5, 5},
	{9, 10},
	{999, 1000},
	{1000, 1000},
	{1400, 2000},
	{1700, 2000},
	{4999, 5000},
	{5000, 5000},
	{5001, 10000},
}

func TestRoundUp(t *testing.T) {
	for _, tt := range roundUpTests {
		actual := testing.RoundUp(tt.v)
		if tt.expected != actual {
			t.Errorf("roundUp(%d): expected %d, actual %d", tt.v, tt.expected, actual)
		}
	}
}

func TestRunParallel(t *testing.T) {
	testing.Benchmark(func(b *testing.B) {
		procs := uint32(0)
		iters := uint64(0)
		b.SetParallelism(3)
		b.RunParallel(func(pb *testing.PB) {
			atomic.AddUint32(&procs, 1)
			for pb.Next() {
				atomic.AddUint64(&iters, 1)
			}
		})
		if want := uint32(3 * runtime.GOMAXPROCS(0)); procs != want {
			t.Errorf("got %v procs, want %v", procs, want)
		}
		if iters != uint64(b.N) {
			t.Errorf("got %v iters, want %v", iters, b.N)
		}
	})
}

func TestRunParallelFail(t *testing.T) {
	testing.Benchmark(func(b *testing.B) {
		b.RunParallel(func(pb *testing.PB) {
			// The function must be able to log/abort
			// w/o crashing/deadlocking the whole benchmark.
			b.Log("log")
			b.Error("error")
			b.Fatal("fatal")
		})
	})
}

func ExampleB_RunParallel() {
	// Parallel benchmark for text/template.Template.Execute on a single object.
	testing.Benchmark(func(b *testing.B) {
		templ := template.Must(template.New("test").Parse("Hello, {{.}}!"))
		// RunParallel will create GOMAXPROCS goroutines
		// and distribute work among them.
		b.RunParallel(func(pb *testing.PB) {
			// Each goroutine has its own bytes.Buffer.
			var buf bytes.Buffer
			for pb.Next() {
				// The loop body is executed b.N times total across all goroutines.
				buf.Reset()
				templ.Execute(&buf, "World")
			}
		})
	})
}
