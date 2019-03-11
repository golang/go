// Copyright 2013 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

package testing_test

import (
	"bytes"
	"runtime"
	"sort"
	"strings"
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
	{3, 3},
	{5, 5},
	{9, 10},
	{999, 1000},
	{1000, 1000},
	{1400, 2000},
	{1700, 2000},
	{2700, 3000},
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

var prettyPrintTests = []struct {
	v        float64
	expected string
}{
	{0, "         0 x"},
	{1234.1, "      1234 x"},
	{-1234.1, "     -1234 x"},
	{99.950001, "       100 x"},
	{99.949999, "        99.9 x"},
	{9.9950001, "        10.0 x"},
	{9.9949999, "         9.99 x"},
	{-9.9949999, "        -9.99 x"},
	{0.0099950001, "         0.0100 x"},
	{0.0099949999, "         0.00999 x"},
}

func TestPrettyPrint(t *testing.T) {
	for _, tt := range prettyPrintTests {
		buf := new(strings.Builder)
		testing.PrettyPrint(buf, tt.v, "x")
		if tt.expected != buf.String() {
			t.Errorf("prettyPrint(%v): expected %q, actual %q", tt.v, tt.expected, buf.String())
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

func TestReportMetric(t *testing.T) {
	res := testing.Benchmark(func(b *testing.B) {
		b.ReportMetric(12345, "ns/op")
		b.ReportMetric(0.2, "frobs/op")
	})
	// Test built-in overriding.
	if res.NsPerOp() != 12345 {
		t.Errorf("NsPerOp: expected %v, actual %v", 12345, res.NsPerOp())
	}
	// Test stringing.
	res.N = 1 // Make the output stable
	want := "       1\t     12345 ns/op\t         0.200 frobs/op"
	if want != res.String() {
		t.Errorf("expected %q, actual %q", want, res.String())
	}
}

func ExampleB_ReportMetric() {
	// This reports a custom benchmark metric relevant to a
	// specific algorithm (in this case, sorting).
	testing.Benchmark(func(b *testing.B) {
		var compares int64
		for i := 0; i < b.N; i++ {
			s := []int{5, 4, 3, 2, 1}
			sort.Slice(s, func(i, j int) bool {
				compares++
				return s[i] < s[j]
			})
		}
		// This metric is per-operation, so divide by b.N and
		// report it as a "/op" unit.
		b.ReportMetric(float64(compares)/float64(b.N), "compares/op")
	})
}
