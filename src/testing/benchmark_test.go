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
	"time"
)

var prettyPrintTests = []struct {
	v        float64
	expected string
}{
	{0, "         0 x"},
	{1234.1, "      1234 x"},
	{-1234.1, "     -1234 x"},
	{999.950001, "      1000 x"},
	{999.949999, "       999.9 x"},
	{99.9950001, "       100.0 x"},
	{99.9949999, "        99.99 x"},
	{-99.9949999, "       -99.99 x"},
	{0.000999950001, "         0.001000 x"},
	{0.000999949999, "         0.0009999 x"}, // smallest case
	{0.0000999949999, "         0.0001000 x"},
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

func TestResultString(t *testing.T) {
	// Test fractional ns/op handling
	r := testing.BenchmarkResult{
		N: 100,
		T: 240 * time.Nanosecond,
	}
	if r.NsPerOp() != 2 {
		t.Errorf("NsPerOp: expected 2, actual %v", r.NsPerOp())
	}
	if want, got := "     100\t         2.400 ns/op", r.String(); want != got {
		t.Errorf("String: expected %q, actual %q", want, got)
	}

	// Test sub-1 ns/op (issue #31005)
	r.T = 40 * time.Nanosecond
	if want, got := "     100\t         0.4000 ns/op", r.String(); want != got {
		t.Errorf("String: expected %q, actual %q", want, got)
	}

	// Test 0 ns/op
	r.T = 0
	if want, got := "     100", r.String(); want != got {
		t.Errorf("String: expected %q, actual %q", want, got)
	}
}

func TestRunParallel(t *testing.T) {
	if testing.Short() {
		t.Skip("skipping in short mode")
	}
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
	want := "       1\t     12345 ns/op\t         0.2000 frobs/op"
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
