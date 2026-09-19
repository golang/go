// Copyright 2026 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

package runtime_test

import (
	"bytes"
	"math"
	"runtime"
	"strconv"
	"strings"
	"sync"
	"testing"
)

// printlock is skipped while a goroutine's output is diverted into its own
// writebuf, so exercise that path from several goroutines at once and check
// that each one still gets exactly its own output.
func TestPrintConcurrentWritebuf(t *testing.T) {
	const goroutines = 8
	const iters = 200

	var wg sync.WaitGroup
	for g := range goroutines {
		wg.Add(1)
		go func() {
			defer wg.Done()
			for i := range iters {
				v := g*iters + i
				if got, want := runtime.DumpPrint(v), strconv.Itoa(v); got != want {
					t.Errorf("DumpPrint(%d) = %q, want %q", v, got, want)
					return
				}
			}
		}()
	}
	wg.Wait()
}

// Stack writes through the same path. Concurrent callers asking only for
// their own stack must each get one well-formed traceback.
func TestStackConcurrent(t *testing.T) {
	const goroutines = 8
	const iters = 50

	var wg sync.WaitGroup
	for range goroutines {
		wg.Add(1)
		go func() {
			defer wg.Done()
			buf := make([]byte, 8192)
			for range iters {
				s := string(buf[:runtime.Stack(buf, false)])
				if !strings.HasPrefix(s, "goroutine ") {
					t.Errorf("Stack does not begin with %q: %.64q", "goroutine ", s)
					return
				}
				if n := strings.Count(s, "\ngoroutine "); n != 0 {
					t.Errorf("Stack(all=false) reported %d other goroutines:\n%s", n, s)
					return
				}
				if !strings.Contains(s, "runtime_test.TestStackConcurrent") {
					t.Errorf("Stack is missing its own frame:\n%s", s)
					return
				}
			}
		}()
	}
	wg.Wait()
}

// Diverted output must not reach the print backlog, which is a global that
// recordForPanic maintains under printlock. If it did, concurrent Stack calls
// would race on it, and each call would also overwrite the crash context the
// backlog exists to preserve.
func TestStackLeavesPrintBacklogAlone(t *testing.T) {
	var wg sync.WaitGroup
	for range 8 {
		wg.Add(1)
		go func() {
			defer wg.Done()
			buf := make([]byte, 8192)
			for range 50 {
				runtime.Stack(buf, false)
				runtime.DumpPrint(12345)
			}
		}()
	}
	wg.Wait()

	// Look for text only a traceback produces.
	backlog := runtime.PrintBacklog()
	for _, marker := range []string{"TestStackLeavesPrintBacklogAlone", "goroutine "} {
		if bytes.Contains(backlog, []byte(marker)) {
			t.Errorf("print backlog contains traceback text %q:\n%s",
				marker, bytes.Trim(backlog, "\x00"))
		}
	}
}

func BenchmarkStack(b *testing.B) {
	buf := make([]byte, 8192)
	for b.Loop() {
		runtime.Stack(buf, false)
	}
}

func BenchmarkStackParallel(b *testing.B) {
	b.RunParallel(func(pb *testing.PB) {
		buf := make([]byte, 8192)
		for pb.Next() {
			runtime.Stack(buf, false)
		}
	})
}

func BenchmarkStackAll(b *testing.B) {
	var wg sync.WaitGroup
	stop := make(chan struct{})
	for range 32 {
		wg.Add(1)
		go func() { defer wg.Done(); <-stop }()
	}
	buf := make([]byte, 1<<16)
	for b.Loop() {
		runtime.Stack(buf, true)
	}
	close(stop)
	wg.Wait()
}

func FuzzPrintFloat64(f *testing.F) {
	f.Add(math.SmallestNonzeroFloat64)
	f.Add(math.MaxFloat64)
	f.Add(-1.7976931348623157e+308) // requires 24 digits

	f.Fuzz(func(t *testing.T, v float64) {
		s := runtime.DumpPrint(v)
		if len(s) > runtime.Float64Bytes {
			t.Errorf("print(%f) got %s (len %d) want len <= %d", v, s, len(s), runtime.Float64Bytes)
		}
	})
}

func FuzzPrintFloat32(f *testing.F) {
	f.Add(float32(math.SmallestNonzeroFloat32))
	f.Add(float32(math.MaxFloat32))
	f.Add(float32(-1.06338233e+37)) // requires 15 digits

	f.Fuzz(func(t *testing.T, v float32) {
		s := runtime.DumpPrint(v)
		if len(s) > runtime.Float32Bytes {
			t.Errorf("print(%f) got %s (len %d) want len <= %d", v, s, len(s), runtime.Float32Bytes)
		}
	})
}

func FuzzPrintComplex128(f *testing.F) {
	f.Add(math.SmallestNonzeroFloat64, math.SmallestNonzeroFloat64)
	f.Add(math.MaxFloat64, math.MaxFloat64)
	f.Add(-1.7976931348623157e+308, -1.7976931348623157e+308) // requires 51 digits

	f.Fuzz(func(t *testing.T, r, i float64) {
		v := complex(r, i)
		s := runtime.DumpPrint(v)
		if len(s) > runtime.Complex128Bytes {
			t.Errorf("print(%f) got %s (len %d) want len <= %d", v, s, len(s), runtime.Complex128Bytes)
		}
	})
}

func FuzzPrintComplex64(f *testing.F) {
	f.Add(float32(math.SmallestNonzeroFloat32), float32(math.SmallestNonzeroFloat32))
	f.Add(float32(math.MaxFloat32), float32(math.MaxFloat32))
	f.Add(float32(-1.06338233e+37), float32(-1.06338233e+37)) // requires 33 digits

	f.Fuzz(func(t *testing.T, r, i float32) {
		v := complex(r, i)
		s := runtime.DumpPrint(v)
		if len(s) > runtime.Complex64Bytes {
			t.Errorf("print(%f) got %s (len %d) want len <= %d", v, s, len(s), runtime.Complex64Bytes)
		}
	})
}
