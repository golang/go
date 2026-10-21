// Copyright 2026 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

package runtime_test

// Mode-specific behavioral checks for the -O tiers (logos-docs SPEC
// T-105, T-302).
//
// A tier raises the variable-make stack buffer (base.Debug
// .VariableMakeThreshold) from the upstream 32 bytes to 256 (O2) or 4096
// (O3), and raises the inline budget. Upstream tests that pin the 32-byte
// boundary in their expected values, or the code layout the 80-cost
// budget produces, then fail without any runtime fault. Each such test
// keeps its exact upstream body and calls one hook from this file first.
//
// The expected mode never comes from observing the optimization. The
// checked-in runner logos_mode_gate.bash pairs each build configuration
// (-O=1, -O=2, -O=3) with the stack-buffer size that configuration must
// produce and passes it in LOGOS_TEST_EXPECT_STACKBUF. With the variable
// unset (an ordinary go test) every hook returns false at once and the
// upstream body runs unchanged. With it set, a hook first measures the
// stack-buffer size this package was compiled with, by finding the
// largest non-escaping make that costs no heap allocation, and fails the
// test when the measurement differs from the expectation: a tier whose
// allocation optimization regressed, or never applied, fails here and
// never falls back to the upstream path. An expected 32 (-O=1) then runs
// the upstream body; any other value runs the tier check, which asserts
// the same properties as the upstream test against the expected
// boundary instead of the default one:
//
//   - allocation counts at, below and one past the buffer boundary;
//   - capacities, derived from the heap size-class rounding that the
//     same binary reports, never from a table of constants;
//   - data integrity across the stack-to-heap move and growth;
//   - traceback resolution of a function tail.
//
// A wrong boundary, a lost element or a mis-resolved tail PC fails the
// test. The only skips are the ones the upstream tests already carry for
// -race, -asan and -msan, which remove the optimization by design.

import (
	"os"
	"runtime"
	"strconv"
	"strings"
	"sync"
	"testing"
	"unsafe"

	"internal/asan"
	"internal/msan"
	"internal/race"
	"internal/testenv"
)

// logosUpstreamStackBuf is the upstream variable-make stack buffer in
// bytes (cmd/compile/internal/base/flag.go, VariableMakeThreshold).
const logosUpstreamStackBuf = 32

// logosProbeLimit bounds the stack-buffer search: far above O3's 4096.
const logosProbeLimit = 1 << 16

// logosExpectStackBufEnv carries the stack-buffer size, in bytes, that
// the build configuration under test must produce. Only the runner sets
// it.
const logosExpectStackBufEnv = "LOGOS_TEST_EXPECT_STACKBUF"

var logosStackBufProbe struct {
	once  sync.Once
	bytes int
	err   string
}

// logosProbeMake is a non-escaping variable-size make. The compiler puts
// it on the stack exactly when n is at most the compiled buffer size.
//
//go:noinline
func logosProbeMake(n int) byte {
	b := make([]byte, n)
	b[0] = 1
	return b[n-1]
}

func logosProbeAllocs(n int) float64 {
	return testing.AllocsPerRun(10, func() { _ = logosProbeMake(n) })
}

// logosMeasureStackBuf returns the compiled stack-buffer size in bytes:
// zero when no make of any size avoids the heap. The result is checked
// for a single monotone boundary, so a buffer that behaves differently
// at different sizes reports an error instead of a number.
func logosMeasureStackBuf() (int, string) {
	if logosProbeAllocs(1) != 0 {
		return 0, ""
	}
	if logosProbeAllocs(logosProbeLimit) == 0 {
		return 0, "a make of 64 KiB took no heap allocation"
	}
	lo, hi := 1, logosProbeLimit // lo is stack, hi is heap
	for hi-lo > 1 {
		mid := lo + (hi-lo)/2
		if logosProbeAllocs(mid) == 0 {
			lo = mid
		} else {
			hi = mid
		}
	}
	probes := []struct {
		n     int
		stack bool
	}{{lo, true}, {lo + 1, false}, {lo/2 + 1, true}, {2*lo + 1, false}}
	for _, p := range probes {
		if got := logosProbeAllocs(p.n) == 0; got != p.stack {
			return 0, "stack-buffer boundary is not monotone around " + strconv.Itoa(lo) + " bytes"
		}
	}
	return lo, ""
}

// logosTierStackBuf returns the stack-buffer size in bytes the runner
// expects when it differs from the upstream default, and zero when the
// upstream body should run (no runner, or an expected upstream size). A
// compiled size that differs from the expectation fails the test.
func logosTierStackBuf(t *testing.T) int {
	t.Helper()
	v := os.Getenv(logosExpectStackBufEnv)
	if v == "" {
		return 0
	}
	want, err := strconv.Atoi(v)
	if err != nil || want <= 0 {
		t.Fatalf("%s=%q: want a positive byte count", logosExpectStackBufEnv, v)
	}
	if race.Enabled || asan.Enabled || msan.Enabled {
		t.Skip("instrumentation breaks this optimization")
	}
	p := &logosStackBufProbe
	p.once.Do(func() { p.bytes, p.err = logosMeasureStackBuf() })
	if p.err != "" {
		t.Fatalf("measuring the compiled stack buffer: %s", p.err)
	}
	if p.bytes != want {
		t.Fatalf("compiled variable-make stack buffer is %d bytes, this build configuration must give %d", p.bytes, want)
	}
	if want == logosUpstreamStackBuf {
		return 0
	}
	return want
}

// logosHeapCapBytes and logosHeapCapPtrs report the capacity that a heap
// append of n elements receives: the allocator's size-class rounding for
// the element kind, which a stack buffer that is moved to the heap or a
// growslice from nil must reproduce exactly. The append from nil is a
// capacity probe, not a copy: bytes.Clone would drop the rounded
// capacity this reads.
//
//go:noinline
func logosHeapCapBytes(n int) int {
	return cap(append([]byte(nil), make([]byte, n)...))
}

//go:noinline
func logosHeapCapPtrs(n int) int {
	return cap(append([]*byte(nil), make([]*byte, n)...))
}

// logosLocalAppendBytes and logosLocalAppendPtrs append n elements to a
// slice that never leaves the frame. They take no heap allocation while
// the slice fits the compiled stack buffer and at least one beyond it.
//
//go:noinline
func logosLocalAppendBytes(n int) int {
	var r []byte
	for i := range n {
		r = append(r, byte(i))
	}
	return len(r)
}

//go:noinline
func logosLocalAppendPtrs(n int) int {
	var r []*byte
	for range n {
		r = append(r, nil)
	}
	return len(r)
}

// logosSweep lists the sizes in [0, k] the in-buffer checks measure: every
// size up to 64, and the neighbours of every power of two up to k, which
// cover each size-class edge below the buffer and the boundary itself.
func logosSweep(k int) []int {
	seen := map[int]bool{}
	var out []int
	add := func(n int) {
		if n >= 0 && n <= k && !seen[n] {
			seen[n] = true
			out = append(out, n)
		}
	}
	for n := range min(k, 64) + 1 {
		add(n)
	}
	for p := 1; p <= k; p *= 2 {
		add(p - 1)
		add(p)
		add(p + 1)
	}
	add(k - 1)
	add(k)
	return out
}

// logosSkipInstrumented mirrors the upstream skips of the tests it
// replaces.
func logosSkipInstrumented(t *testing.T) {
	t.Helper()
	if race.Enabled {
		t.Skip("skipping in -race mode")
	}
	if asan.Enabled || msan.Enabled {
		t.Skip("skipping in sanitizer mode")
	}
}

// logosOptAppendGrowthStack replaces TestAppendGrowthStack's
// default-buffer capacity ladder. The first append to a frame-local slice
// takes the whole k-element buffer (k = buffer/8 int64 elements), so the
// capacity is exactly k through k appends; past the buffer the slice
// grows by the runtime's rule, so the capacity covers the length and
// never shrinks. A local append loop costs no heap allocation up to k and
// at least one beyond, and every element survives the move off the
// stack.
func logosOptAppendGrowthStack(t *testing.T) bool {
	buf := logosTierStackBuf(t)
	if buf == 0 {
		return false
	}
	if race.Enabled || asan.Enabled || msan.Enabled {
		t.Skip("instrumentation breaks this optimization")
	}
	k := buf / 8
	if k < 1 {
		t.Fatalf("stack buffer of %d bytes holds no int64", buf)
	}

	// x is only indexed and measured, never ranged over or passed, so the
	// slice stays frame-local and the first append takes the full buffer.
	var x []int64
	prev := 0
	for i := 1; i <= k+100; i++ {
		x = append(x, int64(i))
		c := cap(x)
		if i <= k {
			if c != k {
				t.Fatalf("len=%d, cap=%d, want cap=%d (a %d-byte stack buffer)", i, c, k, buf)
			}
		} else if c < i || c <= k || c < prev {
			t.Fatalf("len=%d, cap=%d, want cap >= len, cap > %d, cap >= %d", i, c, k, prev)
		}
		prev = c
	}
	for i := range len(x) {
		if x[i] != int64(i+1) {
			t.Fatalf("x[%d] = %d, want %d", i, x[i], i+1)
		}
	}

	allocs := func(n int) float64 {
		return testing.AllocsPerRun(10, func() {
			var y []int64
			for i := range n {
				y = append(y, int64(i))
			}
			if len(y) != n || y[n-1] != int64(n-1) {
				t.Errorf("appending %d elements: got len %d", n, len(y))
			}
		})
	}
	if got := allocs(k); got != 0 {
		t.Errorf("%d appends: got %f allocs, want 0 inside the %d-byte buffer", k, got, buf)
	}
	if got := allocs(k + 1); got < 1 {
		t.Errorf("%d appends: got %f allocs, want at least 1 past the %d-byte buffer", k+1, got, buf)
	}
	return true
}

// logosOptAppendByteInLoop replaces TestAppendByteInLoop's cap/alloc
// table. For n up to the buffer (k = buffer bytes) the loop costs exactly
// the one allocation of the escape move and the capacity is the heap size
// class of n, which the same binary reports for a plain heap append.
// One past the buffer costs exactly one growslice. Beyond that the count
// is bounded and the data intact. Frame-local appends mark the boundary
// itself.
func logosOptAppendByteInLoop(t *testing.T) bool {
	buf := logosTierStackBuf(t)
	if buf == 0 {
		return false
	}
	testenv.SkipIfOptimizationOff(t)
	logosSkipInstrumented(t)
	k := buf

	for _, n := range logosSweep(k) {
		wantAllocs := 1.0
		if n == 0 {
			wantAllocs = 0
		}
		var r []byte
		got := testing.AllocsPerRun(10, func() { r = byteSlice(n) })
		if got != wantAllocs {
			t.Errorf("for size %d, got %f allocs, want %v", n, got, wantAllocs)
		}
		wantCap := 0
		if n > 0 {
			wantCap = logosHeapCapBytes(n)
		}
		if cap(r) != wantCap {
			t.Errorf("for size %d, got capacity %d, want %d", n, cap(r), wantCap)
		}
		logosCheckBytes(t, "byteSlice", n, r)
	}
	logosCheckPastBuffer(t, k, func(n int) (float64, int) {
		var r []byte
		a := testing.AllocsPerRun(10, func() { r = byteSlice(n) })
		logosCheckBytes(t, "byteSlice", n, r)
		return a, cap(r)
	})
	if got := testing.AllocsPerRun(10, func() { _ = logosLocalAppendBytes(k) }); got != 0 {
		t.Errorf("%d local appends: got %f allocs, want 0 inside the %d-byte buffer", k, got, buf)
	}
	if got := testing.AllocsPerRun(10, func() { _ = logosLocalAppendBytes(k + 1) }); got < 1 {
		t.Errorf("%d local appends: got %f allocs, want at least 1 past the buffer", k+1, got)
	}
	return true
}

// logosOptAppendPtrInLoop is logosOptAppendByteInLoop for pointer
// elements, whose buffer holds k = buffer/PtrSize elements and whose heap
// size class carries the malloc header above 512 bytes.
func logosOptAppendPtrInLoop(t *testing.T) bool {
	buf := logosTierStackBuf(t)
	if buf == 0 {
		return false
	}
	testenv.SkipIfOptimizationOff(t)
	logosSkipInstrumented(t)
	k := buf / runtime.PtrSize
	if k < 1 {
		t.Fatalf("stack buffer of %d bytes holds no pointer", buf)
	}

	for _, n := range logosSweep(k) {
		wantAllocs := 1.0
		if n == 0 {
			wantAllocs = 0
		}
		var r []*byte
		got := testing.AllocsPerRun(10, func() { ptrSlice(n, &r) })
		if got != wantAllocs {
			t.Errorf("for size %d, got %f allocs, want %v", n, got, wantAllocs)
		}
		wantCap := 0
		if n > 0 {
			wantCap = logosHeapCapPtrs(n)
		}
		if cap(r) != wantCap {
			t.Errorf("for size %d, got capacity %d, want %d", n, cap(r), wantCap)
		}
		logosCheckPtrs(t, n, r)
	}
	logosCheckPastBuffer(t, k, func(n int) (float64, int) {
		var r []*byte
		a := testing.AllocsPerRun(10, func() { ptrSlice(n, &r) })
		logosCheckPtrs(t, n, r)
		return a, cap(r)
	})
	if got := testing.AllocsPerRun(10, func() { _ = logosLocalAppendPtrs(k) }); got != 0 {
		t.Errorf("%d local appends: got %f allocs, want 0 inside the %d-byte buffer", k, got, buf)
	}
	if got := testing.AllocsPerRun(10, func() { _ = logosLocalAppendPtrs(k + 1) }); got < 1 {
		t.Errorf("%d local appends: got %f allocs, want at least 1 past the buffer", k+1, got)
	}
	return true
}

// logosOptAppendByteCapInLoop is logosOptAppendByteInLoop for the
// cap-reading helper of TestAppendByteCapInLoop. Reading the capacity
// selects the growsliceBuf path, which walks the size-class ladder inside
// the buffer, so the capacity for n up to the buffer is the heap size
// class of n clamped to the buffer, and the clamp is what keeps a pointer
// store from running over the frame.
func logosOptAppendByteCapInLoop(t *testing.T) bool {
	buf := logosTierStackBuf(t)
	if buf == 0 {
		return false
	}
	testenv.SkipIfOptimizationOff(t)
	logosSkipInstrumented(t)
	k := buf

	for _, n := range logosSweep(k) {
		wantAllocs := 1.0
		if n == 0 {
			wantAllocs = 0
		}
		var r []byte
		var c int
		got := testing.AllocsPerRun(10, func() { r, c = byteCapSlice(n) })
		if got != wantAllocs {
			t.Errorf("for size %d, got %f allocs, want %v", n, got, wantAllocs)
		}
		wantCap := 0
		if n > 0 {
			wantCap = min(logosHeapCapBytes(n), k)
		}
		if c != wantCap || cap(r) != wantCap {
			t.Errorf("for size %d, got capacity %d (slice %d), want %d", n, c, cap(r), wantCap)
		}
		logosCheckBytes(t, "byteCapSlice", n, r)
	}
	logosCheckPastBuffer(t, k, func(n int) (float64, int) {
		var r []byte
		var c int
		a := testing.AllocsPerRun(10, func() { r, c = byteCapSlice(n) })
		logosCheckBytes(t, "byteCapSlice", n, r)
		if c != cap(r) {
			t.Errorf("for size %d, reported capacity %d, slice capacity %d", n, c, cap(r))
		}
		return a, c
	})
	return true
}

// logosCheckPastBuffer measures sizes past a k-element buffer through
// run, which reports the allocation count and capacity of one build of n
// elements. One element past the buffer is exactly one growslice off the
// stack buffer into a heap array that covers the length; four buffers
// out the count is positive and bounded, because each growth step is at
// least 25% for slices this large.
func logosCheckPastBuffer(t *testing.T, k int, run func(n int) (float64, int)) {
	t.Helper()
	a, c := run(k + 1)
	if a != 1 {
		t.Errorf("for size %d, got %f allocs, want 1 growth past the %d-element buffer", k+1, a, k)
	}
	if c < k+1 {
		t.Errorf("for size %d, got capacity %d, want at least %d", k+1, c, k+1)
	}
	a, c = run(4 * k)
	if a < 1 || a > 24 {
		t.Errorf("for size %d, got %f allocs, want between 1 and 24", 4*k, a)
	}
	if c < 4*k {
		t.Errorf("for size %d, got capacity %d, want at least %d", 4*k, c, 4*k)
	}
}

func logosCheckBytes(t *testing.T, name string, n int, r []byte) {
	t.Helper()
	if len(r) != n {
		t.Errorf("%s for size %d, got length %d", name, n, len(r))
		return
	}
	for i := range len(r) {
		if r[i] != byte(i) {
			t.Errorf("%s for size %d, r[%d] = %d, want %d", name, n, i, r[i], byte(i))
			return
		}
	}
}

func logosCheckPtrs(t *testing.T, n int, r []*byte) {
	t.Helper()
	if len(r) != n {
		t.Errorf("ptrSlice for size %d, got length %d", n, len(r))
		return
	}
	for i := range len(r) {
		if r[i] != nil {
			t.Errorf("ptrSlice for size %d, r[%d] = %p, want nil", n, i, r[i])
			return
		}
	}
}

// logosOptFunctionAlignmentTraceback replaces
// TestFunctionAlignmentTraceback's pointer comparison and filler-byte
// assertion. Under a tier's larger inline budget tracebackFunc's tail
// holds inlined code, whose PCs FuncForPC reports as separate inlined
// *Func values, so the upstream walk can stop inside the function; and
// the function can end exactly on the alignment boundary with no filler.
// The check walks by function entry instead, then asserts the properties
// issue 44971 needs: the last mapped PC belongs to tracebackFunc, the
// next PC is the entry of the following function, an amd64 tail is INT3
// filler or the next entry is 32-byte aligned with no filler needed, and
// CallersFrames resolves the PC to tracebackFunc without crashing.
func logosOptFunctionAlignmentTraceback(t *testing.T) bool {
	if logosTierStackBuf(t) == 0 {
		return false
	}
	pc := tracebackFunc(t)

	f := runtime.FuncForPC(pc)
	if f == nil || !strings.HasSuffix(f.Name(), "tracebackFunc") {
		t.Fatalf("Caller(0) = %+v, want tracebackFunc", f)
	}
	entry := f.Entry()

	const maxTail = 1 << 20
	for n := 0; ; n++ {
		if n > maxTail {
			t.Fatalf("no function boundary within %d bytes of %#x", maxTail, pc)
		}
		g := runtime.FuncForPC(pc)
		if g == nil || g.Entry() != entry {
			break
		}
		pc++
	}
	// pc is the first PC outside tracebackFunc; back up to its last.
	next := pc
	pc--
	if g := runtime.FuncForPC(next); g == nil || g.Entry() != next {
		t.Fatalf("PC %#x after tracebackFunc is not a function entry: %+v", next, g)
	}
	if next <= entry {
		t.Fatalf("function after tracebackFunc enters at %#x, before %#x", next, entry)
	}

	if runtime.GOARCH == "amd64" {
		const funcAlign = 32 // cmd/link/internal/amd64: funcAlign
		code := *(*uint8)(unsafe.Pointer(pc))
		if code != 0xcc && next%funcAlign != 0 { // INT $3
			t.Errorf("PC %#x code got %#x, want 0xcc filler before the unaligned entry %#x", pc, code, next)
		}
	}

	frames := runtime.CallersFrames([]uintptr{pc})
	var last runtime.Frame
	for n := 0; ; n++ {
		frame, more := frames.Next()
		if frame.Entry != entry {
			t.Errorf("frame %d of %#x has entry %#x, want tracebackFunc at %#x", n, pc, frame.Entry, entry)
		}
		last = frame
		if !more {
			break
		}
	}
	if !strings.HasSuffix(last.Function, "tracebackFunc") {
		t.Errorf("outermost frame of %#x is %q, want tracebackFunc", pc, last.Function)
	}
	return true
}
