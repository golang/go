// Copyright 2026 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

package pprof

// Mode-specific behavioral check for the -O tiers (logos-docs SPEC
// T-105, T-302); see runtime/optmode_logos_test.go for the contract.
//
// TestGenericsHashKeyInPprofBuilder drops the result of each
// genericAllocFunc call and expects every make to reach the heap
// profile. A tier raises the variable-make stack buffer (256 bytes at
// O2, 4096 at O3), so a discarded make that fits the buffer takes no
// heap allocation and the upstream sample list cannot hold. The hook is
// driven by the stack-buffer size the checked-in runner
// (runtime/logos_mode_gate.bash) expects for the build configuration:
// unset leaves the upstream body alone, a measured size that differs from
// the expectation fails, and a tier size runs the check below. It keeps
// the issue 64528 coverage, distinct profile locations for distinct
// shapes and for the same shape at distinct sizes, in two parts:
//
//   - the upstream sizes, with each result stored in a package variable so
//     the make always reaches the heap, must produce the upstream sample
//     for every shape and size;
//   - discarded makes of two and four times the buffer size, which must
//     take the heap fallback past the buffer boundary, must produce one
//     sample per shape and size, each of one object whose size covers the
//     request.

import (
	"bytes"
	"internal/asan"
	"internal/msan"
	"internal/profile"
	"internal/race"
	"os"
	"runtime"
	"slices"
	"strconv"
	"strings"
	"sync"
	"testing"
)

const (
	logosUpstreamStackBuf  = 32
	logosProbeLimit        = 1 << 16
	logosExpectStackBufEnv = "LOGOS_TEST_EXPECT_STACKBUF"
)

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

// logosMeasureStackBuf returns the compiled stack-buffer size in bytes,
// zero when no make avoids the heap, after checking that the boundary is
// monotone.
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

// logosTierStackBuf is the pprof twin of the runtime helper of the same
// name: the stack-buffer size the runner expects when it is not the
// upstream default, and zero when the upstream body runs.
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

var (
	logosSink32 []uint32
	logosSink64 []uint64
)

// logosGenericEscaping makes the upstream allocations with each result
// stored, so the stack buffer cannot absorb them, then drops the stores
// so the garbage collection in the test frees them.
//
//go:noinline
func logosGenericEscaping() {
	for _, sz := range []int{128, 256} {
		logosSink32 = genericAllocFunc[uint32](sz / 4)
	}
	for _, sz := range []int{64, 128} {
		logosSink64 = genericAllocFunc[uint64](sz / 8)
	}
	logosSink32, logosSink64 = nil, nil
}

// logosGenericDiscarded makes allocations of two and four buffers and
// drops each result, like the upstream body: each is too large for the
// stack buffer and must take the heap fallback.
//
//go:noinline
func logosGenericDiscarded(buf int) {
	for _, sz := range []int{2 * buf, 4 * buf} {
		genericAllocFunc[uint32](sz / 4)
	}
	for _, sz := range []int{2 * buf, 4 * buf} {
		genericAllocFunc[uint64](sz / 8)
	}
}

// logosSampleStack renders the call chain of s, outermost first, joined
// by semicolons, as sampleToString does without the values.
func logosSampleStack(s *profile.Sample) string {
	var funcs []string
	for i := len(s.Location) - 1; i >= 0; i-- {
		funcs = locationToStrings(s.Location[i], funcs)
	}
	return strings.Join(funcs, ";")
}

// logosOptGenericsHashKeyInPprofBuilder is the tier body of
// TestGenericsHashKeyInPprofBuilder; it reports whether it ran.
func logosOptGenericsHashKeyInPprofBuilder(t *testing.T) bool {
	buf := logosTierStackBuf(t)
	if buf == 0 {
		return false
	}

	previousRate := runtime.MemProfileRate
	runtime.MemProfileRate = 1
	defer func() {
		runtime.MemProfileRate = previousRate
	}()
	logosGenericEscaping()
	logosGenericDiscarded(buf)

	runtime.GC()
	out := bytes.NewBuffer(nil)
	if err := WriteHeapProfile(out); err != nil {
		t.Fatalf("writing profile: %v", err)
	}
	p, err := profile.Parse(out)
	if err != nil {
		t.Fatalf("profile.Parse: %v", err)
	}

	const (
		test    = "testing.tRunner;runtime/pprof.TestGenericsHashKeyInPprofBuilder;runtime/pprof.logosOptGenericsHashKeyInPprofBuilder;"
		escapes = test + "runtime/pprof.logosGenericEscaping;runtime/pprof.genericAllocFunc"
		dropped = test + "runtime/pprof.logosGenericDiscarded;runtime/pprof.genericAllocFunc"
	)

	// Alloc objects and alloc space of every sample, keyed by call chain.
	// Several samples can share a chain; they differ by object size.
	type counts struct{ objects, space int64 }
	byStack := map[string][]counts{}
	var all []string
	for _, s := range p.Sample {
		stack := logosSampleStack(s)
		all = append(all, sampleToString(s))
		byStack[stack] = append(byStack[stack], counts{s.Value[0], s.Value[1]})
	}

	// The upstream sizes, escaping: one object of exactly each size.
	for _, want := range []struct {
		stack string
		size  int64
	}{
		{escapes + "[go.shape.uint32]", 128},
		{escapes + "[go.shape.uint32]", 256},
		{escapes + "[go.shape.uint64]", 64},
		{escapes + "[go.shape.uint64]", 128},
	} {
		if !slices.Contains(byStack[want.stack], counts{1, want.size}) {
			t.Errorf("profile = %v\nwant one object of %d bytes at %s", strings.Join(all, "\n"), want.size, want.stack)
		}
	}

	// Discarded makes past the buffer: one object whose size covers the
	// request, separately for every shape and size.
	for _, shape := range []string{"[go.shape.uint32]", "[go.shape.uint64]"} {
		for _, size := range []int64{int64(2 * buf), int64(4 * buf)} {
			found := false
			for _, c := range byStack[dropped+shape] {
				if c.objects == 1 && c.space >= size && c.space < 2*size {
					found = true
				}
			}
			if !found {
				t.Errorf("profile = %v\nwant one object of %d to %d bytes at %s", strings.Join(all, "\n"), size, 2*size-1, dropped+shape)
			}
		}
	}
	return true
}
