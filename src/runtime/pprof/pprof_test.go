// Copyright 2011 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

//go:build !js

package pprof

import (
	"bytes"
	"context"
	"fmt"
	"internal/abi"
	"internal/profile"
	"internal/runtime/pprof/label"
	"internal/syscall/unix"
	"internal/testenv"
	"io"
	"iter"
	"math"
	"math/big"
	"os"
	"regexp"
	"runtime"
	"runtime/debug"
	"slices"
	"strconv"
	"strings"
	"sync"
	"sync/atomic"
	"testing"
	"time"
	_ "unsafe"
)

func cpuHogger(f func(x int) int, y *int, dur time.Duration) {
	// We only need to get one 100 Hz clock tick, so we've got
	// a large safety buffer.
	// But do at least 500 iterations (which should take about 100ms),
	// otherwise TestCPUProfileMultithreaded can fail if only one
	// thread is scheduled during the testing period.
	t0 := time.Now()
	accum := *y
	for i := 0; i < 500 || time.Since(t0) < dur; i++ {
		accum = f(accum)
	}
	*y = accum
}

var (
	salt1 = 0
	salt2 = 0
)

// The actual CPU hogging function.
// Must not call other functions nor access heap/globals in the loop,
// otherwise under race detector the samples will be in the race runtime.
func cpuHog1(x int) int {
	return cpuHog0(x, 1e5)
}

func cpuHog0(x, n int) int {
	foo := x
	for i := 0; i < n; i++ {
		if foo > 0 {
			foo *= foo
		} else {
			foo *= foo + 1
		}
	}
	return foo
}

func cpuHog2(x int) int {
	foo := x
	for i := 0; i < 1e5; i++ {
		if foo > 0 {
			foo *= foo
		} else {
			foo *= foo + 2
		}
	}
	return foo
}

// Return a list of functions that we don't want to ever appear in CPU
// profiles. For gccgo, that list includes the sigprof handler itself.
func avoidFunctions() []string {
	if runtime.Compiler == "gccgo" {
		return []string{"runtime.sigprof"}
	}
	return nil
}

func TestCPUProfile(t *testing.T) {
	matches := matchAndAvoidStacks(stackContains, []string{"runtime/pprof.cpuHog1"}, avoidFunctions())
	testCPUProfile(t, matches, func(dur time.Duration) {
		cpuHogger(cpuHog1, &salt1, dur)
	})
}

func TestCPUProfileMultithreaded(t *testing.T) {
	defer runtime.GOMAXPROCS(runtime.GOMAXPROCS(2))
	matches := matchAndAvoidStacks(stackContains, []string{"runtime/pprof.cpuHog1", "runtime/pprof.cpuHog2"}, avoidFunctions())
	testCPUProfile(t, matches, func(dur time.Duration) {
		c := make(chan int)
		go func() {
			cpuHogger(cpuHog1, &salt1, dur)
			c <- 1
		}()
		cpuHogger(cpuHog2, &salt2, dur)
		<-c
	})
}

func TestCPUProfileMultithreadMagnitude(t *testing.T) {
	if runtime.GOOS != "linux" {
		t.Skip("issue 35057 is only confirmed on Linux")
	}

	defer func() {
		if t.Failed() {
			t.Logf("Failure of this test may indicate that your system suffers from a known Linux kernel bug fixed on newer kernels. See https://golang.org/issue/49065.")
		}
	}()

	// Disable on affected builders to avoid flakiness, but otherwise keep
	// it enabled to potentially warn users that they are on a broken
	// kernel.
	if testenv.Builder() != "" && (runtime.GOARCH == "386" || runtime.GOARCH == "amd64") {
		// Linux [5.9,5.16) has a kernel bug that can break CPU timers on newly
		// created threads, breaking our CPU accounting.
		if unix.KernelVersionGE(5, 9) && !unix.KernelVersionGE(5, 16) {
			testenv.SkipFlaky(t, 49065)
		}
	}

	// Run a workload in a single goroutine, then run copies of the same
	// workload in several goroutines. For both the serial and parallel cases,
	// the CPU time the process measures with its own profiler should match the
	// total CPU usage that the OS reports.
	//
	// We could also check that increases in parallelism (GOMAXPROCS) lead to a
	// linear increase in the CPU usage reported by both the OS and the
	// profiler, but without a guarantee of exclusive access to CPU resources
	// that is likely to be a flaky test.

	// Require the smaller value to be within 10%, or 40% in short mode.
	maxDiff := 0.10
	if testing.Short() {
		maxDiff = 0.40
	}

	compare := func(a, b time.Duration, maxDiff float64) error {
		if a <= 0 || b <= 0 {
			return fmt.Errorf("Expected both time reports to be positive")
		}

		if a < b {
			a, b = b, a
		}

		diff := float64(a-b) / float64(a)
		if diff > maxDiff {
			return fmt.Errorf("CPU usage reports are too different (limit -%.1f%%, got -%.1f%%)", maxDiff*100, diff*100)
		}

		return nil
	}

	for _, tc := range []struct {
		name    string
		workers int
	}{
		{
			name:    "serial",
			workers: 1,
		},
		{
			name:    "parallel",
			workers: runtime.GOMAXPROCS(0),
		},
	} {
		// check that the OS's perspective matches what the Go runtime measures.
		t.Run(tc.name, func(t *testing.T) {
			t.Logf("Running with %d workers", tc.workers)

			var userTime, systemTime time.Duration
			matches := matchAndAvoidStacks(stackContains, []string{"runtime/pprof.cpuHog1"}, avoidFunctions())
			acceptProfile := func(t *testing.T, p *profile.Profile) bool {
				if !matches(t, p) {
					return false
				}

				ok := true
				for i, unit := range []string{"count", "nanoseconds"} {
					if have, want := p.SampleType[i].Unit, unit; have != want {
						t.Logf("pN SampleType[%d]; %q != %q", i, have, want)
						ok = false
					}
				}

				// cpuHog1 called below is the primary source of CPU
				// load, but there may be some background work by the
				// runtime. Since the OS rusage measurement will
				// include all work done by the process, also compare
				// against all samples in our profile.
				var value time.Duration
				for _, sample := range p.Sample {
					value += time.Duration(sample.Value[1]) * time.Nanosecond
				}

				totalTime := userTime + systemTime
				t.Logf("compare %s user + %s system = %s vs %s", userTime, systemTime, totalTime, value)
				if err := compare(totalTime, value, maxDiff); err != nil {
					t.Logf("compare got %v want nil", err)
					ok = false
				}

				return ok
			}

			testCPUProfile(t, acceptProfile, func(dur time.Duration) {
				userTime, systemTime = diffCPUTime(t, func() {
					var wg sync.WaitGroup
					var once sync.Once
					for i := 0; i < tc.workers; i++ {
						wg.Add(1)
						go func() {
							defer wg.Done()
							var salt = 0
							cpuHogger(cpuHog1, &salt, dur)
							once.Do(func() { salt1 = salt })
						}()
					}
					wg.Wait()
				})
			})
		})
	}
}

// containsInlinedCall reports whether the function body for the function f is
// known to contain an inlined function call within the first maxBytes bytes.
func containsInlinedCall(f any, maxBytes int) bool {
	_, found := findInlinedCall(f, maxBytes)
	return found
}

// findInlinedCall returns the PC of an inlined function call within
// the function body for the function f if any.
func findInlinedCall(f any, maxBytes int) (pc uint64, found bool) {
	fFunc := runtime.FuncForPC(uintptr(abi.FuncPCABIInternal(f)))
	if fFunc == nil || fFunc.Entry() == 0 {
		panic("failed to locate function entry")
	}

	for offset := 0; offset < maxBytes; offset++ {
		innerPC := fFunc.Entry() + uintptr(offset)
		inner := runtime.FuncForPC(innerPC)
		if inner == nil {
			// No function known for this PC value.
			// It might simply be misaligned, so keep searching.
			continue
		}
		if inner.Entry() != fFunc.Entry() {
			// Scanned past f and didn't find any inlined functions.
			break
		}
		if inner.Name() != fFunc.Name() {
			// This PC has f as its entry-point, but is not f. Therefore, it must be a
			// function inlined into f.
			return uint64(innerPC), true
		}
	}

	return 0, false
}

func TestCPUProfileInlining(t *testing.T) {
	if !containsInlinedCall(inlinedCaller, 4<<10) {
		t.Skip("Can't determine whether inlinedCallee was inlined into inlinedCaller.")
	}

	matches := matchAndAvoidStacks(stackContains, []string{"runtime/pprof.inlinedCallee", "runtime/pprof.inlinedCaller"}, avoidFunctions())
	p := testCPUProfile(t, matches, func(dur time.Duration) {
		cpuHogger(inlinedCaller, &salt1, dur)
	})

	// Check if inlined function locations are encoded correctly. The inlinedCalee and inlinedCaller should be in one location.
	for _, loc := range p.Location {
		hasInlinedCallerAfterInlinedCallee, hasInlinedCallee := false, false
		for _, line := range loc.Line {
			if line.Function.Name == "runtime/pprof.inlinedCallee" {
				hasInlinedCallee = true
			}
			if hasInlinedCallee && line.Function.Name == "runtime/pprof.inlinedCaller" {
				hasInlinedCallerAfterInlinedCallee = true
			}
		}
		if hasInlinedCallee != hasInlinedCallerAfterInlinedCallee {
			t.Fatalf("want inlinedCallee followed by inlinedCaller, got separate Location entries:\n%v", p)
		}
	}
}

func inlinedCaller(x int) int {
	x = inlinedCallee(x, 1e5)
	return x
}

func inlinedCallee(x, n int) int {
	return cpuHog0(x, n)
}

//go:noinline
func dumpCallers(pcs []uintptr) {
	if pcs == nil {
		return
	}

	skip := 2 // Callers and dumpCallers
	runtime.Callers(skip, pcs)
}

//go:noinline
func inlinedCallerDump(pcs []uintptr) {
	inlinedCalleeDump(pcs)
}

func inlinedCalleeDump(pcs []uintptr) {
	dumpCallers(pcs)
}

type inlineWrapperInterface interface {
	dump(stack []uintptr)
}

type inlineWrapper struct {
}

func (h inlineWrapper) dump(pcs []uintptr) {
	dumpCallers(pcs)
}

func inlinedWrapperCallerDump(pcs []uintptr) {
	var h inlineWrapperInterface

	// Take the address of h, such that h.dump() call (below)
	// does not get devirtualized by the compiler.
	_ = &h

	h = &inlineWrapper{}
	h.dump(pcs)
}

func TestCPUProfileRecursion(t *testing.T) {
	matches := matchAndAvoidStacks(stackContains, []string{"runtime/pprof.inlinedCallee", "runtime/pprof.recursionCallee", "runtime/pprof.recursionCaller"}, avoidFunctions())
	p := testCPUProfile(t, matches, func(dur time.Duration) {
		cpuHogger(recursionCaller, &salt1, dur)
	})

	// check the Location encoding was not confused by recursive calls.
	for i, loc := range p.Location {
		recursionFunc := 0
		for _, line := range loc.Line {
			if name := line.Function.Name; name == "runtime/pprof.recursionCaller" || name == "runtime/pprof.recursionCallee" {
				recursionFunc++
			}
		}
		if recursionFunc > 1 {
			t.Fatalf("want at most one recursionCaller or recursionCallee in one Location, got a violating Location (index: %d):\n%v", i, p)
		}
	}
}

func recursionCaller(x int) int {
	y := recursionCallee(3, x)
	return y
}

func recursionCallee(n, x int) int {
	if n == 0 {
		return 1
	}
	y := inlinedCallee(x, 1e4)
	return y * recursionCallee(n-1, x)
}

func recursionChainTop(x int, pcs []uintptr) {
	if x < 0 {
		return
	}
	recursionChainMiddle(x, pcs)
}

func recursionChainMiddle(x int, pcs []uintptr) {
	recursionChainBottom(x, pcs)
}

func recursionChainBottom(x int, pcs []uintptr) {
	// This will be called each time, we only care about the last. We
	// can't make this conditional or this function won't be inlined.
	dumpCallers(pcs)

	recursionChainTop(x-1, pcs)
}

func parseProfile(t *testing.T, valBytes []byte, f func(uintptr, []*profile.Location, map[string][]string)) *profile.Profile {
	p, err := profile.Parse(bytes.NewReader(valBytes))
	if err != nil {
		t.Fatal(err)
	}
	for _, sample := range p.Sample {
		count := uintptr(sample.Value[0])
		f(count, sample.Location, sample.Label)
	}
	return p
}

// testCPUProfile runs f under the CPU profiler, checking for some conditions specified by need,
// as interpreted by matches, and returns the parsed profile.
func testCPUProfile(t *testing.T, matches profileMatchFunc, f func(dur time.Duration)) *profile.Profile {
	switch runtime.GOOS {
	case "darwin":
		out, err := testenv.Command(t, "uname", "-a").CombinedOutput()
		if err != nil {
			t.Fatal(err)
		}
		vers := string(out)
		t.Logf("uname -a: %v", vers)
	case "plan9":
		t.Skip("skipping on plan9")
	case "wasip1":
		t.Skip("skipping on wasip1")
	}

	broken := testenv.CPUProfilingBroken()

	deadline, ok := t.Deadline()
	if broken || !ok {
		if broken && testing.Short() {
			// If it's expected to be broken, no point waiting around.
			deadline = time.Now().Add(1 * time.Second)
		} else {
			deadline = time.Now().Add(10 * time.Second)
		}
	}

	// If we're running a long test, start with a long duration
	// for tests that try to make sure something *doesn't* happen.
	duration := 5 * time.Second
	if testing.Short() {
		duration = 100 * time.Millisecond
	}

	// Profiling tests are inherently flaky, especially on a
	// loaded system, such as when this test is running with
	// several others under go test std. If a test fails in a way
	// that could mean it just didn't run long enough, try with a
	// longer duration.
	for {
		var prof bytes.Buffer
		if err := StartCPUProfile(&prof); err != nil {
			t.Fatal(err)
		}
		f(duration)
		StopCPUProfile()

		if p, ok := profileOk(t, matches, &prof, duration); ok {
			return p
		}

		duration *= 2
		if time.Until(deadline) < duration {
			break
		}
		t.Logf("retrying with %s duration", duration)
	}

	if broken {
		t.Skipf("ignoring failure on %s/%s; see golang.org/issue/13841", runtime.GOOS, runtime.GOARCH)
	}

	// Ignore the failure if the tests are running in a QEMU-based emulator,
	// QEMU is not perfect at emulating everything.
	// IN_QEMU environmental variable is set by some of the Go builders.
	// IN_QEMU=1 indicates that the tests are running in QEMU. See issue 9605.
	if os.Getenv("IN_QEMU") == "1" {
		t.Skip("ignore the failure in QEMU; see golang.org/issue/9605")
	}
	t.FailNow()
	return nil
}

var diffCPUTimeImpl func(f func()) (user, system time.Duration)

func diffCPUTime(t *testing.T, f func()) (user, system time.Duration) {
	if fn := diffCPUTimeImpl; fn != nil {
		return fn(f)
	}
	t.Fatalf("cannot measure CPU time on GOOS=%s GOARCH=%s", runtime.GOOS, runtime.GOARCH)
	return 0, 0
}

// stackContains matches if a function named spec appears anywhere in the stack trace.
func stackContains(spec string, count uintptr, stk []*profile.Location, labels map[string][]string) bool {
	for _, loc := range stk {
		for _, line := range loc.Line {
			if strings.Contains(line.Function.Name, spec) {
				return true
			}
		}
	}
	return false
}

type sampleMatchFunc func(spec string, count uintptr, stk []*profile.Location, labels map[string][]string) bool

func profileOk(t *testing.T, matches profileMatchFunc, prof *bytes.Buffer, duration time.Duration) (_ *profile.Profile, ok bool) {
	ok = true

	var samples uintptr
	var buf strings.Builder
	p := parseProfile(t, prof.Bytes(), func(count uintptr, stk []*profile.Location, labels map[string][]string) {
		fmt.Fprintf(&buf, "%d:", count)
		fprintStack(&buf, stk)
		fmt.Fprintf(&buf, " labels: %v\n", labels)
		samples += count
		fmt.Fprintf(&buf, "\n")
	})
	t.Logf("total %d CPU profile samples collected:\n%s", samples, buf.String())

	if samples < 10 && runtime.GOOS == "windows" {
		// On some windows machines we end up with
		// not enough samples due to coarse timer
		// resolution. Let it go.
		t.Log("too few samples on Windows (golang.org/issue/10842)")
		return p, false
	}

	// Check that we got a reasonable number of samples.
	// We used to always require at least ideal/4 samples,
	// but that is too hard to guarantee on a loaded system.
	// Now we accept 10 or more samples, which we take to be
	// enough to show that at least some profiling is occurring.
	if ideal := uintptr(duration * 100 / time.Second); samples == 0 || (samples < ideal/4 && samples < 10) {
		t.Logf("too few samples; got %d, want at least %d, ideally %d", samples, ideal/4, ideal)
		ok = false
	}

	if matches != nil && !matches(t, p) {
		ok = false
	}

	return p, ok
}

type profileMatchFunc func(*testing.T, *profile.Profile) bool

func matchAndAvoidStacks(matches sampleMatchFunc, need []string, avoid []string) profileMatchFunc {
	return func(t *testing.T, p *profile.Profile) (ok bool) {
		ok = true

		// Check that profile is well formed, contains 'need', and does not contain
		// anything from 'avoid'.
		have := make([]uintptr, len(need))
		avoidSamples := make([]uintptr, len(avoid))

		for _, sample := range p.Sample {
			count := uintptr(sample.Value[0])
			for i, spec := range need {
				if matches(spec, count, sample.Location, sample.Label) {
					have[i] += count
				}
			}
			for i, name := range avoid {
				for _, loc := range sample.Location {
					for _, line := range loc.Line {
						if strings.Contains(line.Function.Name, name) {
							avoidSamples[i] += count
						}
					}
				}
			}
		}

		for i, name := range avoid {
			bad := avoidSamples[i]
			if bad != 0 {
				t.Logf("found %d samples in avoid-function %s\n", bad, name)
				ok = false
			}
		}

		if len(need) == 0 {
			return
		}

		var total uintptr
		for i, name := range need {
			total += have[i]
			t.Logf("found %d samples in expected function %s\n", have[i], name)
		}
		if total == 0 {
			t.Logf("no samples in expected functions")
			ok = false
		}

		// We'd like to check a reasonable minimum, like
		// total / len(have) / smallconstant, but this test is
		// pretty flaky (see bug 7095).  So we'll just test to
		// make sure we got at least one sample.
		min := uintptr(1)
		for i, name := range need {
			if have[i] < min {
				t.Logf("%s has %d samples out of %d, want at least %d, ideally %d", name, have[i], total, min, total/uintptr(len(have)))
				ok = false
			}
		}
		return
	}
}

// Fork can hang if preempted with signals frequently enough (see issue 5517).
// Ensure that we do not do this.
func TestCPUProfileWithFork(t *testing.T) {
	testenv.MustHaveExec(t)

	exe, err := os.Executable()
	if err != nil {
		t.Fatal(err)
	}

	heap := 1 << 30
	if runtime.GOOS == "android" {
		// Use smaller size for Android to avoid crash.
		heap = 100 << 20
	}
	if testing.Short() {
		heap = 100 << 20
	}
	// This makes fork slower.
	garbage := make([]byte, heap)
	// Need to touch the slice, otherwise it won't be paged in.
	done := make(chan bool)
	go func() {
		for i := range garbage {
			garbage[i] = 42
		}
		done <- true
	}()
	<-done

	var prof bytes.Buffer
	if err := StartCPUProfile(&prof); err != nil {
		t.Fatal(err)
	}
	defer StopCPUProfile()

	for i := 0; i < 10; i++ {
		testenv.Command(t, exe, "-h").CombinedOutput()
	}
}

// Test that profiler does not observe runtime.gogo as "user" goroutine execution.
// If it did, it would see inconsistent state and would either record an incorrect stack
// or crash because the stack was malformed.
func TestGoroutineSwitch(t *testing.T) {
	if runtime.Compiler == "gccgo" {
		t.Skip("not applicable for gccgo")
	}
	// How much to try. These defaults take about 1 seconds
	// on a 2012 MacBook Pro. The ones in short mode take
	// about 0.1 seconds.
	tries := 10
	count := 1000000
	if testing.Short() {
		tries = 1
	}
	for try := 0; try < tries; try++ {
		var prof bytes.Buffer
		if err := StartCPUProfile(&prof); err != nil {
			t.Fatal(err)
		}
		for i := 0; i < count; i++ {
			runtime.Gosched()
		}
		StopCPUProfile()

		// Read profile to look for entries for gogo with an attempt at a traceback.
		// "runtime.gogo" is OK, because that's the part of the context switch
		// before the actual switch begins. But we should not see "gogo",
		// aka "gogo<>(SB)", which does the actual switch and is marked SPWRITE.
		parseProfile(t, prof.Bytes(), func(count uintptr, stk []*profile.Location, _ map[string][]string) {
			// An entry with two frames with 'System' in its top frame
			// exists to record a PC without a traceback. Those are okay.
			if len(stk) == 2 {
				name := stk[1].Line[0].Function.Name
				if name == "runtime._System" || name == "runtime._ExternalCode" || name == "runtime._GC" {
					return
				}
			}

			// An entry with just one frame is OK too:
			// it knew to stop at gogo.
			if len(stk) == 1 {
				return
			}

			// Otherwise, should not see gogo.
			// The place we'd see it would be the inner most frame.
			name := stk[0].Line[0].Function.Name
			if name == "gogo" {
				var buf strings.Builder
				fprintStack(&buf, stk)
				t.Fatalf("found profile entry for gogo:\n%s", buf.String())
			}
		})
	}
}

func fprintStack(w io.Writer, stk []*profile.Location) {
	if len(stk) == 0 {
		fmt.Fprintf(w, " (stack empty)")
	}
	for _, loc := range stk {
		fmt.Fprintf(w, " %#x", loc.Address)
		fmt.Fprintf(w, " (")
		for i, line := range loc.Line {
			if i > 0 {
				fmt.Fprintf(w, " ")
			}
			fmt.Fprintf(w, "%s:%d", line.Function.Name, line.Line)
		}
		fmt.Fprintf(w, ")")
	}
}

// Test that profiling of division operations is okay, especially on ARM. See issue 6681.
func TestMathBigDivide(t *testing.T) {
	testCPUProfile(t, nil, func(duration time.Duration) {
		t := time.After(duration)
		pi := new(big.Int)
		for {
			for i := 0; i < 100; i++ {
				n := big.NewInt(2646693125139304345)
				d := big.NewInt(842468587426513207)
				pi.Div(n, d)
			}
			select {
			case <-t:
				return
			default:
			}
		}
	})
}

// stackContainsAll matches if all functions in spec (comma-separated) appear somewhere in the stack trace.
func stackContainsAll(spec string, count uintptr, stk []*profile.Location, labels map[string][]string) bool {
	for _, f := range strings.Split(spec, ",") {
		if !stackContains(f, count, stk, labels) {
			return false
		}
	}
	return true
}

func TestMorestack(t *testing.T) {
	matches := matchAndAvoidStacks(stackContainsAll, []string{"runtime.newstack,runtime/pprof.growstack"}, avoidFunctions())
	testCPUProfile(t, matches, func(duration time.Duration) {
		t := time.After(duration)
		c := make(chan bool)
		for {
			go func() {
				growstack1()
				// NOTE(vsaioc): This goroutine may leak without this select.
				select {
				case c <- true:
				case <-time.After(duration):
				}
			}()
			select {
			case <-t:
				return
			case <-c:
			}
		}
	})
}

//go:noinline
func growstack1() {
	growstack(10)
}

//go:noinline
func growstack(n int) {
	var buf [8 << 18]byte
	use(buf)
	if n > 0 {
		growstack(n - 1)
	}
}

//go:noinline
func use(x [8 << 18]byte) {}

func TestBlockProfile(t *testing.T) {
	type TestCase struct {
		name string
		f    func(*testing.T)
		stk  []string
		re   string
	}
	tests := [...]TestCase{
		{
			name: "chan recv",
			f:    blockChanRecv,
			stk: []string{
				"runtime.chanrecv1",
				"runtime/pprof.blockChanRecv",
				"runtime/pprof.TestBlockProfile",
			},
			re: `
[0-9]+ [0-9]+ @( 0x[[:xdigit:]]+)+
#	0x[0-9a-f]+	runtime\.chanrecv1\+0x[0-9a-f]+	.*runtime/chan.go:[0-9]+
#	0x[0-9a-f]+	runtime/pprof\.blockChanRecv\+0x[0-9a-f]+	.*runtime/pprof/pprof_test.go:[0-9]+
#	0x[0-9a-f]+	runtime/pprof\.TestBlockProfile\+0x[0-9a-f]+	.*runtime/pprof/pprof_test.go:[0-9]+
`},
		{
			name: "chan send",
			f:    blockChanSend,
			stk: []string{
				"runtime.chansend1",
				"runtime/pprof.blockChanSend",
				"runtime/pprof.TestBlockProfile",
			},
			re: `
[0-9]+ [0-9]+ @( 0x[[:xdigit:]]+)+
#	0x[0-9a-f]+	runtime\.chansend1\+0x[0-9a-f]+	.*runtime/chan.go:[0-9]+
#	0x[0-9a-f]+	runtime/pprof\.blockChanSend\+0x[0-9a-f]+	.*runtime/pprof/pprof_test.go:[0-9]+
#	0x[0-9a-f]+	runtime/pprof\.TestBlockProfile\+0x[0-9a-f]+	.*runtime/pprof/pprof_test.go:[0-9]+
`},
		{
			name: "chan close",
			f:    blockChanClose,
			stk: []string{
				"runtime.chanrecv1",
				"runtime/pprof.blockChanClose",
				"runtime/pprof.TestBlockProfile",
			},
			re: `
[0-9]+ [0-9]+ @( 0x[[:xdigit:]]+)+
#	0x[0-9a-f]+	runtime\.chanrecv1\+0x[0-9a-f]+	.*runtime/chan.go:[0-9]+
#	0x[0-9a-f]+	runtime/pprof\.blockChanClose\+0x[0-9a-f]+	.*runtime/pprof/pprof_test.go:[0-9]+
#	0x[0-9a-f]+	runtime/pprof\.TestBlockProfile\+0x[0-9a-f]+	.*runtime/pprof/pprof_test.go:[0-9]+
`},
		{
			name: "select recv async",
			f:    blockSelectRecvAsync,
			stk: []string{
				"runtime.selectgo",
				"runtime/pprof.blockSelectRecvAsync",
				"runtime/pprof.TestBlockProfile",
			},
			re: `
[0-9]+ [0-9]+ @( 0x[[:xdigit:]]+)+
#	0x[0-9a-f]+	runtime\.selectgo\+0x[0-9a-f]+	.*runtime/select.go:[0-9]+
#	0x[0-9a-f]+	runtime/pprof\.blockSelectRecvAsync\+0x[0-9a-f]+	.*runtime/pprof/pprof_test.go:[0-9]+
#	0x[0-9a-f]+	runtime/pprof\.TestBlockProfile\+0x[0-9a-f]+	.*runtime/pprof/pprof_test.go:[0-9]+
`},
		{
			name: "select send sync",
			f:    blockSelectSendSync,
			stk: []string{
				"runtime.selectgo",
				"runtime/pprof.blockSelectSendSync",
				"runtime/pprof.TestBlockProfile",
			},
			re: `
[0-9]+ [0-9]+ @( 0x[[:xdigit:]]+)+
#	0x[0-9a-f]+	runtime\.selectgo\+0x[0-9a-f]+	.*runtime/select.go:[0-9]+
#	0x[0-9a-f]+	runtime/pprof\.blockSelectSendSync\+0x[0-9a-f]+	.*runtime/pprof/pprof_test.go:[0-9]+
#	0x[0-9a-f]+	runtime/pprof\.TestBlockProfile\+0x[0-9a-f]+	.*runtime/pprof/pprof_test.go:[0-9]+
`},
		{
			name: "mutex",
			f:    blockMutex,
			stk: []string{
				"sync.(*Mutex).Lock",
				"runtime/pprof.blockMutex",
				"runtime/pprof.TestBlockProfile",
			},
			re: `
[0-9]+ [0-9]+ @( 0x[[:xdigit:]]+)+
#	0x[0-9a-f]+	sync\.\(\*Mutex\)\.Lock\+0x[0-9a-f]+	.*sync/mutex\.go:[0-9]+
#	0x[0-9a-f]+	runtime/pprof\.blockMutex\+0x[0-9a-f]+	.*runtime/pprof/pprof_test.go:[0-9]+
#	0x[0-9a-f]+	runtime/pprof\.TestBlockProfile\+0x[0-9a-f]+	.*runtime/pprof/pprof_test.go:[0-9]+
`},
		{
			name: "cond",
			f:    blockCond,
			stk: []string{
				"sync.(*Cond).Wait",
				"runtime/pprof.blockCond",
				"runtime/pprof.TestBlockProfile",
			},
			re: `
[0-9]+ [0-9]+ @( 0x[[:xdigit:]]+)+
#	0x[0-9a-f]+	sync\.\(\*Cond\)\.Wait\+0x[0-9a-f]+	.*sync/cond\.go:[0-9]+
#	0x[0-9a-f]+	runtime/pprof\.blockCond\+0x[0-9a-f]+	.*runtime/pprof/pprof_test.go:[0-9]+
#	0x[0-9a-f]+	runtime/pprof\.TestBlockProfile\+0x[0-9a-f]+	.*runtime/pprof/pprof_test.go:[0-9]+
`},
	}

	// Generate block profile
	runtime.SetBlockProfileRate(1)
	defer runtime.SetBlockProfileRate(0)
	for _, test := range tests {
		test.f(t)
	}

	t.Run("debug=1", func(t *testing.T) {
		var w strings.Builder
		Lookup("block").WriteTo(&w, 1)
		prof := w.String()

		if !strings.HasPrefix(prof, "--- contention:\ncycles/second=") {
			t.Fatalf("Bad profile header:\n%v", prof)
		}

		if strings.HasSuffix(prof, "#\t0x0\n\n") {
			t.Errorf("Useless 0 suffix:\n%v", prof)
		}

		for _, test := range tests {
			if !regexp.MustCompile(strings.ReplaceAll(test.re, "\t", "\t+")).MatchString(prof) {
				t.Errorf("Bad %v entry, expect:\n%v\ngot:\n%v", test.name, test.re, prof)
			}
		}
	})

	t.Run("proto", func(t *testing.T) {
		// proto format
		var w bytes.Buffer
		Lookup("block").WriteTo(&w, 0)
		p, err := profile.Parse(&w)
		if err != nil {
			t.Fatalf("failed to parse profile: %v", err)
		}
		t.Logf("parsed proto: %s", p)
		if err := p.CheckValid(); err != nil {
			t.Fatalf("invalid profile: %v", err)
		}

		stks := profileStacks(p)
		for _, test := range tests {
			if !containsStack(stks, test.stk) {
				t.Errorf("No matching stack entry for %v, want %+v", test.name, test.stk)
			}
		}
	})

}

func profileStacks(p *profile.Profile) (res [][]string) {
	for _, s := range p.Sample {
		var stk []string
		for _, l := range s.Location {
			for _, line := range l.Line {
				stk = append(stk, line.Function.Name)
			}
		}
		res = append(res, stk)
	}
	return res
}

func blockRecordStacks(records []runtime.BlockProfileRecord) (res [][]string) {
	for _, record := range records {
		frames := runtime.CallersFrames(record.Stack())
		var stk []string
		for {
			frame, more := frames.Next()
			stk = append(stk, frame.Function)
			if !more {
				break
			}
		}
		res = append(res, stk)
	}
	return res
}

func containsStack(got [][]string, want []string) bool {
	for _, stk := range got {
		if len(stk) < len(want) {
			continue
		}
		for i, f := range want {
			if f != stk[i] {
				break
			}
			if i == len(want)-1 {
				return true
			}
		}
	}
	return false
}

// awaitBlockedGoroutine spins on runtime.Gosched until a runtime stack dump
// shows a goroutine in the given state with a stack frame in
// runtime/pprof.<fName>.
func awaitBlockedGoroutine(t *testing.T, state, fName string, count int) {
	re := fmt.Sprintf(`(?m)^goroutine \d+ \[%s\]:\n(?:.+\n\t.+\n)*runtime/pprof\.%s`, regexp.QuoteMeta(state), fName)
	r := regexp.MustCompile(re)

	if deadline, ok := t.Deadline(); ok {
		if d := time.Until(deadline); d > 1*time.Second {
			timer := time.AfterFunc(d-1*time.Second, func() {
				debug.SetTraceback("all")
				panic(fmt.Sprintf("timed out waiting for %#q", re))
			})
			defer timer.Stop()
		}
	}

	buf := make([]byte, 64<<10)
	for {
		runtime.Gosched()
		n := runtime.Stack(buf, true)
		if n == len(buf) {
			// Buffer wasn't large enough for a full goroutine dump.
			// Resize it and try again.
			buf = make([]byte, 2*len(buf))
			continue
		}
		if len(r.FindAll(buf[:n], -1)) >= count {
			return
		}
	}
}

func blockChanRecv(t *testing.T) {
	c := make(chan bool)
	go func() {
		awaitBlockedGoroutine(t, "chan receive", "blockChanRecv", 1)
		c <- true
	}()
	<-c
}

func blockChanSend(t *testing.T) {
	c := make(chan bool)
	go func() {
		awaitBlockedGoroutine(t, "chan send", "blockChanSend", 1)
		<-c
	}()
	c <- true
}

func blockChanClose(t *testing.T) {
	c := make(chan bool)
	go func() {
		awaitBlockedGoroutine(t, "chan receive", "blockChanClose", 1)
		close(c)
	}()
	<-c
}

func blockSelectRecvAsync(t *testing.T) {
	const numTries = 3
	c := make(chan bool, 1)
	c2 := make(chan bool, 1)
	go func() {
		for i := 0; i < numTries; i++ {
			awaitBlockedGoroutine(t, "select", "blockSelectRecvAsync", 1)
			c <- true
		}
	}()
	for i := 0; i < numTries; i++ {
		select {
		case <-c:
		case <-c2:
		}
	}
}

func blockSelectSendSync(t *testing.T) {
	c := make(chan bool)
	c2 := make(chan bool)
	go func() {
		awaitBlockedGoroutine(t, "select", "blockSelectSendSync", 1)
		<-c
	}()
	select {
	case c <- true:
	case c2 <- true:
	}
}

func blockMutex(t *testing.T) {
	var mu sync.Mutex
	mu.Lock()
	go func() {
		awaitBlockedGoroutine(t, "sync.Mutex.Lock", "blockMutex", 1)
		mu.Unlock()
	}()
	// Note: Unlock releases mu before recording the mutex event,
	// so it's theoretically possible for this to proceed and
	// capture the profile before the event is recorded. As long
	// as this is blocked before the unlock happens, it's okay.
	mu.Lock()
}

func blockMutexN(t *testing.T, n int, d time.Duration) {
	var wg sync.WaitGroup
	var mu sync.Mutex
	mu.Lock()
	go func() {
		awaitBlockedGoroutine(t, "sync.Mutex.Lock", "blockMutex", n)
		time.Sleep(d)
		mu.Unlock()
	}()
	// Note: Unlock releases mu before recording the mutex event,
	// so it's theoretically possible for this to proceed and
	// capture the profile before the event is recorded. As long
	// as this is blocked before the unlock happens, it's okay.
	for i := 0; i < n; i++ {
		wg.Add(1)
		go func() {
			defer wg.Done()
			mu.Lock()
			mu.Unlock()
		}()
	}
	wg.Wait()
}

func blockCond(t *testing.T) {
	var mu sync.Mutex
	c := sync.NewCond(&mu)
	mu.Lock()
	go func() {
		awaitBlockedGoroutine(t, "sync.Cond.Wait", "blockCond", 1)
		mu.Lock()
		c.Signal()
		mu.Unlock()
	}()
	c.Wait()
	mu.Unlock()
}

// See http://golang.org/cl/299991.
func TestBlockProfileBias(t *testing.T) {
	rate := int(1000) // arbitrary value
	runtime.SetBlockProfileRate(rate)
	defer runtime.SetBlockProfileRate(0)

	// simulate blocking events
	blockFrequentShort(rate)
	blockInfrequentLong(rate)

	var w bytes.Buffer
	Lookup("block").WriteTo(&w, 0)
	p, err := profile.Parse(&w)
	if err != nil {
		t.Fatalf("failed to parse profile: %v", err)
	}
	t.Logf("parsed proto: %s", p)

	il := float64(-1) // blockInfrequentLong duration
	fs := float64(-1) // blockFrequentShort duration
	for _, s := range p.Sample {
		for _, l := range s.Location {
			for _, line := range l.Line {
				if len(s.Value) < 2 {
					t.Fatal("block profile has less than 2 sample types")
				}

				if line.Function.Name == "runtime/pprof.blockInfrequentLong" {
					il = float64(s.Value[1])
				} else if line.Function.Name == "runtime/pprof.blockFrequentShort" {
					fs = float64(s.Value[1])
				}
			}
		}
	}
	if il == -1 || fs == -1 {
		t.Fatal("block profile is missing expected functions")
	}

	// stddev of bias from 100 runs on local machine multiplied by 10x
	const threshold = 0.2
	if bias := (il - fs) / il; math.Abs(bias) > threshold {
		t.Fatalf("bias: abs(%f) > %f", bias, threshold)
	} else {
		t.Logf("bias: abs(%f) < %f", bias, threshold)
	}
}

// blockFrequentShort produces 100000 block events with an average duration of
// rate / 10.
func blockFrequentShort(rate int) {
	for i := 0; i < 100000; i++ {
		blockevent(int64(rate/10), 1)
	}
}

// blockInfrequentLong produces 10000 block events with an average duration of
// rate.
func blockInfrequentLong(rate int) {
	for i := 0; i < 10000; i++ {
		blockevent(int64(rate), 1)
	}
}

// Used by TestBlockProfileBias.
//
//go:linkname blockevent runtime.blockevent
func blockevent(cycles int64, skip int)

func TestMutexProfile(t *testing.T) {
	// Generate mutex profile

	old := runtime.SetMutexProfileFraction(1)
	defer runtime.SetMutexProfileFraction(old)
	if old != 0 {
		t.Fatalf("need MutexProfileRate 0, got %d", old)
	}

	const (
		N = 100
		D = 100 * time.Millisecond
	)
	start := time.Now()
	blockMutexN(t, N, D)
	blockMutexNTime := time.Since(start)

	t.Run("debug=1", func(t *testing.T) {
		var w strings.Builder
		Lookup("mutex").WriteTo(&w, 1)
		prof := w.String()
		t.Logf("received profile: %v", prof)

		if !strings.HasPrefix(prof, "--- mutex:\ncycles/second=") {
			t.Errorf("Bad profile header:\n%v", prof)
		}
		prof = strings.Trim(prof, "\n")
		lines := strings.Split(prof, "\n")
		if len(lines) < 6 {
			t.Fatalf("expected >=6 lines, got %d %q\n%s", len(lines), prof, prof)
		}
		// checking that the line is like "35258904 1 @ 0x48288d 0x47cd28 0x458931"
		r2 := `^\d+ \d+ @(?: 0x[[:xdigit:]]+)+`
		if ok, err := regexp.MatchString(r2, lines[3]); err != nil || !ok {
			t.Errorf("%q didn't match %q", lines[3], r2)
		}
		r3 := "^#.*runtime/pprof.blockMutex.*$"
		if ok, err := regexp.MatchString(r3, lines[5]); err != nil || !ok {
			t.Errorf("%q didn't match %q", lines[5], r3)
		}
		t.Log(prof)
	})
	t.Run("proto", func(t *testing.T) {
		// proto format
		var w bytes.Buffer
		Lookup("mutex").WriteTo(&w, 0)
		p, err := profile.Parse(&w)
		if err != nil {
			t.Fatalf("failed to parse profile: %v", err)
		}
		t.Logf("parsed proto: %s", p)
		if err := p.CheckValid(); err != nil {
			t.Fatalf("invalid profile: %v", err)
		}

		stks := profileStacks(p)
		for _, want := range [][]string{
			{"sync.(*Mutex).Unlock", "runtime/pprof.blockMutexN.func1"},
		} {
			if !containsStack(stks, want) {
				t.Errorf("No matching stack entry for %+v", want)
			}
		}

		i := 0
		for ; i < len(p.SampleType); i++ {
			if p.SampleType[i].Unit == "nanoseconds" {
				break
			}
		}
		if i >= len(p.SampleType) {
			t.Fatalf("profile did not contain nanoseconds sample")
		}
		total := int64(0)
		for _, s := range p.Sample {
			total += s.Value[i]
		}
		// Want d to be at least N*D, but give some wiggle-room to avoid
		// a test flaking. Set an upper-bound proportional to the total
		// wall time spent in blockMutexN. Generally speaking, the total
		// contention time could be arbitrarily high when considering
		// OS scheduler delays, or any other delays from the environment:
		// time keeps ticking during these delays. By making the upper
		// bound proportional to the wall time in blockMutexN, in theory
		// we're accounting for all these possible delays.
		d := time.Duration(total)
		lo := time.Duration(N * D * 9 / 10)
		hi := time.Duration(N) * blockMutexNTime * 11 / 10
		if d < lo || d > hi {
			for _, s := range p.Sample {
				t.Logf("sample: %s", time.Duration(s.Value[i]))
			}
			t.Fatalf("profile samples total %v, want within range [%v, %v] (target: %v)", d, lo, hi, N*D)
		}
	})

	t.Run("records", func(t *testing.T) {
		// Record a mutex profile using the structured record API.
		var records []runtime.BlockProfileRecord
		for {
			n, ok := runtime.MutexProfile(records)
			if ok {
				records = records[:n]
				break
			}
			records = make([]runtime.BlockProfileRecord, n*2)
		}

		// Check that we see the same stack trace as the proto profile. For
		// historical reason we expect a runtime.goexit root frame here that is
		// omitted in the proto profile.
		stks := blockRecordStacks(records)
		want := []string{"sync.(*Mutex).Unlock", "runtime/pprof.blockMutexN.func1", "runtime.goexit"}
		if !containsStack(stks, want) {
			t.Errorf("No matching stack entry for %+v", want)
		}
	})
}

func TestMutexProfileRateAdjust(t *testing.T) {
	old := runtime.SetMutexProfileFraction(1)
	defer runtime.SetMutexProfileFraction(old)
	if old != 0 {
		t.Fatalf("need MutexProfileRate 0, got %d", old)
	}

	readProfile := func() (contentions int64, delay int64) {
		var w bytes.Buffer
		Lookup("mutex").WriteTo(&w, 0)
		p, err := profile.Parse(&w)
		if err != nil {
			t.Fatalf("failed to parse profile: %v", err)
		}
		t.Logf("parsed proto: %s", p)
		if err := p.CheckValid(); err != nil {
			t.Fatalf("invalid profile: %v", err)
		}

		for _, s := range p.Sample {
			var match, runtimeInternal bool
			for _, l := range s.Location {
				for _, line := range l.Line {
					if line.Function.Name == "runtime/pprof.blockMutex.func1" {
						match = true
					}
					if line.Function.Name == "runtime.unlock" {
						runtimeInternal = true
					}
				}
			}
			if match && !runtimeInternal {
				contentions += s.Value[0]
				delay += s.Value[1]
			}
		}
		return
	}

	blockMutex(t)
	contentions, delay := readProfile()
	if contentions == 0 { // low-resolution timers can have delay of 0 in mutex profile
		t.Fatal("did not see expected function in profile")
	}
	runtime.SetMutexProfileFraction(0)
	newContentions, newDelay := readProfile()
	if newContentions != contentions || newDelay != delay {
		t.Fatalf("sample value changed: got [%d, %d], want [%d, %d]", newContentions, newDelay, contentions, delay)
	}
}

func func1(c chan int) { <-c }
func func2(c chan int) { <-c }
func func3(c chan int) { <-c }
func func4(c chan int) { <-c }

func TestGoroutineCounts(t *testing.T) {
	// Setting GOMAXPROCS to 1 ensures we can force all goroutines to the
	// desired blocking point.
	defer runtime.GOMAXPROCS(runtime.GOMAXPROCS(1))

	c := make(chan int)
	for i := 0; i < 100; i++ {
		switch {
		case i%10 == 0:
			go func1(c)
		case i%2 == 0:
			go func2(c)
		default:
			go func3(c)
		}
		// Let goroutines block on channel
		for j := 0; j < 5; j++ {
			runtime.Gosched()
		}
	}
	ctx := context.Background()

	// ... and again, with labels this time (just with fewer iterations to keep
	// sorting deterministic).
	Do(ctx, Labels("label", "value"), func(context.Context) {
		for i := 0; i < 89; i++ {
			switch {
			case i%10 == 0:
				go func1(c)
			case i%2 == 0:
				go func2(c)
			default:
				go func3(c)
			}
			// Let goroutines block on channel
			for j := 0; j < 5; j++ {
				runtime.Gosched()
			}
		}
	})

	SetGoroutineLabels(WithLabels(context.Background(), Labels("self-label", "self-value")))
	defer SetGoroutineLabels(context.Background())

	garbage := new(*int)
	fingReady := make(chan struct{})
	runtime.SetFinalizer(garbage, func(v **int) {
		Do(context.Background(), Labels("fing-label", "fing-value"), func(ctx context.Context) {
			close(fingReady)
			<-c
		})
	})
	garbage = nil
	for i := 0; i < 2; i++ {
		runtime.GC()
	}
	<-fingReady

	var w bytes.Buffer
	goroutineProf := Lookup("goroutine")

	// Check debug profile
	goroutineProf.WriteTo(&w, 1)
	prof := w.String()

	labels := labelMap{label.NewSet(Labels("label", "value").list)}
	labelStr := "\n# labels: " + labels.String()
	selfLabel := labelMap{label.NewSet(Labels("self-label", "self-value").list)}
	selfLabelStr := "\n# labels: " + selfLabel.String()
	fingLabel := labelMap{label.NewSet(Labels("fing-label", "fing-value").list)}
	fingLabelStr := "\n# labels: " + fingLabel.String()
	orderedPrefix := []string{
		"\n50 @ ",
		"\n44 @", labelStr,
		"\n40 @",
		"\n36 @", labelStr,
		"\n10 @",
		"\n9 @", labelStr,
		"\n1 @"}
	if !containsInOrder(prof, append(orderedPrefix, selfLabelStr)...) {
		t.Errorf("expected sorted goroutine counts with Labels:\n%s", prof)
	}
	if !containsInOrder(prof, append(orderedPrefix, fingLabelStr)...) {
		t.Errorf("expected sorted goroutine counts with Labels:\n%s", prof)
	}

	// Check proto profile
	w.Reset()
	goroutineProf.WriteTo(&w, 0)
	p, err := profile.Parse(&w)
	if err != nil {
		t.Errorf("error parsing protobuf profile: %v", err)
	}
	if err := p.CheckValid(); err != nil {
		t.Errorf("protobuf profile is invalid: %v", err)
	}
	expectedLabels := map[int64]map[string]string{
		50: {},
		44: {"label": "value"},
		40: {},
		36: {"label": "value"},
		10: {},
		9:  {"label": "value"},
		1:  {"self-label": "self-value", "fing-label": "fing-value"},
	}
	if !containsCountsLabels(p, expectedLabels) {
		t.Errorf("expected count profile to contain goroutines with counts and labels %v, got %v",
			expectedLabels, p)
	}

	close(c)

	time.Sleep(10 * time.Millisecond) // let goroutines exit
}

func containsInOrder(s string, all ...string) bool {
	for _, t := range all {
		var ok bool
		if _, s, ok = strings.Cut(s, t); !ok {
			return false
		}
	}
	return true
}

func containsCountsLabels(prof *profile.Profile, countLabels map[int64]map[string]string) bool {
	m := make(map[int64]int)
	type nkey struct {
		count    int64
		key, val string
	}
	n := make(map[nkey]int)
	for c, kv := range countLabels {
		m[c]++
		for k, v := range kv {
			n[nkey{
				count: c,
				key:   k,
				val:   v,
			}]++

		}
	}
	for _, s := range prof.Sample {
		// The count is the single value in the sample
		if len(s.Value) != 1 {
			return false
		}
		m[s.Value[0]]--
		for k, vs := range s.Label {
			for _, v := range vs {
				n[nkey{
					count: s.Value[0],
					key:   k,
					val:   v,
				}]--
			}
		}
	}
	for _, n := range m {
		if n > 0 {
			return false
		}
	}
	for _, ncnt := range n {
		if ncnt != 0 {
			return false
		}
	}
	return true
}

// Inlining disabled to make identification simpler.
//
//go:noinline
func goroutineLeakExample() {
	<-make(chan struct{})
	panic("unreachable")
}

func TestGoroutineLeakProfileConcurrency(t *testing.T) {
	const leakCount = 3

	testenv.MustHaveParallelism(t)
	regexLeakCount := regexp.MustCompile("goroutineleak profile: total ")
	whiteSpace := regexp.MustCompile("\\s+")

	// Regular goroutine profile. Used to check that there is no interference between
	// the two profile types.
	goroutineProf := Lookup("goroutine")
	goroutineLeakProf := goroutineLeakProfile

	// We use this helper to count the total number of leaked goroutines in a text profile.
	countLeaks := func(t *testing.T, profText string) int64 {
		t.Helper()

		// Strip the profile header
		parts := regexLeakCount.Split(profText, -1)
		if len(parts) < 2 {
			t.Fatalf("goroutineleak profile does not contain 'goroutineleak profile: total ': %s\nparts: %v", profText, parts)
		}

		parts = whiteSpace.Split(parts[1], -1)

		count, err := strconv.ParseInt(parts[0], 10, 64)
		if err != nil {
			t.Fatalf("goroutineleak profile count is not a number: %s\nerror: %v", profText, err)
		}
		return count
	}

	// checkFrame looks for a specific frame in the stack.
	//
	// i is the location index in the profile and j is the location line index for the location.
	// (Inlining may cause aliasing to the same location.)
	checkFrame := func(t *testing.T, i int, j int, locations []*profile.Location, funcName string) {
		if len(locations) <= i {
			t.Errorf("leaked goroutine stack locations: out of range index %d, length %d", i, len(locations))
			return
		}
		location := locations[i]
		if len(location.Line) <= j {
			t.Errorf("leaked goroutine stack location lines: out of range index %d, length %d", j, len(location.Line))
			return
		}
		if location.Line[j].Function.Name != funcName {
			t.Errorf("leaked goroutine stack expected %s as location[%d].Line[%d] but found %s (%s:%d)", funcName, i, j, location.Line[j].Function.Name, location.Line[j].Function.Filename, location.Line[j].Line)
		}
	}

	// checkLeakStack hooks into profile parsing and performs validation, looking for specific stacks for
	// the goroutines we'll leak in this test.
	checkLeakStack := func(t *testing.T) func(pc uintptr, locations []*profile.Location, _ map[string][]string) {
		return func(pc uintptr, locations []*profile.Location, _ map[string][]string) {
			if pc != leakCount {
				t.Errorf("expected %d leaked goroutines with specific stack configurations, but found %d", leakCount, pc)
				return
			}
			if len(locations) < 4 || len(locations) > 5 {
				message := fmt.Sprintf("leaked goroutine stack expected 4 or 5 locations but found %d", len(locations))
				for _, location := range locations {
					for _, line := range location.Line {
						message += fmt.Sprintf("\n%s:%d", line.Function.Name, line.Line)
					}
				}
				t.Errorf("%s", message)
				return
			}
			// We expect a receive operation. This is the typical stack.
			checkFrame(t, 0, 0, locations, "runtime.gopark")
			checkFrame(t, 1, 0, locations, "runtime.chanrecv")
			checkFrame(t, 2, 0, locations, "runtime.chanrecv1")
			checkFrame(t, 3, 0, locations, "runtime/pprof.goroutineLeakExample")
			if len(locations) == 5 {
				checkFrame(t, 4, 0, locations, "runtime/pprof.TestGoroutineLeakProfileConcurrency.func4")
			}
		}
	}

	// Leak some goroutines that will feature in the goroutine leak profile
	const totalLeaked = leakCount * 2
	for i := 0; i < leakCount; i++ {
		go goroutineLeakExample()
		go func() {
			// Leak another goroutine that will feature a slightly different stack.
			// This includes the frame runtime/pprof.TestGoroutineLeakProfileConcurrency.func1.
			goroutineLeakExample()
			panic("unreachable")
		}()
	}

	// Wait for the goroutines to leak. We might wait here until the timeout,
	// but this is better than intermittent flakes because we didn't wait long
	// enough. If we actually time out, then there's likely a bug.
	attempts := 0
	startTime := time.Now()
	waitFor := 10 * time.Millisecond
	for {
		//
		// If they never get detected, we'll get a timeout.
		time.Sleep(waitFor)

		var w strings.Builder
		goroutineLeakProf.WriteTo(&w, 1)
		n := countLeaks(t, w.String())
		if n >= totalLeaked {
			break
		}

		// Log some messages so if a timeout is seen
		attempts++
		t.Logf("waiting for leak: attempt %d (t=%s): found %d leaked goroutines", attempts, time.Since(startTime), n)

		// Wait a little longer to avoid spamming the log.
		waitFor *= 2
		if waitFor > time.Second {
			waitFor = time.Second
		}
	}

	t.Run("profile contains leak", func(t *testing.T) {
		var w strings.Builder
		goroutineLeakProf.WriteTo(&w, 0)
		parseProfile(t, []byte(w.String()), checkLeakStack(t))
	})

	t.Run("leak persists between sequential profiling runs", func(t *testing.T) {
		for i := 0; i < 2; i++ {
			var w strings.Builder
			goroutineLeakProf.WriteTo(&w, 0)
			parseProfile(t, []byte(w.String()), checkLeakStack(t))
		}
	})

	// Concurrent calls to the goroutine leak profiler should not trigger data races
	// or corruption.
	quickCheckForGoroutine := func(t *testing.T, profType, leak, profText string) {
		if !strings.Contains(profText, leak) {
			t.Errorf("%s profile does not contain expected leaked goroutine %s: %s", profType, leak, profText)
		}
	}
	t.Run("overlapping profile requests", func(t *testing.T) {
		ctx := context.Background()
		ctx, cancel := context.WithTimeout(ctx, time.Second)
		defer cancel()

		var wg sync.WaitGroup
		for i := 0; i < 2; i++ {
			wg.Add(1)
			Do(ctx, Labels("i", fmt.Sprint(i)), func(context.Context) {
				go func() {
					defer wg.Done()
					for ctx.Err() == nil {
						var w strings.Builder
						goroutineLeakProf.WriteTo(&w, 1)
						if n := countLeaks(t, w.String()); n != totalLeaked {
							t.Errorf("expected %d goroutines leaked, got %d: %s", totalLeaked, n, w.String())
						}
						quickCheckForGoroutine(t, "goroutineleak", "runtime/pprof.goroutineLeakExample", w.String())
					}
				}()
			})
		}
		wg.Wait()
	})

	// Concurrent calls to the goroutine leak profiler should not trigger data races
	// or corruption, or interfere with regular goroutine profiles.
	t.Run("overlapping goroutine and goroutine leak profile requests", func(t *testing.T) {
		ctx := context.Background()
		ctx, cancel := context.WithTimeout(ctx, time.Second)
		defer cancel()

		var wg sync.WaitGroup
		for i := 0; i < 2; i++ {
			wg.Add(2)
			Do(ctx, Labels("i", fmt.Sprint(i)), func(context.Context) {
				go func() {
					defer wg.Done()
					for ctx.Err() == nil {
						var w strings.Builder
						goroutineLeakProf.WriteTo(&w, 1)
						if n := countLeaks(t, w.String()); n != totalLeaked {
							t.Errorf("expected %d goroutines leaked, got %d: %s", totalLeaked, n, w.String())
						}
						quickCheckForGoroutine(t, "goroutineleak", "runtime/pprof.goroutineLeakExample", w.String())
					}
				}()
				go func() {
					defer wg.Done()
					for ctx.Err() == nil {
						var w strings.Builder
						goroutineProf.WriteTo(&w, 1)
						// The regular goroutine profile should see the leaked
						// goroutines. We simply check that the goroutine leak
						// profile does not corrupt the goroutine profile state.
						quickCheckForGoroutine(t, "goroutine", "runtime/pprof.goroutineLeakExample", w.String())
					}
				}()
			})
		}
		wg.Wait()
	})
}

func TestGoroutineProfileConcurrency(t *testing.T) {
	testenv.MustHaveParallelism(t)

	goroutineProf := Lookup("goroutine")

	profilerCalls := func(s string) int {
		return strings.Count(s, "\truntime/pprof.runtime_goroutineProfileWithLabels+")
	}

	includesFinalizerOrCleanup := func(s string) bool {
		return strings.Contains(s, "runtime.runFinalizers") || strings.Contains(s, "runtime.runCleanups")
	}

	// Concurrent calls to the goroutine profiler should not trigger data races
	// or corruption.
	t.Run("overlapping profile requests", func(t *testing.T) {
		ctx := context.Background()
		ctx, cancel := context.WithTimeout(ctx, 10*time.Second)
		defer cancel()

		var wg sync.WaitGroup
		for i := 0; i < 2; i++ {
			wg.Add(1)
			Do(ctx, Labels("i", fmt.Sprint(i)), func(context.Context) {
				go func() {
					defer wg.Done()
					for ctx.Err() == nil {
						var w strings.Builder
						goroutineProf.WriteTo(&w, 1)
						prof := w.String()
						count := profilerCalls(prof)
						if count >= 2 {
							t.Logf("prof %d\n%s", count, prof)
							cancel()
						}
					}
				}()
			})
		}
		wg.Wait()
	})

	// The finalizer goroutine should not show up in most profiles, since it's
	// marked as a system goroutine when idle.
	t.Run("finalizer not present", func(t *testing.T) {
		var w strings.Builder
		goroutineProf.WriteTo(&w, 1)
		prof := w.String()
		if includesFinalizerOrCleanup(prof) {
			t.Errorf("profile includes finalizer or cleanup (but should be marked as system):\n%s", prof)
		}
	})

	// The finalizer goroutine should show up when it's running user code.
	t.Run("finalizer present", func(t *testing.T) {
		// T is a pointer type so it won't be allocated by the tiny
		// allocator, which can lead to its finalizer not being called
		// during this test
		type T *byte
		obj := new(T)
		ch1, ch2 := make(chan int), make(chan int)
		defer close(ch2)
		runtime.SetFinalizer(obj, func(_ any) {
			close(ch1)
			<-ch2
		})
		obj = nil
		for i := 10; i >= 0; i-- {
			select {
			case <-ch1:
			default:
				if i == 0 {
					t.Fatalf("finalizer did not run")
				}
				runtime.GC()
			}
		}
		var w strings.Builder
		goroutineProf.WriteTo(&w, 1)
		prof := w.String()
		if !includesFinalizerOrCleanup(prof) {
			t.Errorf("profile does not include finalizer (and it should be marked as user):\n%s", prof)
		}
	})

	// Check that new goroutines only show up in order.
	testLaunches := func(t *testing.T) {
		var done sync.WaitGroup
		defer done.Wait()

		ctx := context.Background()
		ctx, cancel := context.WithCancel(ctx)
		defer cancel()

		ch := make(chan int)
		defer close(ch)

		var ready sync.WaitGroup

		// These goroutines all survive until the end of the subtest, so we can
		// check that a (numbered) goroutine appearing in the profile implies
		// that all older goroutines also appear in the profile.
		ready.Add(1)
		done.Add(1)
		go func() {
			defer done.Done()
			for i := 0; ctx.Err() == nil; i++ {
				// Use SetGoroutineLabels rather than Do we can always expect an
				// extra goroutine (this one) with most recent label.
				SetGoroutineLabels(WithLabels(ctx, Labels(t.Name()+"-loop-i", fmt.Sprint(i))))
				done.Add(1)
				go func() {
					<-ch
					done.Done()
				}()
				for j := 0; j < i; j++ {
					// Spin for longer and longer as the test goes on. This
					// goroutine will do O(N^2) work with the number of
					// goroutines it launches. This should be slow relative to
					// the work involved in collecting a goroutine profile,
					// which is O(N) with the high-water mark of the number of
					// goroutines in this process (in the allgs slice).
					runtime.Gosched()
				}
				if i == 0 {
					ready.Done()
				}
			}
		}()

		// Short-lived goroutines exercise different code paths (goroutines with
		// status _Gdead, for instance). This churn doesn't have behavior that
		// we can test directly, but does help to shake out data races.
		ready.Add(1)
		var churn func(i int)
		churn = func(i int) {
			SetGoroutineLabels(WithLabels(ctx, Labels(t.Name()+"-churn-i", fmt.Sprint(i))))
			if i == 0 {
				ready.Done()
			} else if i%16 == 0 {
				// Yield on occasion so this sequence of goroutine launches
				// doesn't monopolize a P. See issue #52934.
				runtime.Gosched()
			}
			if ctx.Err() == nil {
				go churn(i + 1)
			}
		}
		go func() {
			churn(0)
		}()

		ready.Wait()

		var w [3]bytes.Buffer
		for i := range w {
			goroutineProf.WriteTo(&w[i], 0)
		}
		for i := range w {
			p, err := profile.Parse(bytes.NewReader(w[i].Bytes()))
			if err != nil {
				t.Errorf("error parsing protobuf profile: %v", err)
			}

			// High-numbered loop-i goroutines imply that every lower-numbered
			// loop-i goroutine should be present in the profile too.
			counts := make(map[string]int)
			for _, s := range p.Sample {
				label := s.Label[t.Name()+"-loop-i"]
				if len(label) > 0 {
					counts[label[0]]++
				}
			}
			for j, max := 0, len(counts)-1; j <= max; j++ {
				n := counts[fmt.Sprint(j)]
				if n == 1 || (n == 2 && j == max) {
					continue
				}
				t.Errorf("profile #%d's goroutines with label loop-i:%d; %d != 1 (or 2 for the last entry, %d)",
					i+1, j, n, max)
				t.Logf("counts %v", counts)
				break
			}
		}
	}

	runs := 100
	if testing.Short() {
		runs = 5
	}
	for i := 0; i < runs; i++ {
		// Run multiple times to shake out data races
		t.Run("goroutine launches", testLaunches)
	}
}

// Regression test for #69998.
func TestGoroutineProfileCoro(t *testing.T) {
	testenv.MustHaveParallelism(t)

	goroutineProf := Lookup("goroutine")

	// Set up a goroutine to just create and run coroutine goroutines all day.
	iterFunc := func() {
		p, stop := iter.Pull2(
			func(yield func(int, int) bool) {
				for i := 0; i < 10000; i++ {
					if !yield(i, i) {
						return
					}
				}
			},
		)
		defer stop()
		for {
			_, _, ok := p()
			if !ok {
				break
			}
		}
	}
	var wg sync.WaitGroup
	done := make(chan struct{})
	wg.Add(1)
	go func() {
		defer wg.Done()
		for {
			iterFunc()
			select {
			case <-done:
			default:
			}
		}
	}()

	// Take a goroutine profile. If the bug in #69998 is present, this will crash
	// with high probability. We don't care about the output for this bug.
	goroutineProf.WriteTo(io.Discard, 1)
}

// This test tries to provoke a situation wherein the finalizer goroutine is
// erroneously inspected by the goroutine profiler in such a way that could
// cause a crash. See go.dev/issue/74090.
func TestGoroutineProfileIssue74090(t *testing.T) {
	testenv.MustHaveParallelism(t)

	goroutineProf := Lookup("goroutine")

	// T is a pointer type so it won't be allocated by the tiny
	// allocator, which can lead to its finalizer not being called
	// during this test.
	type T *byte
	for range 10 {
		// We use finalizers for this test because finalizers transition between
		// system and user goroutine on each call, since there's substantially
		// more work to do to set up a finalizer call. Cleanups, on the other hand,
		// transition once for a whole batch, and so are less likely to trigger
		// the failure. Under stress testing conditions this test fails approximately
		// 5 times every 1000 executions on a 64 core machine without the appropriate
		// fix, which is not ideal but if this test crashes at all, it's a clear
		// signal that something is broken.
		var objs []*T
		for range 10000 {
			obj := new(T)
			runtime.SetFinalizer(obj, func(_ any) {})
			objs = append(objs, obj)
		}
		objs = nil

		// Queue up all the finalizers.
		runtime.GC()

		// Try to run a goroutine profile concurrently with finalizer execution
		// to trigger the bug.
		var w strings.Builder
		goroutineProf.WriteTo(&w, 1)
	}
}

func BenchmarkGoroutine(b *testing.B) {
	withIdle := func(n int, fn func(b *testing.B)) func(b *testing.B) {
		return func(b *testing.B) {
			c := make(chan int)
			var ready, done sync.WaitGroup
			defer func() {
				close(c)
				done.Wait()
			}()

			for i := 0; i < n; i++ {
				ready.Add(1)
				done.Add(1)
				go func() {
					ready.Done()
					<-c
					done.Done()
				}()
			}
			// Let goroutines block on channel
			ready.Wait()
			for i := 0; i < 5; i++ {
				runtime.Gosched()
			}

			fn(b)
		}
	}

	withChurn := func(fn func(b *testing.B)) func(b *testing.B) {
		return func(b *testing.B) {
			ctx := context.Background()
			ctx, cancel := context.WithCancel(ctx)
			defer cancel()

			var ready sync.WaitGroup
			ready.Add(1)
			var count int64
			var churn func(i int)
			churn = func(i int) {
				SetGoroutineLabels(WithLabels(ctx, Labels("churn-i", fmt.Sprint(i))))
				atomic.AddInt64(&count, 1)
				if i == 0 {
					ready.Done()
				}
				if ctx.Err() == nil {
					go churn(i + 1)
				}
			}
			go func() {
				churn(0)
			}()
			ready.Wait()

			fn(b)
			b.ReportMetric(float64(atomic.LoadInt64(&count))/float64(b.N), "concurrent_launches/op")
		}
	}

	benchWriteTo := func(b *testing.B) {
		goroutineProf := Lookup("goroutine")
		b.ResetTimer()
		for i := 0; i < b.N; i++ {
			goroutineProf.WriteTo(io.Discard, 0)
		}
		b.StopTimer()
	}

	benchGoroutineProfile := func(b *testing.B) {
		p := make([]runtime.StackRecord, 10000)
		b.ResetTimer()
		for i := 0; i < b.N; i++ {
			runtime.GoroutineProfile(p)
		}
		b.StopTimer()
	}

	// Note that some costs of collecting a goroutine profile depend on the
	// length of the runtime.allgs slice, which never shrinks. Stay within race
	// detector's 8k-goroutine limit
	for _, n := range []int{50, 500, 5000} {
		b.Run(fmt.Sprintf("Profile.WriteTo idle %d", n), withIdle(n, benchWriteTo))
		b.Run(fmt.Sprintf("Profile.WriteTo churn %d", n), withIdle(n, withChurn(benchWriteTo)))
		b.Run(fmt.Sprintf("runtime.GoroutineProfile churn %d", n), withIdle(n, withChurn(benchGoroutineProfile)))
	}
}

var emptyCallStackTestRun int64

// Issue 18836.
func TestEmptyCallStack(t *testing.T) {
	name := fmt.Sprintf("test18836_%d", emptyCallStackTestRun)
	emptyCallStackTestRun++

	t.Parallel()
	var buf strings.Builder
	p := NewProfile(name)

	p.Add("foo", 47674)
	p.WriteTo(&buf, 1)
	p.Remove("foo")
	got := buf.String()
	prefix := name + " profile: total 1\n"
	if !strings.HasPrefix(got, prefix) {
		t.Fatalf("got:\n\t%q\nwant prefix:\n\t%q\n", got, prefix)
	}
	lostevent := "lostProfileEvent"
	if !strings.Contains(got, lostevent) {
		t.Fatalf("got:\n\t%q\ndoes not contain:\n\t%q\n", got, lostevent)
	}
}

// stackContainsLabeled takes a spec like funcname;key=value and matches if the stack has that key
// and value and has funcname somewhere in the stack.
func stackContainsLabeled(spec string, count uintptr, stk []*profile.Location, labels map[string][]string) bool {
	base, kv, ok := strings.Cut(spec, ";")
	if !ok {
		panic("no semicolon in key/value spec")
	}
	k, v, ok := strings.Cut(kv, "=")
	if !ok {
		panic("missing = in key/value spec")
	}
	if !slices.Contains(labels[k], v) {
		return false
	}
	return stackContains(base, count, stk, labels)
}

func TestCPUProfileLabel(t *testing.T) {
	matches := matchAndAvoidStacks(stackContainsLabeled, []string{"runtime/pprof.cpuHogger;key=value"}, avoidFunctions())
	testCPUProfile(t, matches, func(dur time.Duration) {
		Do(context.Background(), Labels("key", "value"), func(context.Context) {
			cpuHogger(cpuHog1, &salt1, dur)
		})
	})
}

func TestLabelRace(t *testing.T) {
	testenv.MustHaveParallelism(t)
	// Test the race detector annotations for synchronization
	// between setting labels and consuming them from the
	// profile.
	matches := matchAndAvoidStacks(stackContainsLabeled, []string{"runtime/pprof.cpuHogger;key=value"}, nil)
	testCPUProfile(t, matches, func(dur time.Duration) {
		start := time.Now()
		var wg sync.WaitGroup
		for time.Since(start) < dur {
			var salts [10]int
			for i := 0; i < 10; i++ {
				wg.Add(1)
				go func(j int) {
					Do(context.Background(), Labels("key", "value"), func(context.Context) {
						cpuHogger(cpuHog1, &salts[j], time.Millisecond)
					})
					wg.Done()
				}(i)
			}
			wg.Wait()
		}
	})
}

func TestGoroutineProfileLabelRace(t *testing.T) {
	testenv.MustHaveParallelism(t)
	// Test the race detector annotations for synchronization
	// between setting labels and consuming them from the
	// goroutine profile. See issue #50292.

	t.Run("reset", func(t *testing.T) {
		ctx := context.Background()
		ctx, cancel := context.WithCancel(ctx)
		defer cancel()

		go func() {
			goroutineProf := Lookup("goroutine")
			for ctx.Err() == nil {
				var w strings.Builder
				goroutineProf.WriteTo(&w, 1)
				prof := w.String()
				if strings.Contains(prof, "loop-i") {
					cancel()
				}
			}
		}()

		for i := 0; ctx.Err() == nil; i++ {
			Do(ctx, Labels("loop-i", fmt.Sprint(i)), func(ctx context.Context) {
			})
		}
	})

	t.Run("churn", func(t *testing.T) {
		ctx := context.Background()
		ctx, cancel := context.WithCancel(ctx)
		defer cancel()

		var ready sync.WaitGroup
		ready.Add(1)
		var churn func(i int)
		churn = func(i int) {
			SetGoroutineLabels(WithLabels(ctx, Labels("churn-i", fmt.Sprint(i))))
			if i == 0 {
				ready.Done()
			}
			if ctx.Err() == nil {
				go churn(i + 1)
			}
		}
		go func() {
			churn(0)
		}()
		ready.Wait()

		goroutineProf := Lookup("goroutine")
		for i := 0; i < 10; i++ {
			goroutineProf.WriteTo(io.Discard, 1)
		}
	})
}

// TestLabelSystemstack makes sure CPU profiler samples of goroutines running
// on systemstack include the correct pprof labels. See issue #48577
func TestLabelSystemstack(t *testing.T) {
	// Grab and re-set the initial value before continuing to ensure
	// GOGC doesn't actually change following the test.
	gogc := debug.SetGCPercent(100)
	debug.SetGCPercent(gogc)

	matches := matchAndAvoidStacks(stackContainsLabeled, []string{"runtime.systemstack;key=value"}, avoidFunctions())
	p := testCPUProfile(t, matches, func(dur time.Duration) {
		Do(context.Background(), Labels("key", "value"), func(ctx context.Context) {
			parallelLabelHog(ctx, dur, gogc)
		})
	})

	// Two conditions to check:
	// * labelHog should always be labeled.
	// * The label should _only_ appear on labelHog and the Do call above.
	for _, s := range p.Sample {
		isLabeled := s.Label != nil && slices.Contains(s.Label["key"], "value")
		var (
			mayBeLabeled     bool
			mustBeLabeled    string
			mustNotBeLabeled string
		)
		for _, loc := range s.Location {
			for _, l := range loc.Line {
				switch l.Function.Name {
				case "runtime/pprof.labelHog", "runtime/pprof.parallelLabelHog", "runtime/pprof.parallelLabelHog.func1":
					mustBeLabeled = l.Function.Name
				case "runtime/pprof.Do":
					// Do sets the labels, so samples may
					// or may not be labeled depending on
					// which part of the function they are
					// at.
					mayBeLabeled = true
				case "runtime.bgsweep", "runtime.bgscavenge", "runtime.forcegchelper", "runtime.gcBgMarkWorker", "runtime.runFinalizers", "runtime.runCleanups", "runtime.sysmon":
					// Runtime system goroutines or threads
					// (such as those identified by
					// runtime.isSystemGoroutine). These
					// should never be labeled.
					mustNotBeLabeled = l.Function.Name
				case "gogo", "gosave_systemstack_switch", "racecall":
					// These are context switch/race
					// critical that we can't do a full
					// traceback from. Typically this would
					// be covered by the runtime check
					// below, but these symbols don't have
					// the package name.
					mayBeLabeled = true
				}

				if strings.HasPrefix(l.Function.Name, "runtime.") {
					// There are many places in the runtime
					// where we can't do a full traceback.
					// Ideally we'd list them all, but
					// barring that allow anything in the
					// runtime, unless explicitly excluded
					// above.
					mayBeLabeled = true
				}
			}
		}
		errorStack := func(f string, args ...any) {
			var buf strings.Builder
			fprintStack(&buf, s.Location)
			t.Errorf("%s: %s", fmt.Sprintf(f, args...), buf.String())
		}
		if mustBeLabeled != "" && mustNotBeLabeled != "" {
			errorStack("sample contains both %s, which must be labeled, and %s, which must not be labeled", mustBeLabeled, mustNotBeLabeled)
			continue
		}
		if mustBeLabeled != "" || mustNotBeLabeled != "" {
			// We found a definitive frame, so mayBeLabeled hints are not relevant.
			mayBeLabeled = false
		}
		if mayBeLabeled {
			// This sample may or may not be labeled, so there's nothing we can check.
			continue
		}
		if mustBeLabeled != "" && !isLabeled {
			errorStack("sample must be labeled because of %s, but is not", mustBeLabeled)
		}
		if mustNotBeLabeled != "" && isLabeled {
			errorStack("sample must not be labeled because of %s, but is", mustNotBeLabeled)
		}
	}
}

// labelHog is designed to burn CPU time in a way that a high number of CPU
// samples end up running on systemstack.
func labelHog(stop chan struct{}, gogc int) {
	// Regression test for issue 50032. We must give GC an opportunity to
	// be initially triggered by a labelled goroutine.
	runtime.GC()

	for i := 0; ; i++ {
		select {
		case <-stop:
			return
		default:
			debug.SetGCPercent(gogc)
		}
	}
}

// parallelLabelHog runs GOMAXPROCS goroutines running labelHog.
func parallelLabelHog(ctx context.Context, dur time.Duration, gogc int) {
	var wg sync.WaitGroup
	stop := make(chan struct{})
	for i := 0; i < runtime.GOMAXPROCS(0); i++ {
		wg.Add(1)
		go func() {
			defer wg.Done()
			labelHog(stop, gogc)
		}()
	}

	time.Sleep(dur)
	close(stop)
	wg.Wait()
}

// Check that there is no deadlock when the program receives SIGPROF while in
// 64bit atomics' critical section. Used to happen on mips{,le}. See #20146.
func TestAtomicLoadStore64(t *testing.T) {
	f, err := os.CreateTemp("", "profatomic")
	if err != nil {
		t.Fatalf("TempFile: %v", err)
	}
	defer os.Remove(f.Name())
	defer f.Close()

	if err := StartCPUProfile(f); err != nil {
		t.Fatal(err)
	}
	defer StopCPUProfile()

	var flag uint64
	done := make(chan bool, 1)

	go func() {
		for atomic.LoadUint64(&flag) == 0 {
			runtime.Gosched()
		}
		done <- true
	}()
	time.Sleep(50 * time.Millisecond)
	atomic.StoreUint64(&flag, 1)
	<-done
}

func TestTracebackAll(t *testing.T) {
	// With gccgo, if a profiling signal arrives at the wrong time
	// during traceback, it may crash or hang. See issue #29448.
	f, err := os.CreateTemp("", "proftraceback")
	if err != nil {
		t.Fatalf("TempFile: %v", err)
	}
	defer os.Remove(f.Name())
	defer f.Close()

	if err := StartCPUProfile(f); err != nil {
		t.Fatal(err)
	}
	defer StopCPUProfile()

	ch := make(chan int)
	defer close(ch)

	count := 10
	for i := 0; i < count; i++ {
		go func() {
			<-ch // block
		}()
	}

	N := 10000
	if testing.Short() {
		N = 500
	}
	buf := make([]byte, 10*1024)
	for i := 0; i < N; i++ {
		runtime.Stack(buf, true)
	}
}

// TestTryAdd tests the cases that are hard to test with real program execution.
//
// For example, the current go compilers may not always inline functions
// involved in recursion but that may not be true in the future compilers. This
// tests such cases by using fake call sequences and forcing the profile build
// utilizing translateCPUProfile defined in proto_test.go
func TestTryAdd(t *testing.T) {
	if _, found := findInlinedCall(inlinedCallerDump, 4<<10); !found {
		t.Skip("Can't determine whether anything was inlined into inlinedCallerDump.")
	}

	// inlinedCallerDump
	//   inlinedCalleeDump
	pcs := make([]uintptr, 2)
	inlinedCallerDump(pcs)
	inlinedCallerStack := make([]uint64, 2)
	for i := range pcs {
		inlinedCallerStack[i] = uint64(pcs[i])
	}
	wrapperPCs := make([]uintptr, 1)
	inlinedWrapperCallerDump(wrapperPCs)

	if _, found := findInlinedCall(recursionChainBottom, 4<<10); !found {
		t.Skip("Can't determine whether anything was inlined into recursionChainBottom.")
	}

	// recursionChainTop
	//   recursionChainMiddle
	//     recursionChainBottom
	//       recursionChainTop
	//         recursionChainMiddle
	//           recursionChainBottom
	pcs = make([]uintptr, 6)
	recursionChainTop(1, pcs)
	recursionStack := make([]uint64, len(pcs))
	for i := range pcs {
		recursionStack[i] = uint64(pcs[i])
	}

	period := int64(2000 * 1000) // 1/500*1e9 nanosec.

	testCases := []struct {
		name        string
		input       []uint64          // following the input format assumed by profileBuilder.addCPUData.
		count       int               // number of records in input.
		wantLocs    [][]string        // ordered location entries with function names.
		wantSamples []*profile.Sample // ordered samples, we care only about Value and the profile location IDs.
	}{{
		// Sanity test for a normal, complete stack trace.
		name: "full_stack_trace",
		input: []uint64{
			3, 0, 500, // hz = 500. Must match the period.
			5, 0, 50, inlinedCallerStack[0], inlinedCallerStack[1],
		},
		count: 2,
		wantLocs: [][]string{
			{"runtime/pprof.inlinedCalleeDump", "runtime/pprof.inlinedCallerDump"},
		},
		wantSamples: []*profile.Sample{
			{Value: []int64{50, 50 * period}, Location: []*profile.Location{{ID: 1}}},
		},
	}, {
		name: "bug35538",
		input: []uint64{
			3, 0, 500, // hz = 500. Must match the period.
			// Fake frame: tryAdd will have inlinedCallerDump
			// (stack[1]) on the deck when it encounters the next
			// inline function. It should accept this.
			7, 0, 10, inlinedCallerStack[0], inlinedCallerStack[1], inlinedCallerStack[0], inlinedCallerStack[1],
			5, 0, 20, inlinedCallerStack[0], inlinedCallerStack[1],
		},
		count:    3,
		wantLocs: [][]string{{"runtime/pprof.inlinedCalleeDump", "runtime/pprof.inlinedCallerDump"}},
		wantSamples: []*profile.Sample{
			{Value: []int64{10, 10 * period}, Location: []*profile.Location{{ID: 1}, {ID: 1}}},
			{Value: []int64{20, 20 * period}, Location: []*profile.Location{{ID: 1}}},
		},
	}, {
		name: "bug38096",
		input: []uint64{
			3, 0, 500, // hz = 500. Must match the period.
			// count (data[2]) == 0 && len(stk) == 1 is an overflow
			// entry. The "stk" entry is actually the count.
			4, 0, 0, 4242,
		},
		count:    2,
		wantLocs: [][]string{{"runtime/pprof.lostProfileEvent"}},
		wantSamples: []*profile.Sample{
			{Value: []int64{4242, 4242 * period}, Location: []*profile.Location{{ID: 1}}},
		},
	}, {
		// If a function is directly called recursively then it must
		// not be inlined in the caller.
		//
		// N.B. We're generating an impossible profile here, with a
		// recursive inlineCalleeDump call. This is simulating a non-Go
		// function that looks like an inlined Go function other than
		// its recursive property. See pcDeck.tryAdd.
		name: "directly_recursive_func_is_not_inlined",
		input: []uint64{
			3, 0, 500, // hz = 500. Must match the period.
			5, 0, 30, inlinedCallerStack[0], inlinedCallerStack[0],
			4, 0, 40, inlinedCallerStack[0],
		},
		count: 3,
		// inlinedCallerDump shows up here because
		// runtime_expandFinalInlineFrame adds it to the stack frame.
		wantLocs: [][]string{{"runtime/pprof.inlinedCalleeDump"}, {"runtime/pprof.inlinedCallerDump"}},
		wantSamples: []*profile.Sample{
			{Value: []int64{30, 30 * period}, Location: []*profile.Location{{ID: 1}, {ID: 1}, {ID: 2}}},
			{Value: []int64{40, 40 * period}, Location: []*profile.Location{{ID: 1}, {ID: 2}}},
		},
	}, {
		name: "recursion_chain_inline",
		input: []uint64{
			3, 0, 500, // hz = 500. Must match the period.
			9, 0, 10, recursionStack[0], recursionStack[1], recursionStack[2], recursionStack[3], recursionStack[4], recursionStack[5],
		},
		count: 2,
		wantLocs: [][]string{
			{"runtime/pprof.recursionChainBottom"},
			{
				"runtime/pprof.recursionChainMiddle",
				"runtime/pprof.recursionChainTop",
				"runtime/pprof.recursionChainBottom",
			},
			{
				"runtime/pprof.recursionChainMiddle",
				"runtime/pprof.recursionChainTop",
				"runtime/pprof.TestTryAdd", // inlined into the test.
			},
		},
		wantSamples: []*profile.Sample{
			{Value: []int64{10, 10 * period}, Location: []*profile.Location{{ID: 1}, {ID: 2}, {ID: 3}}},
		},
	}, {
		name: "truncated_stack_trace_later",
		input: []uint64{
			3, 0, 500, // hz = 500. Must match the period.
			5, 0, 50, inlinedCallerStack[0], inlinedCallerStack[1],
			4, 0, 60, inlinedCallerStack[0],
		},
		count:    3,
		wantLocs: [][]string{{"runtime/pprof.inlinedCalleeDump", "runtime/pprof.inlinedCallerDump"}},
		wantSamples: []*profile.Sample{
			{Value: []int64{50, 50 * period}, Location: []*profile.Location{{ID: 1}}},
			{Value: []int64{60, 60 * period}, Location: []*profile.Location{{ID: 1}}},
		},
	}, {
		name: "truncated_stack_trace_first",
		input: []uint64{
			3, 0, 500, // hz = 500. Must match the period.
			4, 0, 70, inlinedCallerStack[0],
			5, 0, 80, inlinedCallerStack[0], inlinedCallerStack[1],
		},
		count:    3,
		wantLocs: [][]string{{"runtime/pprof.inlinedCalleeDump", "runtime/pprof.inlinedCallerDump"}},
		wantSamples: []*profile.Sample{
			{Value: []int64{70, 70 * period}, Location: []*profile.Location{{ID: 1}}},
			{Value: []int64{80, 80 * period}, Location: []*profile.Location{{ID: 1}}},
		},
	}, {
		// We can recover the inlined caller from a truncated stack.
		name: "truncated_stack_trace_only",
		input: []uint64{
			3, 0, 500, // hz = 500. Must match the period.
			4, 0, 70, inlinedCallerStack[0],
		},
		count:    2,
		wantLocs: [][]string{{"runtime/pprof.inlinedCalleeDump", "runtime/pprof.inlinedCallerDump"}},
		wantSamples: []*profile.Sample{
			{Value: []int64{70, 70 * period}, Location: []*profile.Location{{ID: 1}}},
		},
	}, {
		// The same location is used for duplicated stacks.
		name: "truncated_stack_trace_twice",
		input: []uint64{
			3, 0, 500, // hz = 500. Must match the period.
			4, 0, 70, inlinedCallerStack[0],
			// Fake frame: add a fake call to
			// inlinedCallerDump to prevent this sample
			// from getting merged into above.
			5, 0, 80, inlinedCallerStack[1], inlinedCallerStack[0],
		},
		count: 3,
		wantLocs: [][]string{
			{"runtime/pprof.inlinedCalleeDump", "runtime/pprof.inlinedCallerDump"},
			{"runtime/pprof.inlinedCallerDump"},
		},
		wantSamples: []*profile.Sample{
			{Value: []int64{70, 70 * period}, Location: []*profile.Location{{ID: 1}}},
			{Value: []int64{80, 80 * period}, Location: []*profile.Location{{ID: 2}, {ID: 1}}},
		},
	}, {
		name: "expand_wrapper_function",
		input: []uint64{
			3, 0, 500, // hz = 500. Must match the period.
			4, 0, 50, uint64(wrapperPCs[0]),
		},
		count:    2,
		wantLocs: [][]string{{"runtime/pprof.inlineWrapper.dump"}},
		wantSamples: []*profile.Sample{
			{Value: []int64{50, 50 * period}, Location: []*profile.Location{{ID: 1}}},
		},
	}}

	for _, tc := range testCases {
		t.Run(tc.name, func(t *testing.T) {
			p, err := translateCPUProfile(tc.input, tc.count)
			if err != nil {
				t.Fatalf("translating profile: %v", err)
			}
			t.Logf("Profile: %v\n", p)

			// One location entry with all inlined functions.
			var gotLoc [][]string
			for _, loc := range p.Location {
				var names []string
				for _, line := range loc.Line {
					names = append(names, line.Function.Name)
				}
				gotLoc = append(gotLoc, names)
			}
			if got, want := fmtJSON(gotLoc), fmtJSON(tc.wantLocs); got != want {
				t.Errorf("Got Location = %+v\n\twant %+v", got, want)
			}
			// All samples should point to one location.
			var gotSamples []*profile.Sample
			for _, sample := range p.Sample {
				var locs []*profile.Location
				for _, loc := range sample.Location {
					locs = append(locs, &profile.Location{ID: loc.ID})
				}
				gotSamples = append(gotSamples, &profile.Sample{Value: sample.Value, Location: locs})
			}
			if got, want := fmtJSON(gotSamples), fmtJSON(tc.wantSamples); got != want {
				t.Errorf("Got Samples = %+v\n\twant %+v", got, want)
			}
		})
	}
}

func TestTimeVDSO(t *testing.T) {
	// Test that time functions have the right stack trace. In particular,
	// it shouldn't be recursive.

	if runtime.GOOS == "android" {
		// Flaky on Android, issue 48655. VDSO may not be enabled.
		testenv.SkipFlaky(t, 48655)
	}

	matches := matchAndAvoidStacks(stackContains, []string{"time.now"}, avoidFunctions())
	p := testCPUProfile(t, matches, func(dur time.Duration) {
		t0 := time.Now()
		for {
			t := time.Now()
			if t.Sub(t0) >= dur {
				return
			}
		}
	})

	// Check for recursive time.now sample.
	for _, sample := range p.Sample {
		var seenNow bool
		for _, loc := range sample.Location {
			for _, line := range loc.Line {
				if line.Function.Name == "time.now" {
					if seenNow {
						t.Fatalf("unexpected recursive time.now")
					}
					seenNow = true
				}
			}
		}
	}
}

func TestProfilerStackDepth(t *testing.T) {
	t.Cleanup(disableSampling())

	const depth = 128
	go produceProfileEvents(t, depth)
	awaitBlockedGoroutine(t, "chan receive", "goroutineDeep", 1)

	tests := []struct {
		profiler string
		prefix   []string
	}{
		{"heap", []string{"runtime/pprof.allocDeep"}},
		{"block", []string{"runtime.chanrecv1", "runtime/pprof.blockChanDeep"}},
		{"mutex", []string{"sync.(*Mutex).Unlock", "runtime/pprof.blockMutexDeep"}},
		{"goroutine", []string{"runtime.gopark", "runtime.chanrecv", "runtime.chanrecv1", "runtime/pprof.goroutineDeep"}},
	}

	for _, test := range tests {
		t.Run(test.profiler, func(t *testing.T) {
			var buf bytes.Buffer
			if err := Lookup(test.profiler).WriteTo(&buf, 0); err != nil {
				t.Fatalf("failed to write heap profile: %v", err)
			}
			p, err := profile.Parse(&buf)
			if err != nil {
				t.Fatalf("failed to parse heap profile: %v", err)
			}
			t.Logf("Profile = %v", p)

			stks := profileStacks(p)
			var matchedStacks [][]string
			for _, stk := range stks {
				if !hasPrefix(stk, test.prefix) {
					continue
				}
				// We may get multiple stacks which contain the prefix we want, but
				// which might not have enough frames, e.g. if the profiler hides
				// some leaf frames that would count against the stack depth limit.
				// Check for at least one match
				matchedStacks = append(matchedStacks, stk)
				if len(stk) != depth {
					continue
				}
				if rootFn, wantFn := stk[depth-1], "runtime/pprof.produceProfileEvents"; rootFn != wantFn {
					continue
				}
				// Found what we wanted
				return
			}
			for _, stk := range matchedStacks {
				t.Logf("matched stack=%s", stk)
				if len(stk) != depth {
					t.Errorf("want stack depth = %d, got %d", depth, len(stk))
					continue
				}

				if rootFn, wantFn := stk[depth-1], "runtime/pprof.allocDeep"; rootFn != wantFn {
					t.Errorf("want stack stack root %s, got %v", wantFn, rootFn)
				}
			}
		})
	}
}

func hasPrefix(stk []string, prefix []string) bool {
	return len(prefix) <= len(stk) && slices.Equal(stk[:len(prefix)], prefix)
}

// ensure that stack records are valid map keys (comparable)
var _ = map[runtime.MemProfileRecord]struct{}{}
var _ = map[runtime.StackRecord]struct{}{}

// allocDeep calls itself n times before calling fn.
func allocDeep(n int) {
	if n > 1 {
		allocDeep(n - 1)
		return
	}
	memSink = make([]byte, 1<<20)
}

// blockChanDeep produces a block profile event at stack depth n, including the
// caller.
func blockChanDeep(t *testing.T, n int) {
	if n > 1 {
		blockChanDeep(t, n-1)
		return
	}
	ch := make(chan struct{})
	go func() {
		awaitBlockedGoroutine(t, "chan receive", "blockChanDeep", 1)
		ch <- struct{}{}
	}()
	<-ch
}

// blockMutexDeep produces a block profile event at stack depth n, including the
// caller.
func blockMutexDeep(t *testing.T, n int) {
	if n > 1 {
		blockMutexDeep(t, n-1)
		return
	}
	var mu sync.Mutex
	go func() {
		mu.Lock()
		mu.Lock()
	}()
	awaitBlockedGoroutine(t, "sync.Mutex.Lock", "blockMutexDeep", 1)
	mu.Unlock()
}

// goroutineDeep blocks at stack depth n, including the caller until the test is
// finished.
func goroutineDeep(t *testing.T, n int) {
	if n > 1 {
		goroutineDeep(t, n-1)
		return
	}
	wait := make(chan struct{}, 1)
	t.Cleanup(func() {
		wait <- struct{}{}
	})
	<-wait
}

// produceProfileEvents produces pprof events at the given stack depth and then
// blocks in goroutineDeep until the test completes. The stack traces are
// guaranteed to have exactly the desired depth with produceProfileEvents as
// their root frame which is expected by TestProfilerStackDepth.
func produceProfileEvents(t *testing.T, depth int) {
	allocDeep(depth - 1)       // -1 for produceProfileEvents, **
	blockChanDeep(t, depth-2)  // -2 for produceProfileEvents, **, chanrecv1
	blockMutexDeep(t, depth-2) // -2 for produceProfileEvents, **, Unlock
	memSink = nil
	runtime.GC()
	goroutineDeep(t, depth-4) // -4 for produceProfileEvents, **, chanrecv1, chanrev, gopark
}

func getProfileStacks(collect func([]runtime.BlockProfileRecord) (int, bool), fileLine bool, pcs bool) []string {
	var n int
	var ok bool
	var p []runtime.BlockProfileRecord
	for {
		p = make([]runtime.BlockProfileRecord, n)
		n, ok = collect(p)
		if ok {
			p = p[:n]
			break
		}
	}
	var stacks []string
	for _, r := range p {
		var stack strings.Builder
		for i, pc := range r.Stack() {
			if i > 0 {
				stack.WriteByte('\n')
			}
			if pcs {
				fmt.Fprintf(&stack, "%x ", pc)
			}
			// Use FuncForPC instead of CallersFrames,
			// because we want to see the info for exactly
			// the PCs returned by the mutex profile to
			// ensure inlined calls have already been properly
			// expanded.
			f := runtime.FuncForPC(pc - 1)
			stack.WriteString(f.Name())
			if fileLine {
				stack.WriteByte(' ')
				file, line := f.FileLine(pc - 1)
				stack.WriteString(file)
				stack.WriteByte(':')
				stack.WriteString(strconv.Itoa(line))
			}
		}
		stacks = append(stacks, stack.String())
	}
	return stacks
}

func TestMutexBlockFullAggregation(t *testing.T) {
	// This regression test is adapted from
	// https://github.com/grafana/pyroscope-go/issues/103,
	// authored by Tolya Korniltsev

	var m sync.Mutex

	prev := runtime.SetMutexProfileFraction(-1)
	defer runtime.SetMutexProfileFraction(prev)

	const fraction = 1
	const iters = 100
	const workers = 2

	runtime.SetMutexProfileFraction(fraction)
	runtime.SetBlockProfileRate(1)
	defer runtime.SetBlockProfileRate(0)

	wg := sync.WaitGroup{}
	wg.Add(workers)
	for range workers {
		go func() {
			for range iters {
				m.Lock()
				// Wait at least 1 millisecond to pass the
				// starvation threshold for the mutex
				time.Sleep(time.Millisecond)
				m.Unlock()
			}
			wg.Done()
		}()
	}
	wg.Wait()

	assertNoDuplicates := func(name string, collect func([]runtime.BlockProfileRecord) (int, bool)) {
		stacks := getProfileStacks(collect, true, true)
		seen := make(map[string]struct{})
		for _, s := range stacks {
			if _, ok := seen[s]; ok {
				t.Errorf("saw duplicate entry in %s profile with stack:\n%s", name, s)
			}
			seen[s] = struct{}{}
		}
		if len(seen) == 0 {
			t.Errorf("did not see any samples in %s profile for this test", name)
		}
	}
	t.Run("mutex", func(t *testing.T) {
		assertNoDuplicates("mutex", runtime.MutexProfile)
	})
	t.Run("block", func(t *testing.T) {
		assertNoDuplicates("block", runtime.BlockProfile)
	})
}

func inlineA(mu *sync.Mutex, wg *sync.WaitGroup) { inlineB(mu, wg) }
func inlineB(mu *sync.Mutex, wg *sync.WaitGroup) { inlineC(mu, wg) }
func inlineC(mu *sync.Mutex, wg *sync.WaitGroup) {
	defer wg.Done()
	mu.Lock()
	mu.Unlock()
}

func inlineD(mu *sync.Mutex, wg *sync.WaitGroup) { inlineE(mu, wg) }
func inlineE(mu *sync.Mutex, wg *sync.WaitGroup) { inlineF(mu, wg) }
func inlineF(mu *sync.Mutex, wg *sync.WaitGroup) {
	defer wg.Done()
	mu.Unlock()
}

func TestBlockMutexProfileInlineExpansion(t *testing.T) {
	runtime.SetBlockProfileRate(1)
	defer runtime.SetBlockProfileRate(0)
	prev := runtime.SetMutexProfileFraction(1)
	defer runtime.SetMutexProfileFraction(prev)

	var mu sync.Mutex
	var wg sync.WaitGroup
	wg.Add(2)
	mu.Lock()
	go inlineA(&mu, &wg)
	awaitBlockedGoroutine(t, "sync.Mutex.Lock", "inlineC", 1)
	// inlineD will unblock inlineA
	go inlineD(&mu, &wg)
	wg.Wait()

	tcs := []struct {
		Name     string
		Collect  func([]runtime.BlockProfileRecord) (int, bool)
		SubStack string
	}{
		{
			Name:    "mutex",
			Collect: runtime.MutexProfile,
			SubStack: `sync.(*Mutex).Unlock
runtime/pprof.inlineF
runtime/pprof.inlineE
runtime/pprof.inlineD`,
		},
		{
			Name:    "block",
			Collect: runtime.BlockProfile,
			SubStack: `sync.(*Mutex).Lock
runtime/pprof.inlineC
runtime/pprof.inlineB
runtime/pprof.inlineA`,
		},
	}

	for _, tc := range tcs {
		t.Run(tc.Name, func(t *testing.T) {
			stacks := getProfileStacks(tc.Collect, false, false)
			for _, s := range stacks {
				if strings.Contains(s, tc.SubStack) {
					return
				}
			}
			t.Error("did not see expected stack")
			t.Logf("wanted:\n%s", tc.SubStack)
			t.Logf("got: %s", stacks)
		})
	}
}

func TestProfileRecordNullPadding(t *testing.T) {
	// Produce events for the different profile types.
	t.Cleanup(disableSampling())
	memSink = make([]byte, 1)      // MemProfile
	<-time.After(time.Millisecond) // BlockProfile
	blockMutex(t)                  // MutexProfile
	runtime.GC()

	// Test that all profile records are null padded.
	testProfileRecordNullPadding(t, "MutexProfile", runtime.MutexProfile)
	testProfileRecordNullPadding(t, "GoroutineProfile", runtime.GoroutineProfile)
	testProfileRecordNullPadding(t, "BlockProfile", runtime.BlockProfile)
	testProfileRecordNullPadding(t, "MemProfile/inUseZero=true", func(p []runtime.MemProfileRecord) (int, bool) {
		return runtime.MemProfile(p, true)
	})
	testProfileRecordNullPadding(t, "MemProfile/inUseZero=false", func(p []runtime.MemProfileRecord) (int, bool) {
		return runtime.MemProfile(p, false)
	})
	// Not testing ThreadCreateProfile because it is broken, see issue 6104.
}

func testProfileRecordNullPadding[T runtime.StackRecord | runtime.MemProfileRecord | runtime.BlockProfileRecord](t *testing.T, name string, fn func([]T) (int, bool)) {
	stack0 := func(sr *T) *[32]uintptr {
		switch t := any(sr).(type) {
		case *runtime.StackRecord:
			return &t.Stack0
		case *runtime.MemProfileRecord:
			return &t.Stack0
		case *runtime.BlockProfileRecord:
			return &t.Stack0
		default:
			panic(fmt.Sprintf("unexpected type %T", sr))
		}
	}

	t.Run(name, func(t *testing.T) {
		var p []T
		for {
			n, ok := fn(p)
			if ok {
				p = p[:n]
				break
			}
			p = make([]T, n*2)
			for i := range p {
				s0 := stack0(&p[i])
				for j := range s0 {
					// Poison the Stack0 array to identify lack of zero padding
					s0[j] = ^uintptr(0)
				}
			}
		}

		if len(p) == 0 {
			t.Fatal("no records found")
		}

		for _, sr := range p {
			for i, v := range stack0(&sr) {
				if v == ^uintptr(0) {
					t.Fatalf("record p[%d].Stack0 is not null padded: %+v", i, sr)
				}
			}
		}
	})
}

// disableSampling configures the profilers to capture all events, otherwise
// it's difficult to assert anything.
func disableSampling() func() {
	oldMemRate := runtime.MemProfileRate
	runtime.MemProfileRate = 1
	runtime.SetBlockProfileRate(1)
	oldMutexRate := runtime.SetMutexProfileFraction(1)
	return func() {
		runtime.MemProfileRate = oldMemRate
		runtime.SetBlockProfileRate(0)
		runtime.SetMutexProfileFraction(oldMutexRate)
	}
}
