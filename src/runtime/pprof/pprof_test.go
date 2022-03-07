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
	"internal/testenv"
	"io"
	"math"
	"math/big"
	"os"
	"os/exec"
	"regexp"
	"runtime"
	"runtime/debug"
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

	// Linux [5.9,5.16) has a kernel bug that can break CPU timers on newly
	// created threads, breaking our CPU accounting.
	major, minor, patch, err := linuxKernelVersion()
	if err != nil {
		t.Errorf("Error determining kernel version: %v", err)
	}
	t.Logf("Running on Linux %d.%d.%d", major, minor, patch)
	defer func() {
		if t.Failed() {
			t.Logf("Failure of this test may indicate that your system suffers from a known Linux kernel bug fixed on newer kernels. See https://golang.org/issue/49065.")
		}
	}()

	// Disable on affected builders to avoid flakiness, but otherwise keep
	// it enabled to potentially warn users that they are on a broken
	// kernel.
	if testenv.Builder() != "" && (runtime.GOARCH == "386" || runtime.GOARCH == "amd64") {
		have59 := major > 5 || (major == 5 && minor >= 9)
		have516 := major > 5 || (major == 5 && minor >= 16)
		if have59 && !have516 {
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

			var cpuTime time.Duration
			matches := matchAndAvoidStacks(stackContains, []string{"runtime/pprof.cpuHog1"}, avoidFunctions())
			p := testCPUProfile(t, matches, func(dur time.Duration) {
				cpuTime = diffCPUTime(t, func() {
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

			for i, unit := range []string{"count", "nanoseconds"} {
				if have, want := p.SampleType[i].Unit, unit; have != want {
					t.Errorf("pN SampleType[%d]; %q != %q", i, have, want)
				}
			}

			// cpuHog1 called above is the primary source of CPU
			// load, but there may be some background work by the
			// runtime. Since the OS rusage measurement will
			// include all work done by the process, also compare
			// against all samples in our profile.
			var value time.Duration
			for _, sample := range p.Sample {
				value += time.Duration(sample.Value[1]) * time.Nanosecond
			}

			t.Logf("compare %s vs %s", cpuTime, value)
			if err := compare(cpuTime, value, maxDiff); err != nil {
				t.Errorf("compare got %v want nil", err)
			}
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

func cpuProfilingBroken() bool {
	switch runtime.GOOS {
	case "plan9":
		// Profiling unimplemented.
		return true
	case "aix":
		// See https://golang.org/issue/45170.
		return true
	case "ios", "dragonfly", "netbsd", "illumos", "solaris":
		// See https://golang.org/issue/13841.
		return true
	case "openbsd":
		if runtime.GOARCH == "arm" || runtime.GOARCH == "arm64" {
			// See https://golang.org/issue/13841.
			return true
		}
	}

	return false
}

// testCPUProfile runs f under the CPU profiler, checking for some conditions specified by need,
// as interpreted by matches, and returns the parsed profile.
func testCPUProfile(t *testing.T, matches profileMatchFunc, f func(dur time.Duration)) *profile.Profile {
	switch runtime.GOOS {
	case "darwin":
		out, err := exec.Command("uname", "-a").CombinedOutput()
		if err != nil {
			t.Fatal(err)
		}
		vers := string(out)
		t.Logf("uname -a: %v", vers)
	case "plan9":
		t.Skip("skipping on plan9")
	}

	broken := cpuProfilingBroken()

	maxDuration := 5 * time.Second
	if testing.Short() && broken {
		// If it's expected to be broken, no point waiting around.
		maxDuration /= 10
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
	for duration <= maxDuration {
		var prof bytes.Buffer
		if err := StartCPUProfile(&prof); err != nil {
			t.Fatal(err)
		}
		f(duration)
		StopCPUProfile()

		if p, ok := profileOk(t, matches, prof, duration); ok {
			return p
		}

		duration *= 2
		if duration <= maxDuration {
			t.Logf("retrying with %s duration", duration)
		}
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

var diffCPUTimeImpl func(f func()) time.Duration

func diffCPUTime(t *testing.T, f func()) time.Duration {
	if fn := diffCPUTimeImpl; fn != nil {
		return fn(f)
	}
	t.Fatalf("cannot measure CPU time on GOOS=%s GOARCH=%s", runtime.GOOS, runtime.GOARCH)
	return 0
}

func contains(slice []string, s string) bool {
	for i := range slice {
		if slice[i] == s {
			return true
		}
	}
	return false
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

func profileOk(t *testing.T, matches profileMatchFunc, prof bytes.Buffer, duration time.Duration) (_ *profile.Profile, ok bool) {
	ok = true

	var samples uintptr
	var buf bytes.Buffer
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
			t.Logf("%s: %d\n", name, have[i])
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

	heap := 1 << 30
	if runtime.GOOS == "android" {
		// Use smaller size for Android to avoid crash.
		heap = 100 << 20
	}
	if runtime.GOOS == "windows" && runtime.GOARCH == "arm" {
		// Use smaller heap for Windows/ARM to avoid crash.
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
		exec.Command(os.Args[0], "-h").CombinedOutput()
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
				var buf bytes.Buffer
				fprintStack(&buf, stk)
				t.Fatalf("found profile entry for gogo:\n%s", buf.String())
			}
		})
	}
}

func fprintStack(w io.Writer, stk []*profile.Location) {
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
				c <- true
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
#	0x[0-9a-f]+	runtime\.chanrecv1\+0x[0-9a-f]+	.*/src/runtime/chan.go:[0-9]+
#	0x[0-9a-f]+	runtime/pprof\.blockChanRecv\+0x[0-9a-f]+	.*/src/runtime/pprof/pprof_test.go:[0-9]+
#	0x[0-9a-f]+	runtime/pprof\.TestBlockProfile\+0x[0-9a-f]+	.*/src/runtime/pprof/pprof_test.go:[0-9]+
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
#	0x[0-9a-f]+	runtime\.chansend1\+0x[0-9a-f]+	.*/src/runtime/chan.go:[0-9]+
#	0x[0-9a-f]+	runtime/pprof\.blockChanSend\+0x[0-9a-f]+	.*/src/runtime/pprof/pprof_test.go:[0-9]+
#	0x[0-9a-f]+	runtime/pprof\.TestBlockProfile\+0x[0-9a-f]+	.*/src/runtime/pprof/pprof_test.go:[0-9]+
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
#	0x[0-9a-f]+	runtime\.chanrecv1\+0x[0-9a-f]+	.*/src/runtime/chan.go:[0-9]+
#	0x[0-9a-f]+	runtime/pprof\.blockChanClose\+0x[0-9a-f]+	.*/src/runtime/pprof/pprof_test.go:[0-9]+
#	0x[0-9a-f]+	runtime/pprof\.TestBlockProfile\+0x[0-9a-f]+	.*/src/runtime/pprof/pprof_test.go:[0-9]+
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
#	0x[0-9a-f]+	runtime\.selectgo\+0x[0-9a-f]+	.*/src/runtime/select.go:[0-9]+
#	0x[0-9a-f]+	runtime/pprof\.blockSelectRecvAsync\+0x[0-9a-f]+	.*/src/runtime/pprof/pprof_test.go:[0-9]+
#	0x[0-9a-f]+	runtime/pprof\.TestBlockProfile\+0x[0-9a-f]+	.*/src/runtime/pprof/pprof_test.go:[0-9]+
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
#	0x[0-9a-f]+	runtime\.selectgo\+0x[0-9a-f]+	.*/src/runtime/select.go:[0-9]+
#	0x[0-9a-f]+	runtime/pprof\.blockSelectSendSync\+0x[0-9a-f]+	.*/src/runtime/pprof/pprof_test.go:[0-9]+
#	0x[0-9a-f]+	runtime/pprof\.TestBlockProfile\+0x[0-9a-f]+	.*/src/runtime/pprof/pprof_test.go:[0-9]+
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
#	0x[0-9a-f]+	sync\.\(\*Mutex\)\.Lock\+0x[0-9a-f]+	.*/src/sync/mutex\.go:[0-9]+
#	0x[0-9a-f]+	runtime/pprof\.blockMutex\+0x[0-9a-f]+	.*/src/runtime/pprof/pprof_test.go:[0-9]+
#	0x[0-9a-f]+	runtime/pprof\.TestBlockProfile\+0x[0-9a-f]+	.*/src/runtime/pprof/pprof_test.go:[0-9]+
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
#	0x[0-9a-f]+	sync\.\(\*Cond\)\.Wait\+0x[0-9a-f]+	.*/src/sync/cond\.go:[0-9]+
#	0x[0-9a-f]+	runtime/pprof\.blockCond\+0x[0-9a-f]+	.*/src/runtime/pprof/pprof_test.go:[0-9]+
#	0x[0-9a-f]+	runtime/pprof\.TestBlockProfile\+0x[0-9a-f]+	.*/src/runtime/pprof/pprof_test.go:[0-9]+
`},
	}

	// Generate block profile
	runtime.SetBlockProfileRate(1)
	defer runtime.SetBlockProfileRate(0)
	for _, test := range tests {
		test.f(t)
	}

	t.Run("debug=1", func(t *testing.T) {
		var w bytes.Buffer
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

		stks := stacks(p)
		for _, test := range tests {
			if !containsStack(stks, test.stk) {
				t.Errorf("No matching stack entry for %v, want %+v", test.name, test.stk)
			}
		}
	})

}

func stacks(p *profile.Profile) (res [][]string) {
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
func awaitBlockedGoroutine(t *testing.T, state, fName string) {
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
		if r.Match(buf[:n]) {
			return
		}
	}
}

func blockChanRecv(t *testing.T) {
	c := make(chan bool)
	go func() {
		awaitBlockedGoroutine(t, "chan receive", "blockChanRecv")
		c <- true
	}()
	<-c
}

func blockChanSend(t *testing.T) {
	c := make(chan bool)
	go func() {
		awaitBlockedGoroutine(t, "chan send", "blockChanSend")
		<-c
	}()
	c <- true
}

func blockChanClose(t *testing.T) {
	c := make(chan bool)
	go func() {
		awaitBlockedGoroutine(t, "chan receive", "blockChanClose")
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
			awaitBlockedGoroutine(t, "select", "blockSelectRecvAsync")
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
		awaitBlockedGoroutine(t, "select", "blockSelectSendSync")
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
		awaitBlockedGoroutine(t, "semacquire", "blockMutex")
		mu.Unlock()
	}()
	// Note: Unlock releases mu before recording the mutex event,
	// so it's theoretically possible for this to proceed and
	// capture the profile before the event is recorded. As long
	// as this is blocked before the unlock happens, it's okay.
	mu.Lock()
}

func blockCond(t *testing.T) {
	var mu sync.Mutex
	c := sync.NewCond(&mu)
	mu.Lock()
	go func() {
		awaitBlockedGoroutine(t, "sync.Cond.Wait", "blockCond")
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

// blockFrequentShort produces 10000 block events with an average duration of
// rate.
func blockInfrequentLong(rate int) {
	for i := 0; i < 10000; i++ {
		blockevent(int64(rate), 1)
	}
}

// Used by TestBlockProfileBias.
//go:linkname blockevent runtime.blockevent
func blockevent(cycles int64, skip int)

func TestMutexProfile(t *testing.T) {
	// Generate mutex profile

	old := runtime.SetMutexProfileFraction(1)
	defer runtime.SetMutexProfileFraction(old)
	if old != 0 {
		t.Fatalf("need MutexProfileRate 0, got %d", old)
	}

	blockMutex(t)

	t.Run("debug=1", func(t *testing.T) {
		var w bytes.Buffer
		Lookup("mutex").WriteTo(&w, 1)
		prof := w.String()
		t.Logf("received profile: %v", prof)

		if !strings.HasPrefix(prof, "--- mutex:\ncycles/second=") {
			t.Errorf("Bad profile header:\n%v", prof)
		}
		prof = strings.Trim(prof, "\n")
		lines := strings.Split(prof, "\n")
		if len(lines) != 6 {
			t.Errorf("expected 6 lines, got %d %q\n%s", len(lines), prof, prof)
		}
		if len(lines) < 6 {
			return
		}
		// checking that the line is like "35258904 1 @ 0x48288d 0x47cd28 0x458931"
		r2 := `^\d+ \d+ @(?: 0x[[:xdigit:]]+)+`
		//r2 := "^[0-9]+ 1 @ 0x[0-9a-f x]+$"
		if ok, err := regexp.MatchString(r2, lines[3]); err != nil || !ok {
			t.Errorf("%q didn't match %q", lines[3], r2)
		}
		r3 := "^#.*runtime/pprof.blockMutex.*$"
		if ok, err := regexp.MatchString(r3, lines[5]); err != nil || !ok {
			t.Errorf("%q didn't match %q", lines[5], r3)
		}
		t.Logf(prof)
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

		stks := stacks(p)
		for _, want := range [][]string{
			{"sync.(*Mutex).Unlock", "runtime/pprof.blockMutex.func1"},
		} {
			if !containsStack(stks, want) {
				t.Errorf("No matching stack entry for %+v", want)
			}
		}
	})
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

	var w bytes.Buffer
	goroutineProf := Lookup("goroutine")

	// Check debug profile
	goroutineProf.WriteTo(&w, 1)
	prof := w.String()

	labels := labelMap{"label": "value"}
	labelStr := "\n# labels: " + labels.String()
	if !containsInOrder(prof, "\n50 @ ", "\n44 @", labelStr,
		"\n40 @", "\n36 @", labelStr, "\n10 @", "\n9 @", labelStr, "\n1 @") {
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
		50: map[string]string{},
		44: map[string]string{"label": "value"},
		40: map[string]string{},
		36: map[string]string{"label": "value"},
		10: map[string]string{},
		9:  map[string]string{"label": "value"},
		1:  map[string]string{},
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

var emptyCallStackTestRun int64

// Issue 18836.
func TestEmptyCallStack(t *testing.T) {
	name := fmt.Sprintf("test18836_%d", emptyCallStackTestRun)
	emptyCallStackTestRun++

	t.Parallel()
	var buf bytes.Buffer
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
	if !contains(labels[k], v) {
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
	// Test the race detector annotations for synchronization
	// between settings labels and consuming them from the
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
		isLabeled := s.Label != nil && contains(s.Label["key"], "value")
		var (
			mayBeLabeled     bool
			mustBeLabeled    bool
			mustNotBeLabeled bool
		)
		for _, loc := range s.Location {
			for _, l := range loc.Line {
				switch l.Function.Name {
				case "runtime/pprof.labelHog", "runtime/pprof.parallelLabelHog", "runtime/pprof.parallelLabelHog.func1":
					mustBeLabeled = true
				case "runtime/pprof.Do":
					// Do sets the labels, so samples may
					// or may not be labeled depending on
					// which part of the function they are
					// at.
					mayBeLabeled = true
				case "runtime.bgsweep", "runtime.bgscavenge", "runtime.forcegchelper", "runtime.gcBgMarkWorker", "runtime.runfinq", "runtime.sysmon":
					// Runtime system goroutines or threads
					// (such as those identified by
					// runtime.isSystemGoroutine). These
					// should never be labeled.
					mustNotBeLabeled = true
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
		if mustNotBeLabeled {
			// If this must not be labeled, then mayBeLabeled hints
			// are not relevant.
			mayBeLabeled = false
		}
		if mustBeLabeled && !isLabeled {
			var buf bytes.Buffer
			fprintStack(&buf, s.Location)
			t.Errorf("Sample labeled got false want true: %s", buf.String())
		}
		if mustNotBeLabeled && isLabeled {
			var buf bytes.Buffer
			fprintStack(&buf, s.Location)
			t.Errorf("Sample labeled got true want false: %s", buf.String())
		}
		if isLabeled && !(mayBeLabeled || mustBeLabeled) {
			var buf bytes.Buffer
			fprintStack(&buf, s.Location)
			t.Errorf("Sample labeled got true want false: %s", buf.String())
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
