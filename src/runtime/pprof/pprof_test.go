// Copyright 2011 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

// +build !nacl,!js

package pprof

import (
	"bytes"
	"context"
	"fmt"
	"internal/testenv"
	"io"
	"io/ioutil"
	"math/big"
	"os"
	"os/exec"
	"regexp"
	"runtime"
	"runtime/pprof/internal/profile"
	"strings"
	"sync"
	"sync/atomic"
	"testing"
	"time"
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
	foo := x
	for i := 0; i < 1e5; i++ {
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
	testCPUProfile(t, stackContains, []string{"runtime/pprof.cpuHog1"}, avoidFunctions(), func(dur time.Duration) {
		cpuHogger(cpuHog1, &salt1, dur)
	})
}

func TestCPUProfileMultithreaded(t *testing.T) {
	defer runtime.GOMAXPROCS(runtime.GOMAXPROCS(2))
	testCPUProfile(t, stackContains, []string{"runtime/pprof.cpuHog1", "runtime/pprof.cpuHog2"}, avoidFunctions(), func(dur time.Duration) {
		c := make(chan int)
		go func() {
			cpuHogger(cpuHog1, &salt1, dur)
			c <- 1
		}()
		cpuHogger(cpuHog2, &salt2, dur)
		<-c
	})
}

func TestCPUProfileInlining(t *testing.T) {
	testCPUProfile(t, stackContains, []string{"runtime/pprof.inlinedCallee", "runtime/pprof.inlinedCaller"}, avoidFunctions(), func(dur time.Duration) {
		cpuHogger(inlinedCaller, &salt1, dur)
	})
}

func inlinedCaller(x int) int {
	x = inlinedCallee(x)
	return x
}

func inlinedCallee(x int) int {
	// We could just use cpuHog1, but for loops prevent inlining
	// right now. :(
	foo := x
	i := 0
loop:
	if foo > 0 {
		foo *= foo
	} else {
		foo *= foo + 1
	}
	if i++; i < 1e5 {
		goto loop
	}
	return foo
}

func parseProfile(t *testing.T, valBytes []byte, f func(uintptr, []*profile.Location, map[string][]string)) {
	p, err := profile.Parse(bytes.NewReader(valBytes))
	if err != nil {
		t.Fatal(err)
	}
	for _, sample := range p.Sample {
		count := uintptr(sample.Value[0])
		f(count, sample.Location, sample.Label)
	}
}

// testCPUProfile runs f under the CPU profiler, checking for some conditions specified by need,
// as interpreted by matches.
func testCPUProfile(t *testing.T, matches matchFunc, need []string, avoid []string, f func(dur time.Duration)) {
	switch runtime.GOOS {
	case "darwin":
		switch runtime.GOARCH {
		case "arm", "arm64":
			// nothing
		default:
			out, err := exec.Command("uname", "-a").CombinedOutput()
			if err != nil {
				t.Fatal(err)
			}
			vers := string(out)
			t.Logf("uname -a: %v", vers)
		}
	case "plan9":
		t.Skip("skipping on plan9")
	}

	broken := false
	switch runtime.GOOS {
	case "darwin", "dragonfly", "netbsd", "illumos", "solaris":
		broken = true
	case "openbsd":
		if runtime.GOARCH == "arm" || runtime.GOARCH == "arm64" {
			broken = true
		}
	}

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

		if profileOk(t, matches, need, avoid, prof, duration) {
			return
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

type matchFunc func(spec string, count uintptr, stk []*profile.Location, labels map[string][]string) bool

func profileOk(t *testing.T, matches matchFunc, need []string, avoid []string, prof bytes.Buffer, duration time.Duration) (ok bool) {
	ok = true

	// Check that profile is well formed, contains 'need', and does not contain
	// anything from 'avoid'.
	have := make([]uintptr, len(need))
	avoidSamples := make([]uintptr, len(avoid))
	var samples uintptr
	var buf bytes.Buffer
	parseProfile(t, prof.Bytes(), func(count uintptr, stk []*profile.Location, labels map[string][]string) {
		fmt.Fprintf(&buf, "%d:", count)
		fprintStack(&buf, stk)
		samples += count
		for i, spec := range need {
			if matches(spec, count, stk, labels) {
				have[i] += count
			}
		}
		for i, name := range avoid {
			for _, loc := range stk {
				for _, line := range loc.Line {
					if strings.Contains(line.Function.Name, name) {
						avoidSamples[i] += count
					}
				}
			}
		}
		fmt.Fprintf(&buf, "\n")
	})
	t.Logf("total %d CPU profile samples collected:\n%s", samples, buf.String())

	if samples < 10 && runtime.GOOS == "windows" {
		// On some windows machines we end up with
		// not enough samples due to coarse timer
		// resolution. Let it go.
		t.Log("too few samples on Windows (golang.org/issue/10842)")
		return false
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

	for i, name := range avoid {
		bad := avoidSamples[i]
		if bad != 0 {
			t.Logf("found %d samples in avoid-function %s\n", bad, name)
			ok = false
		}
	}

	if len(need) == 0 {
		return ok
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
	return ok
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

		// Read profile to look for entries for runtime.gogo with an attempt at a traceback.
		// The special entry
		parseProfile(t, prof.Bytes(), func(count uintptr, stk []*profile.Location, _ map[string][]string) {
			// An entry with two frames with 'System' in its top frame
			// exists to record a PC without a traceback. Those are okay.
			if len(stk) == 2 {
				name := stk[1].Line[0].Function.Name
				if name == "runtime._System" || name == "runtime._ExternalCode" || name == "runtime._GC" {
					return
				}
			}

			// Otherwise, should not see runtime.gogo.
			// The place we'd see it would be the inner most frame.
			name := stk[0].Line[0].Function.Name
			if name == "runtime.gogo" {
				var buf bytes.Buffer
				fprintStack(&buf, stk)
				t.Fatalf("found profile entry for runtime.gogo:\n%s", buf.String())
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
	fmt.Fprintf(w, "\n")
}

// Test that profiling of division operations is okay, especially on ARM. See issue 6681.
func TestMathBigDivide(t *testing.T) {
	testCPUProfile(t, nil, nil, nil, func(duration time.Duration) {
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
	testCPUProfile(t, stackContainsAll, []string{"runtime.newstack,runtime/pprof.growstack"}, avoidFunctions(), func(duration time.Duration) {
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
	growstack()
}

//go:noinline
func growstack() {
	var buf [8 << 10]byte
	use(buf)
}

//go:noinline
func use(x [8 << 10]byte) {}

func TestBlockProfile(t *testing.T) {
	type TestCase struct {
		name string
		f    func()
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
		test.f()
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

const blockDelay = 10 * time.Millisecond

func blockChanRecv() {
	c := make(chan bool)
	go func() {
		time.Sleep(blockDelay)
		c <- true
	}()
	<-c
}

func blockChanSend() {
	c := make(chan bool)
	go func() {
		time.Sleep(blockDelay)
		<-c
	}()
	c <- true
}

func blockChanClose() {
	c := make(chan bool)
	go func() {
		time.Sleep(blockDelay)
		close(c)
	}()
	<-c
}

func blockSelectRecvAsync() {
	const numTries = 3
	c := make(chan bool, 1)
	c2 := make(chan bool, 1)
	go func() {
		for i := 0; i < numTries; i++ {
			time.Sleep(blockDelay)
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

func blockSelectSendSync() {
	c := make(chan bool)
	c2 := make(chan bool)
	go func() {
		time.Sleep(blockDelay)
		<-c
	}()
	select {
	case c <- true:
	case c2 <- true:
	}
}

func blockMutex() {
	var mu sync.Mutex
	mu.Lock()
	go func() {
		time.Sleep(blockDelay)
		mu.Unlock()
	}()
	// Note: Unlock releases mu before recording the mutex event,
	// so it's theoretically possible for this to proceed and
	// capture the profile before the event is recorded. As long
	// as this is blocked before the unlock happens, it's okay.
	mu.Lock()
}

func blockCond() {
	var mu sync.Mutex
	c := sync.NewCond(&mu)
	mu.Lock()
	go func() {
		time.Sleep(blockDelay)
		mu.Lock()
		c.Signal()
		mu.Unlock()
	}()
	c.Wait()
	mu.Unlock()
}

func TestMutexProfile(t *testing.T) {
	// Generate mutex profile

	old := runtime.SetMutexProfileFraction(1)
	defer runtime.SetMutexProfileFraction(old)
	if old != 0 {
		t.Fatalf("need MutexProfileRate 0, got %d", old)
	}

	blockMutex()

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

	var w bytes.Buffer
	goroutineProf := Lookup("goroutine")

	// Check debug profile
	goroutineProf.WriteTo(&w, 1)
	prof := w.String()

	if !containsInOrder(prof, "\n50 @ ", "\n40 @", "\n10 @", "\n1 @") {
		t.Errorf("expected sorted goroutine counts:\n%s", prof)
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
	if !containsCounts(p, []int64{50, 40, 10, 1}) {
		t.Errorf("expected count profile to contain goroutines with counts %v, got %v",
			[]int64{50, 40, 10, 1}, p)
	}

	close(c)

	time.Sleep(10 * time.Millisecond) // let goroutines exit
}

func containsInOrder(s string, all ...string) bool {
	for _, t := range all {
		i := strings.Index(s, t)
		if i < 0 {
			return false
		}
		s = s[i+len(t):]
	}
	return true
}

func containsCounts(prof *profile.Profile, counts []int64) bool {
	m := make(map[int64]int)
	for _, c := range counts {
		m[c]++
	}
	for _, s := range prof.Sample {
		// The count is the single value in the sample
		if len(s.Value) != 1 {
			return false
		}
		m[s.Value[0]]--
	}
	for _, n := range m {
		if n > 0 {
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
	semi := strings.Index(spec, ";")
	if semi == -1 {
		panic("no semicolon in key/value spec")
	}
	kv := strings.SplitN(spec[semi+1:], "=", 2)
	if len(kv) != 2 {
		panic("missing = in key/value spec")
	}
	if !contains(labels[kv[0]], kv[1]) {
		return false
	}
	return stackContains(spec[:semi], count, stk, labels)
}

func TestCPUProfileLabel(t *testing.T) {
	testCPUProfile(t, stackContainsLabeled, []string{"runtime/pprof.cpuHogger;key=value"}, avoidFunctions(), func(dur time.Duration) {
		Do(context.Background(), Labels("key", "value"), func(context.Context) {
			cpuHogger(cpuHog1, &salt1, dur)
		})
	})
}

func TestLabelRace(t *testing.T) {
	// Test the race detector annotations for synchronization
	// between settings labels and consuming them from the
	// profile.
	testCPUProfile(t, stackContainsLabeled, []string{"runtime/pprof.cpuHogger;key=value"}, nil, func(dur time.Duration) {
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

// Check that there is no deadlock when the program receives SIGPROF while in
// 64bit atomics' critical section. Used to happen on mips{,le}. See #20146.
func TestAtomicLoadStore64(t *testing.T) {
	f, err := ioutil.TempFile("", "profatomic")
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
	f, err := ioutil.TempFile("", "proftraceback")
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
