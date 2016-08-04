// Copyright 2011 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

// +build !nacl

package pprof_test

import (
	"bytes"
	"fmt"
	"internal/testenv"
	"math/big"
	"os"
	"os/exec"
	"regexp"
	"runtime"
	. "runtime/pprof"
	"strings"
	"sync"
	"testing"
	"time"
	"unsafe"
)

func cpuHogger(f func(), dur time.Duration) {
	// We only need to get one 100 Hz clock tick, so we've got
	// a large safety buffer.
	// But do at least 500 iterations (which should take about 100ms),
	// otherwise TestCPUProfileMultithreaded can fail if only one
	// thread is scheduled during the testing period.
	t0 := time.Now()
	for i := 0; i < 500 || time.Since(t0) < dur; i++ {
		f()
	}
}

var (
	salt1 = 0
	salt2 = 0
)

// The actual CPU hogging function.
// Must not call other functions nor access heap/globals in the loop,
// otherwise under race detector the samples will be in the race runtime.
func cpuHog1() {
	foo := salt1
	for i := 0; i < 1e5; i++ {
		if foo > 0 {
			foo *= foo
		} else {
			foo *= foo + 1
		}
	}
	salt1 = foo
}

func cpuHog2() {
	foo := salt2
	for i := 0; i < 1e5; i++ {
		if foo > 0 {
			foo *= foo
		} else {
			foo *= foo + 2
		}
	}
	salt2 = foo
}

func TestCPUProfile(t *testing.T) {
	testCPUProfile(t, []string{"runtime/pprof_test.cpuHog1"}, func(dur time.Duration) {
		cpuHogger(cpuHog1, dur)
	})
}

func TestCPUProfileMultithreaded(t *testing.T) {
	defer runtime.GOMAXPROCS(runtime.GOMAXPROCS(2))
	testCPUProfile(t, []string{"runtime/pprof_test.cpuHog1", "runtime/pprof_test.cpuHog2"}, func(dur time.Duration) {
		c := make(chan int)
		go func() {
			cpuHogger(cpuHog1, dur)
			c <- 1
		}()
		cpuHogger(cpuHog2, dur)
		<-c
	})
}

func parseProfile(t *testing.T, valBytes []byte, f func(uintptr, []uintptr)) {
	// Convert []byte to []uintptr.
	l := len(valBytes)
	if i := bytes.Index(valBytes, []byte("\nMAPPED_LIBRARIES:\n")); i >= 0 {
		l = i
	}
	l /= int(unsafe.Sizeof(uintptr(0)))
	val := *(*[]uintptr)(unsafe.Pointer(&valBytes))
	val = val[:l]

	// 5 for the header, 3 for the trailer.
	if l < 5+3 {
		t.Logf("profile too short: %#x", val)
		if badOS[runtime.GOOS] {
			t.Skipf("ignoring failure on %s; see golang.org/issue/13841", runtime.GOOS)
			return
		}
		t.FailNow()
	}

	hd, val, tl := val[:5], val[5:l-3], val[l-3:]
	if hd[0] != 0 || hd[1] != 3 || hd[2] != 0 || hd[3] != 1e6/100 || hd[4] != 0 {
		t.Fatalf("unexpected header %#x", hd)
	}

	if tl[0] != 0 || tl[1] != 1 || tl[2] != 0 {
		t.Fatalf("malformed end-of-data marker %#x", tl)
	}

	for len(val) > 0 {
		if len(val) < 2 || val[0] < 1 || val[1] < 1 || uintptr(len(val)) < 2+val[1] {
			t.Fatalf("malformed profile.  leftover: %#x", val)
		}
		f(val[0], val[2:2+val[1]])
		val = val[2+val[1]:]
	}
}

func testCPUProfile(t *testing.T, need []string, f func(dur time.Duration)) {
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

	const maxDuration = 5 * time.Second
	// If we're running a long test, start with a long duration
	// because some of the tests (e.g., TestStackBarrierProfiling)
	// are trying to make sure something *doesn't* happen.
	duration := 5 * time.Second
	if testing.Short() {
		duration = 200 * time.Millisecond
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

		if profileOk(t, need, prof, duration) {
			return
		}

		duration *= 2
		if duration <= maxDuration {
			t.Logf("retrying with %s duration", duration)
		}
	}

	if badOS[runtime.GOOS] {
		t.Skipf("ignoring failure on %s; see golang.org/issue/13841", runtime.GOOS)
		return
	}
	// Ignore the failure if the tests are running in a QEMU-based emulator,
	// QEMU is not perfect at emulating everything.
	// IN_QEMU environmental variable is set by some of the Go builders.
	// IN_QEMU=1 indicates that the tests are running in QEMU. See issue 9605.
	if os.Getenv("IN_QEMU") == "1" {
		t.Skip("ignore the failure in QEMU; see golang.org/issue/9605")
		return
	}
	t.FailNow()
}

func profileOk(t *testing.T, need []string, prof bytes.Buffer, duration time.Duration) (ok bool) {
	ok = true

	// Check that profile is well formed and contains need.
	have := make([]uintptr, len(need))
	var samples uintptr
	parseProfile(t, prof.Bytes(), func(count uintptr, stk []uintptr) {
		samples += count
		for _, pc := range stk {
			f := runtime.FuncForPC(pc)
			if f == nil {
				continue
			}
			for i, name := range need {
				if strings.Contains(f.Name(), name) {
					have[i] += count
				}
			}
			if strings.Contains(f.Name(), "stackBarrier") {
				// The runtime should have unwound this.
				t.Fatalf("profile includes stackBarrier")
			}
		}
	})
	t.Logf("total %d CPU profile samples collected", samples)

	if samples < 10 && runtime.GOOS == "windows" {
		// On some windows machines we end up with
		// not enough samples due to coarse timer
		// resolution. Let it go.
		t.Log("too few samples on Windows (golang.org/issue/10842)")
		return false
	}

	// Check that we got a reasonable number of samples.
	if ideal := uintptr(duration * 100 / time.Second); samples == 0 || samples < ideal/4 {
		t.Logf("too few samples; got %d, want at least %d, ideally %d", samples, ideal/4, ideal)
		ok = false
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
		parseProfile(t, prof.Bytes(), func(count uintptr, stk []uintptr) {
			// An entry with two frames with 'System' in its top frame
			// exists to record a PC without a traceback. Those are okay.
			if len(stk) == 2 {
				f := runtime.FuncForPC(stk[1])
				if f != nil && (f.Name() == "runtime._System" || f.Name() == "runtime._ExternalCode" || f.Name() == "runtime._GC") {
					return
				}
			}

			// Otherwise, should not see runtime.gogo.
			// The place we'd see it would be the inner most frame.
			f := runtime.FuncForPC(stk[0])
			if f != nil && f.Name() == "runtime.gogo" {
				var buf bytes.Buffer
				for _, pc := range stk {
					f := runtime.FuncForPC(pc)
					if f == nil {
						fmt.Fprintf(&buf, "%#x ?:0\n", pc)
					} else {
						file, line := f.FileLine(pc)
						fmt.Fprintf(&buf, "%#x %s:%d\n", pc, file, line)
					}
				}
				t.Fatalf("found profile entry for runtime.gogo:\n%s", buf.String())
			}
		})
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

func TestStackBarrierProfiling(t *testing.T) {
	if (runtime.GOOS == "linux" && runtime.GOARCH == "arm") || runtime.GOOS == "openbsd" || runtime.GOOS == "solaris" || runtime.GOOS == "dragonfly" || runtime.GOOS == "freebsd" {
		// This test currently triggers a large number of
		// usleep(100)s. These kernels/arches have poor
		// resolution timers, so this gives up a whole
		// scheduling quantum. On Linux and the BSDs (and
		// probably Solaris), profiling signals are only
		// generated when a process completes a whole
		// scheduling quantum, so this test often gets zero
		// profiling signals and fails.
		t.Skipf("low resolution timers inhibit profiling signals (golang.org/issue/13405)")
		return
	}

	if !strings.Contains(os.Getenv("GODEBUG"), "gcstackbarrierall=1") {
		// Re-execute this test with constant GC and stack
		// barriers at every frame.
		testenv.MustHaveExec(t)
		if runtime.GOARCH == "ppc64" || runtime.GOARCH == "ppc64le" {
			t.Skip("gcstackbarrierall doesn't work on ppc64")
		}
		args := []string{"-test.run=TestStackBarrierProfiling"}
		if testing.Short() {
			args = append(args, "-test.short")
		}
		cmd := exec.Command(os.Args[0], args...)
		cmd.Env = append([]string{"GODEBUG=gcstackbarrierall=1", "GOGC=1", "GOTRACEBACK=system"}, os.Environ()...)
		if out, err := cmd.CombinedOutput(); err != nil {
			t.Fatalf("subprocess failed with %v:\n%s", err, out)
		}
		return
	}

	testCPUProfile(t, nil, func(duration time.Duration) {
		// In long mode, we're likely to get one or two
		// samples in stackBarrier.
		t := time.After(duration)
		for {
			deepStack(1000)
			select {
			case <-t:
				return
			default:
			}
		}
	})
}

var x []byte

func deepStack(depth int) int {
	if depth == 0 {
		return 0
	}
	x = make([]byte, 1024)
	return deepStack(depth-1) + 1
}

// Operating systems that are expected to fail the tests. See issue 13841.
var badOS = map[string]bool{
	"darwin":    true,
	"netbsd":    true,
	"plan9":     true,
	"dragonfly": true,
	"solaris":   true,
}

func TestBlockProfile(t *testing.T) {
	type TestCase struct {
		name string
		f    func()
		re   string
	}
	tests := [...]TestCase{
		{"chan recv", blockChanRecv, `
[0-9]+ [0-9]+ @ 0x[0-9,a-f]+ 0x[0-9,a-f]+ 0x[0-9,a-f]+ 0x[0-9,a-f]+ 0x[0-9,a-f]+
#	0x[0-9,a-f]+	runtime\.chanrecv1\+0x[0-9,a-f]+	.*/src/runtime/chan.go:[0-9]+
#	0x[0-9,a-f]+	runtime/pprof_test\.blockChanRecv\+0x[0-9,a-f]+	.*/src/runtime/pprof/pprof_test.go:[0-9]+
#	0x[0-9,a-f]+	runtime/pprof_test\.TestBlockProfile\+0x[0-9,a-f]+	.*/src/runtime/pprof/pprof_test.go:[0-9]+
`},
		{"chan send", blockChanSend, `
[0-9]+ [0-9]+ @ 0x[0-9,a-f]+ 0x[0-9,a-f]+ 0x[0-9,a-f]+ 0x[0-9,a-f]+ 0x[0-9,a-f]+
#	0x[0-9,a-f]+	runtime\.chansend1\+0x[0-9,a-f]+	.*/src/runtime/chan.go:[0-9]+
#	0x[0-9,a-f]+	runtime/pprof_test\.blockChanSend\+0x[0-9,a-f]+	.*/src/runtime/pprof/pprof_test.go:[0-9]+
#	0x[0-9,a-f]+	runtime/pprof_test\.TestBlockProfile\+0x[0-9,a-f]+	.*/src/runtime/pprof/pprof_test.go:[0-9]+
`},
		{"chan close", blockChanClose, `
[0-9]+ [0-9]+ @ 0x[0-9,a-f]+ 0x[0-9,a-f]+ 0x[0-9,a-f]+ 0x[0-9,a-f]+ 0x[0-9,a-f]+
#	0x[0-9,a-f]+	runtime\.chanrecv1\+0x[0-9,a-f]+	.*/src/runtime/chan.go:[0-9]+
#	0x[0-9,a-f]+	runtime/pprof_test\.blockChanClose\+0x[0-9,a-f]+	.*/src/runtime/pprof/pprof_test.go:[0-9]+
#	0x[0-9,a-f]+	runtime/pprof_test\.TestBlockProfile\+0x[0-9,a-f]+	.*/src/runtime/pprof/pprof_test.go:[0-9]+
`},
		{"select recv async", blockSelectRecvAsync, `
[0-9]+ [0-9]+ @ 0x[0-9,a-f]+ 0x[0-9,a-f]+ 0x[0-9,a-f]+ 0x[0-9,a-f]+ 0x[0-9,a-f]+
#	0x[0-9,a-f]+	runtime\.selectgo\+0x[0-9,a-f]+	.*/src/runtime/select.go:[0-9]+
#	0x[0-9,a-f]+	runtime/pprof_test\.blockSelectRecvAsync\+0x[0-9,a-f]+	.*/src/runtime/pprof/pprof_test.go:[0-9]+
#	0x[0-9,a-f]+	runtime/pprof_test\.TestBlockProfile\+0x[0-9,a-f]+	.*/src/runtime/pprof/pprof_test.go:[0-9]+
`},
		{"select send sync", blockSelectSendSync, `
[0-9]+ [0-9]+ @ 0x[0-9,a-f]+ 0x[0-9,a-f]+ 0x[0-9,a-f]+ 0x[0-9,a-f]+ 0x[0-9,a-f]+
#	0x[0-9,a-f]+	runtime\.selectgo\+0x[0-9,a-f]+	.*/src/runtime/select.go:[0-9]+
#	0x[0-9,a-f]+	runtime/pprof_test\.blockSelectSendSync\+0x[0-9,a-f]+	.*/src/runtime/pprof/pprof_test.go:[0-9]+
#	0x[0-9,a-f]+	runtime/pprof_test\.TestBlockProfile\+0x[0-9,a-f]+	.*/src/runtime/pprof/pprof_test.go:[0-9]+
`},
		{"mutex", blockMutex, `
[0-9]+ [0-9]+ @ 0x[0-9,a-f]+ 0x[0-9,a-f]+ 0x[0-9,a-f]+ 0x[0-9,a-f]+ 0x[0-9,a-f]+
#	0x[0-9,a-f]+	sync\.\(\*Mutex\)\.Lock\+0x[0-9,a-f]+	.*/src/sync/mutex\.go:[0-9]+
#	0x[0-9,a-f]+	runtime/pprof_test\.blockMutex\+0x[0-9,a-f]+	.*/src/runtime/pprof/pprof_test.go:[0-9]+
#	0x[0-9,a-f]+	runtime/pprof_test\.TestBlockProfile\+0x[0-9,a-f]+	.*/src/runtime/pprof/pprof_test.go:[0-9]+
`},
		{"cond", blockCond, `
[0-9]+ [0-9]+ @ 0x[0-9,a-f]+ 0x[0-9,a-f]+ 0x[0-9,a-f]+ 0x[0-9,a-f]+ 0x[0-9,a-f]+
#	0x[0-9,a-f]+	sync\.\(\*Cond\)\.Wait\+0x[0-9,a-f]+	.*/src/sync/cond\.go:[0-9]+
#	0x[0-9,a-f]+	runtime/pprof_test\.blockCond\+0x[0-9,a-f]+	.*/src/runtime/pprof/pprof_test.go:[0-9]+
#	0x[0-9,a-f]+	runtime/pprof_test\.TestBlockProfile\+0x[0-9,a-f]+	.*/src/runtime/pprof/pprof_test.go:[0-9]+
`},
	}

	runtime.SetBlockProfileRate(1)
	defer runtime.SetBlockProfileRate(0)
	for _, test := range tests {
		test.f()
	}
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
		if !regexp.MustCompile(strings.Replace(test.re, "\t", "\t+", -1)).MatchString(prof) {
			t.Fatalf("Bad %v entry, expect:\n%v\ngot:\n%v", test.name, test.re, prof)
		}
	}
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

func func1(c chan int) { <-c }
func func2(c chan int) { <-c }
func func3(c chan int) { <-c }
func func4(c chan int) { <-c }

func TestGoroutineCounts(t *testing.T) {
	if runtime.GOOS == "openbsd" {
		testenv.SkipFlaky(t, 15156)
	}
	c := make(chan int)
	for i := 0; i < 100; i++ {
		if i%10 == 0 {
			go func1(c)
			continue
		}
		if i%2 == 0 {
			go func2(c)
			continue
		}
		go func3(c)
	}
	time.Sleep(10 * time.Millisecond) // let goroutines block on channel

	var w bytes.Buffer
	Lookup("goroutine").WriteTo(&w, 1)
	prof := w.String()

	if !containsInOrder(prof, "\n50 @ ", "\n40 @", "\n10 @", "\n1 @") {
		t.Errorf("expected sorted goroutine counts:\n%s", prof)
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
