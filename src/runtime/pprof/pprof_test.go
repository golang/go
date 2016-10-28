// Copyright 2011 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

// +build !nacl

package pprof_test

import (
	"bytes"
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

func parseProfile(t *testing.T, prof bytes.Buffer, f func(*ProfileTest)) {
	//parse proto to profile struct
	r := bytes.NewReader(prof.Bytes())
	p, err := Parse(r)
	if err != nil {
		t.Fatalf("can't parse pprof profile: %v", err)
	}
	f(p)
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
	var have []string
	var samples uintptr
	parseProfile(t, prof, func(p *ProfileTest) {
		for s := range p.Sample {
			samples += (uintptr)(p.Sample[s].Value[0])
		}
		for i := range p.Function {
			f := p.Function[i]
			if f == nil {
				continue
			}
			for i, name := range need {
				if strings.Contains(f.Name, name) {
					have = append(have, need[i])
				}
			}
			if strings.Contains(f.Name, "stackBarrier") {
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
	if len(have) != len(need) {
		return !ok
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
		parseProfile(t, prof, func(p *ProfileTest) {})
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
		re   []string
	}
	tests := [...]TestCase{
		{"chan recv", blockChanRecv, []string{`runtime\.chanrecv1`, `.*/src/runtime/chan.go`, `runtime/pprof_test\.blockChanRecv`, `.*/src/runtime/pprof/pprof_test.go`, `runtime/pprof_test\.TestBlockProfile`, `.*/src/runtime/pprof/pprof_test.go`}},
		{"chan send", blockChanSend, []string{`runtime\.chansend1`, `.*/src/runtime/chan.go`, `runtime/pprof_test\.blockChanSend`, `.*/src/runtime/pprof/pprof_test.go`, `runtime/pprof_test\.TestBlockProfile`, `.*/src/runtime/pprof/pprof_test.go`}},
		{"chan close", blockChanClose, []string{`runtime\.chanrecv1`, `.*/src/runtime/chan.go`, `runtime/pprof_test\.blockChanClose`, `.*/src/runtime/pprof/pprof_test.go`, `runtime/pprof_test\.TestBlockProfile`, `.*/src/runtime/pprof/pprof_test.go`}},
		{"select recv async", blockSelectRecvAsync, []string{`runtime\.selectgo`, `.*/src/runtime/select.go`, `runtime/pprof_test\.blockSelectRecvAsync`, `.*/src/runtime/pprof/pprof_test.go`, `runtime/pprof_test\.TestBlockProfile`, `.*/src/runtime/pprof/pprof_test.go`}},
		{"select send sync", blockSelectSendSync, []string{`runtime\.selectgo`, `.*/src/runtime/select.go`, `runtime/pprof_test\.blockSelectSendSync`, `.*/src/runtime/pprof/pprof_test.go`, `runtime/pprof_test\.TestBlockProfile`, `.*/src/runtime/pprof/pprof_test.go`}},
		{"mutex", blockMutex, []string{`sync\.\(\*Mutex\)\.Lock`, `.*/src/sync/mutex\.go`, `runtime/pprof_test\.blockMutex`, `.*/src/runtime/pprof/pprof_test.go`, `runtime/pprof_test\.TestBlockProfile`, `.*/src/runtime/pprof/pprof_test.go`}},
		{"cond", blockCond, []string{`sync\.\(\*Cond\)\.Wait`, `.*/src/sync/cond\.go`, `runtime/pprof_test\.blockCond`, `.*/src/runtime/pprof/pprof_test.go`, `runtime/pprof_test\.TestBlockProfile`, `.*/src/runtime/pprof/pprof_test.go`}},
	}

	runtime.SetBlockProfileRate(1)
	defer runtime.SetBlockProfileRate(0)
	for _, test := range tests {
		test.f()
		var prof bytes.Buffer
		Lookup("block").WriteTo(&prof, 1)

		parseProfile(t, prof, func(p *ProfileTest) {
			for n := 0; n < len(test.re); n += 2 {
				found := false
				for i := range p.Function {
					f := p.Function[i]
					t.Log(f.Name, f.Filename)
					if !regexp.MustCompile(strings.Replace(test.re[n], "\t", "\t+", -1)).MatchString(f.Name) || !regexp.MustCompile(strings.Replace(test.re[n+1], "\t", "\t+", -1)).MatchString(f.Filename) {
						found = true
						break
					}
				}
				if !found {
					t.Fatalf("have not found expected function %s from file %s", test.re[n], test.re[n+1])
				}
			}
		})
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

func TestMutexProfile(t *testing.T) {
	old := runtime.SetMutexProfileFraction(1)
	defer runtime.SetMutexProfileFraction(old)
	if old != 0 {
		t.Fatalf("need MutexProfileRate 0, got %d", old)
	}

	blockMutex()

	var w bytes.Buffer
	Lookup("mutex").WriteTo(&w, 1)
	prof := w.String()

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
	r2 := `^\d+ 1 @(?: 0x[[:xdigit:]]+)+`
	//r2 := "^[0-9]+ 1 @ 0x[0-9a-f x]+$"
	if ok, err := regexp.MatchString(r2, lines[3]); err != nil || !ok {
		t.Errorf("%q didn't match %q", lines[3], r2)
	}
	r3 := "^#.*runtime/pprof_test.blockMutex.*$"
	if ok, err := regexp.MatchString(r3, lines[5]); err != nil || !ok {
		t.Errorf("%q didn't match %q", lines[5], r3)
	}
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

	var prof bytes.Buffer
	Lookup("goroutine").WriteTo(&prof, 1)

	parseProfile(t, prof, func(p *ProfileTest) {
		if len(p.Sample) < 4 {
			t.Errorf("few samples, got %v", len(p.Sample))
		}
		if p.Sample[0].Value[0] != 50 || p.Sample[1].Value[0] != 40 || p.Sample[2].Value[0] != 10 || p.Sample[3].Value[0] != 1 {
			t.Errorf("expected sorted goroutine counts:\n 50, 40, 10, 1\ngot:\n", p.Sample[0].Value[0], p.Sample[1].Value[0], p.Sample[2].Value[0], p.Sample[3].Value[0])
		}
	})
	close(c)

	time.Sleep(10 * time.Millisecond) // let goroutines exit
}
