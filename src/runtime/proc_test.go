// Copyright 2011 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

package runtime_test

import (
	"bytes"
	"fmt"
	"internal/race"
	"internal/testenv"
	"internal/trace"
	"internal/trace/testtrace"
	"io"
	"math"
	"net"
	"runtime"
	"runtime/debug"
	"slices"
	"strings"
	"sync"
	"sync/atomic"
	"syscall"
	"testing"
	"time"
)

var stop = make(chan bool, 1)

func perpetuumMobile() {
	select {
	case <-stop:
	default:
		go perpetuumMobile()
	}
}

func TestStopTheWorldDeadlock(t *testing.T) {
	if runtime.GOARCH == "wasm" {
		t.Skip("no preemption on wasm yet")
	}
	if testing.Short() {
		t.Skip("skipping during short test")
	}
	maxprocs := runtime.GOMAXPROCS(3)
	compl := make(chan bool, 2)
	go func() {
		for i := 0; i != 1000; i += 1 {
			runtime.GC()
		}
		compl <- true
	}()
	go func() {
		for i := 0; i != 1000; i += 1 {
			runtime.GOMAXPROCS(3)
		}
		compl <- true
	}()
	go perpetuumMobile()
	<-compl
	<-compl
	stop <- true
	runtime.GOMAXPROCS(maxprocs)
}

func TestYieldProgress(t *testing.T) {
	testYieldProgress(false)
}

func TestYieldLockedProgress(t *testing.T) {
	testYieldProgress(true)
}

func testYieldProgress(locked bool) {
	c := make(chan bool)
	cack := make(chan bool)
	go func() {
		if locked {
			runtime.LockOSThread()
		}
		for {
			select {
			case <-c:
				cack <- true
				return
			default:
				runtime.Gosched()
			}
		}
	}()
	time.Sleep(10 * time.Millisecond)
	c <- true
	<-cack
}

func TestYieldLocked(t *testing.T) {
	const N = 10
	c := make(chan bool)
	go func() {
		runtime.LockOSThread()
		for i := 0; i < N; i++ {
			runtime.Gosched()
			time.Sleep(time.Millisecond)
		}
		c <- true
		// runtime.UnlockOSThread() is deliberately omitted
	}()
	<-c
}

func TestGoroutineParallelism(t *testing.T) {
	if runtime.NumCPU() == 1 {
		// Takes too long, too easy to deadlock, etc.
		t.Skip("skipping on uniprocessor")
	}
	P := 4
	N := 10
	if testing.Short() {
		P = 3
		N = 3
	}
	defer runtime.GOMAXPROCS(runtime.GOMAXPROCS(P))
	// If runtime triggers a forced GC during this test then it will deadlock,
	// since the goroutines can't be stopped/preempted.
	// Disable GC for this test (see issue #10958).
	defer debug.SetGCPercent(debug.SetGCPercent(-1))
	// SetGCPercent waits until the mark phase is over, but the runtime
	// also preempts at the start of the sweep phase, so make sure that's
	// done too. See #45867.
	runtime.GC()
	for try := 0; try < N; try++ {
		done := make(chan bool)
		x := uint32(0)
		for p := 0; p < P; p++ {
			// Test that all P goroutines are scheduled at the same time
			go func(p int) {
				for i := 0; i < 3; i++ {
					expected := uint32(P*i + p)
					for atomic.LoadUint32(&x) != expected {
					}
					atomic.StoreUint32(&x, expected+1)
				}
				done <- true
			}(p)
		}
		for p := 0; p < P; p++ {
			<-done
		}
	}
}

// Test that all runnable goroutines are scheduled at the same time.
func TestGoroutineParallelism2(t *testing.T) {
	//testGoroutineParallelism2(t, false, false)
	testGoroutineParallelism2(t, true, false)
	testGoroutineParallelism2(t, false, true)
	testGoroutineParallelism2(t, true, true)
}

func testGoroutineParallelism2(t *testing.T, load, netpoll bool) {
	if runtime.NumCPU() == 1 {
		// Takes too long, too easy to deadlock, etc.
		t.Skip("skipping on uniprocessor")
	}
	P := 4
	N := 10
	if testing.Short() {
		N = 3
	}
	defer runtime.GOMAXPROCS(runtime.GOMAXPROCS(P))
	// If runtime triggers a forced GC during this test then it will deadlock,
	// since the goroutines can't be stopped/preempted.
	// Disable GC for this test (see issue #10958).
	defer debug.SetGCPercent(debug.SetGCPercent(-1))
	// SetGCPercent waits until the mark phase is over, but the runtime
	// also preempts at the start of the sweep phase, so make sure that's
	// done too. See #45867.
	runtime.GC()
	for try := 0; try < N; try++ {
		if load {
			// Create P goroutines and wait until they all run.
			// When we run the actual test below, worker threads
			// running the goroutines will start parking.
			done := make(chan bool)
			x := uint32(0)
			for p := 0; p < P; p++ {
				go func() {
					if atomic.AddUint32(&x, 1) == uint32(P) {
						done <- true
						return
					}
					for atomic.LoadUint32(&x) != uint32(P) {
					}
				}()
			}
			<-done
		}
		if netpoll {
			// Enable netpoller, affects schedler behavior.
			laddr := "localhost:0"
			if runtime.GOOS == "android" {
				// On some Android devices, there are no records for localhost,
				// see https://golang.org/issues/14486.
				// Don't use 127.0.0.1 for every case, it won't work on IPv6-only systems.
				laddr = "127.0.0.1:0"
			}
			ln, err := net.Listen("tcp", laddr)
			if err == nil {
				defer ln.Close() // yup, defer in a loop
			}
		}
		done := make(chan bool)
		x := uint32(0)
		// Spawn P goroutines in a nested fashion just to differ from TestGoroutineParallelism.
		for p := 0; p < P/2; p++ {
			go func(p int) {
				for p2 := 0; p2 < 2; p2++ {
					go func(p2 int) {
						for i := 0; i < 3; i++ {
							expected := uint32(P*i + p*2 + p2)
							for atomic.LoadUint32(&x) != expected {
							}
							atomic.StoreUint32(&x, expected+1)
						}
						done <- true
					}(p2)
				}
			}(p)
		}
		for p := 0; p < P; p++ {
			<-done
		}
	}
}

func TestBlockLocked(t *testing.T) {
	const N = 10
	c := make(chan bool)
	go func() {
		runtime.LockOSThread()
		for i := 0; i < N; i++ {
			c <- true
		}
		runtime.UnlockOSThread()
	}()
	for i := 0; i < N; i++ {
		<-c
	}
}

func TestTimerFairness(t *testing.T) {
	if runtime.GOARCH == "wasm" {
		t.Skip("no preemption on wasm yet")
	}

	done := make(chan bool)
	c := make(chan bool)
	for i := 0; i < 2; i++ {
		go func() {
			for {
				select {
				case c <- true:
				case <-done:
					return
				}
			}
		}()
	}

	timer := time.After(20 * time.Millisecond)
	for {
		select {
		case <-c:
		case <-timer:
			close(done)
			return
		}
	}
}

func TestTimerFairness2(t *testing.T) {
	if runtime.GOARCH == "wasm" {
		t.Skip("no preemption on wasm yet")
	}

	done := make(chan bool)
	c := make(chan bool)
	for i := 0; i < 2; i++ {
		go func() {
			timer := time.After(20 * time.Millisecond)
			var buf [1]byte
			for {
				syscall.Read(0, buf[0:0])
				select {
				case c <- true:
				case <-c:
				case <-timer:
					done <- true
					return
				}
			}
		}()
	}
	<-done
	<-done
}

// The function is used to test preemption at split stack checks.
// Declaring a var avoids inlining at the call site.
var preempt = func() int {
	var a [128]int
	sum := 0
	for _, v := range a {
		sum += v
	}
	return sum
}

func TestPreemption(t *testing.T) {
	if runtime.GOARCH == "wasm" {
		t.Skip("no preemption on wasm yet")
	}

	// Test that goroutines are preempted at function calls.
	N := 5
	if testing.Short() {
		N = 2
	}
	c := make(chan bool)
	var x uint32
	for g := 0; g < 2; g++ {
		go func(g int) {
			for i := 0; i < N; i++ {
				for atomic.LoadUint32(&x) != uint32(g) {
					preempt()
				}
				atomic.StoreUint32(&x, uint32(1-g))
			}
			c <- true
		}(g)
	}
	<-c
	<-c
}

func TestPreemptionGC(t *testing.T) {
	if runtime.GOARCH == "wasm" {
		t.Skip("no preemption on wasm yet")
	}

	// Test that pending GC preempts running goroutines.
	P := 5
	N := 10
	if testing.Short() {
		P = 3
		N = 2
	}
	defer runtime.GOMAXPROCS(runtime.GOMAXPROCS(P + 1))
	var stop uint32
	for i := 0; i < P; i++ {
		go func() {
			for atomic.LoadUint32(&stop) == 0 {
				preempt()
			}
		}()
	}
	for i := 0; i < N; i++ {
		runtime.Gosched()
		runtime.GC()
	}
	atomic.StoreUint32(&stop, 1)
}

func TestAsyncPreempt(t *testing.T) {
	if !runtime.PreemptMSupported {
		t.Skip("asynchronous preemption not supported on this platform")
	}
	output := runTestProg(t, "testprog", "AsyncPreempt")
	want := "OK\n"
	if output != want {
		t.Fatalf("want %s, got %s\n", want, output)
	}
}

func TestGCFairness(t *testing.T) {
	output := runTestProg(t, "testprog", "GCFairness")
	want := "OK\n"
	if output != want {
		t.Fatalf("want %s, got %s\n", want, output)
	}
}

func TestGCFairness2(t *testing.T) {
	output := runTestProg(t, "testprog", "GCFairness2")
	want := "OK\n"
	if output != want {
		t.Fatalf("want %s, got %s\n", want, output)
	}
}

func TestNumGoroutine(t *testing.T) {
	output := runTestProg(t, "testprog", "NumGoroutine")
	want := "1\n"
	if output != want {
		t.Fatalf("want %q, got %q", want, output)
	}

	buf := make([]byte, 1<<20)

	// Try up to 10 times for a match before giving up.
	// This is a fundamentally racy check but it's important
	// to notice if NumGoroutine and Stack are _always_ out of sync.
	for i := 0; ; i++ {
		// Give goroutines about to exit a chance to exit.
		// The NumGoroutine and Stack below need to see
		// the same state of the world, so anything we can do
		// to keep it quiet is good.
		runtime.Gosched()

		n := runtime.NumGoroutine()
		buf = buf[:runtime.Stack(buf, true)]

		// To avoid double-counting "goroutine" in "goroutine $m [running]:"
		// and "created by $func in goroutine $n", remove the latter
		output := strings.ReplaceAll(string(buf), "in goroutine", "")
		nstk := strings.Count(output, "goroutine ")
		if n == nstk {
			break
		}
		if i >= 10 {
			t.Fatalf("NumGoroutine=%d, but found %d goroutines in stack dump: %s", n, nstk, buf)
		}
	}
}

func TestPingPongHog(t *testing.T) {
	if runtime.GOARCH == "wasm" {
		t.Skip("no preemption on wasm yet")
	}
	if testing.Short() {
		t.Skip("skipping in -short mode")
	}
	if race.Enabled {
		// The race detector randomizes the scheduler,
		// which causes this test to fail (#38266).
		t.Skip("skipping in -race mode")
	}

	defer runtime.GOMAXPROCS(runtime.GOMAXPROCS(1))
	done := make(chan bool)
	hogChan, lightChan := make(chan bool), make(chan bool)
	hogCount, lightCount := 0, 0

	run := func(limit int, counter *int, wake chan bool) {
		for {
			select {
			case <-done:
				return

			case <-wake:
				for i := 0; i < limit; i++ {
					*counter++
				}
				wake <- true
			}
		}
	}

	// Start two co-scheduled hog goroutines.
	for i := 0; i < 2; i++ {
		go run(1e6, &hogCount, hogChan)
	}

	// Start two co-scheduled light goroutines.
	for i := 0; i < 2; i++ {
		go run(1e3, &lightCount, lightChan)
	}

	// Start goroutine pairs and wait for a few preemption rounds.
	hogChan <- true
	lightChan <- true
	time.Sleep(100 * time.Millisecond)
	close(done)
	<-hogChan
	<-lightChan

	// Check that hogCount and lightCount are within a factor of
	// 20, which indicates that both pairs of goroutines handed off
	// the P within a time-slice to their buddy. We can use a
	// fairly large factor here to make this robust: if the
	// scheduler isn't working right, the gap should be ~1000X
	// (was 5, increased to 20, see issue 52207).
	const factor = 20
	if hogCount/factor > lightCount || lightCount/factor > hogCount {
		t.Fatalf("want hogCount/lightCount in [%v, %v]; got %d/%d = %g", 1.0/factor, factor, hogCount, lightCount, float64(hogCount)/float64(lightCount))
	}
}

func BenchmarkPingPongHog(b *testing.B) {
	if b.N == 0 {
		return
	}
	defer runtime.GOMAXPROCS(runtime.GOMAXPROCS(1))

	// Create a CPU hog
	stop, done := make(chan bool), make(chan bool)
	go func() {
		for {
			select {
			case <-stop:
				done <- true
				return
			default:
			}
		}
	}()

	// Ping-pong b.N times
	ping, pong := make(chan bool), make(chan bool)
	go func() {
		for j := 0; j < b.N; j++ {
			pong <- <-ping
		}
		close(stop)
		done <- true
	}()
	go func() {
		for i := 0; i < b.N; i++ {
			ping <- <-pong
		}
		done <- true
	}()
	b.ResetTimer()
	ping <- true // Start ping-pong
	<-stop
	b.StopTimer()
	<-ping // Let last ponger exit
	<-done // Make sure goroutines exit
	<-done
	<-done
}

var padData [128]uint64

func stackGrowthRecursive(i int) {
	var pad [128]uint64
	pad = padData
	for j := range pad {
		if pad[j] != 0 {
			return
		}
	}
	if i != 0 {
		stackGrowthRecursive(i - 1)
	}
}

func TestPreemptSplitBig(t *testing.T) {
	if testing.Short() {
		t.Skip("skipping in -short mode")
	}
	defer runtime.GOMAXPROCS(runtime.GOMAXPROCS(2))
	stop := make(chan int)
	go big(stop)
	for i := 0; i < 3; i++ {
		time.Sleep(10 * time.Microsecond) // let big start running
		runtime.GC()
	}
	close(stop)
}

func big(stop chan int) int {
	n := 0
	for {
		// delay so that gc is sure to have asked for a preemption
		for i := 0; i < 1e9; i++ {
			n++
		}

		// call bigframe, which used to miss the preemption in its prologue.
		bigframe(stop)

		// check if we've been asked to stop.
		select {
		case <-stop:
			return n
		}
	}
}

func bigframe(stop chan int) int {
	// not splitting the stack will overflow.
	// small will notice that it needs a stack split and will
	// catch the overflow.
	var x [8192]byte
	return small(stop, &x)
}

func small(stop chan int, x *[8192]byte) int {
	for i := range x {
		x[i] = byte(i)
	}
	sum := 0
	for i := range x {
		sum += int(x[i])
	}

	// keep small from being a leaf function, which might
	// make it not do any stack check at all.
	nonleaf(stop)

	return sum
}

func nonleaf(stop chan int) bool {
	// do something that won't be inlined:
	select {
	case <-stop:
		return true
	default:
		return false
	}
}

func TestSchedLocalQueue(t *testing.T) {
	runtime.RunSchedLocalQueueTest()
}

func TestSchedLocalQueueSteal(t *testing.T) {
	runtime.RunSchedLocalQueueStealTest()
}

func TestSchedLocalQueueEmpty(t *testing.T) {
	if runtime.NumCPU() == 1 {
		// Takes too long and does not trigger the race.
		t.Skip("skipping on uniprocessor")
	}
	defer runtime.GOMAXPROCS(runtime.GOMAXPROCS(4))

	// If runtime triggers a forced GC during this test then it will deadlock,
	// since the goroutines can't be stopped/preempted during spin wait.
	defer debug.SetGCPercent(debug.SetGCPercent(-1))
	// SetGCPercent waits until the mark phase is over, but the runtime
	// also preempts at the start of the sweep phase, so make sure that's
	// done too. See #45867.
	runtime.GC()

	iters := int(1e5)
	if testing.Short() {
		iters = 1e2
	}
	runtime.RunSchedLocalQueueEmptyTest(iters)
}

func benchmarkStackGrowth(b *testing.B, rec int) {
	b.RunParallel(func(pb *testing.PB) {
		for pb.Next() {
			stackGrowthRecursive(rec)
		}
	})
}

func BenchmarkStackGrowth(b *testing.B) {
	benchmarkStackGrowth(b, 10)
}

func BenchmarkStackGrowthDeep(b *testing.B) {
	benchmarkStackGrowth(b, 1024)
}

func BenchmarkCreateGoroutines(b *testing.B) {
	benchmarkCreateGoroutines(b, 1)
}

func BenchmarkCreateGoroutinesParallel(b *testing.B) {
	benchmarkCreateGoroutines(b, runtime.GOMAXPROCS(-1))
}

func benchmarkCreateGoroutines(b *testing.B, procs int) {
	c := make(chan bool)
	var f func(n int)
	f = func(n int) {
		if n == 0 {
			c <- true
			return
		}
		go f(n - 1)
	}
	for i := 0; i < procs; i++ {
		go f(b.N / procs)
	}
	for i := 0; i < procs; i++ {
		<-c
	}
}

func BenchmarkCreateGoroutinesCapture(b *testing.B) {
	b.ReportAllocs()
	for i := 0; i < b.N; i++ {
		const N = 4
		var wg sync.WaitGroup
		wg.Add(N)
		for i := 0; i < N; i++ {
			go func() {
				if i >= N {
					b.Logf("bad") // just to capture b
				}
				wg.Done()
			}()
		}
		wg.Wait()
	}
}

// warmupScheduler ensures the scheduler has at least targetThreadCount threads
// in its thread pool.
func warmupScheduler(targetThreadCount int) {
	var wg sync.WaitGroup
	var count int32
	for i := 0; i < targetThreadCount; i++ {
		wg.Add(1)
		go func() {
			atomic.AddInt32(&count, 1)
			for atomic.LoadInt32(&count) < int32(targetThreadCount) {
				// spin until all threads started
			}

			// spin a bit more to ensure they are all running on separate CPUs.
			doWork(time.Millisecond)
			wg.Done()
		}()
	}
	wg.Wait()
}

func doWork(dur time.Duration) {
	start := time.Now()
	for time.Since(start) < dur {
	}
}

// BenchmarkCreateGoroutinesSingle creates many goroutines, all from a single
// producer (the main benchmark goroutine).
//
// Compared to BenchmarkCreateGoroutines, this causes different behavior in the
// scheduler because Ms are much more likely to need to steal work from the
// main P rather than having work in the local run queue.
func BenchmarkCreateGoroutinesSingle(b *testing.B) {
	// Since we are interested in stealing behavior, warm the scheduler to
	// get all the Ps running first.
	warmupScheduler(runtime.GOMAXPROCS(0))
	b.ResetTimer()

	var wg sync.WaitGroup
	wg.Add(b.N)
	for i := 0; i < b.N; i++ {
		go func() {
			wg.Done()
		}()
	}
	wg.Wait()
}

func BenchmarkClosureCall(b *testing.B) {
	sum := 0
	off1 := 1
	for i := 0; i < b.N; i++ {
		off2 := 2
		func() {
			sum += i + off1 + off2
		}()
	}
	_ = sum
}

func benchmarkWakeupParallel(b *testing.B, spin func(time.Duration)) {
	if runtime.GOMAXPROCS(0) == 1 {
		b.Skip("skipping: GOMAXPROCS=1")
	}

	wakeDelay := 5 * time.Microsecond
	for _, delay := range []time.Duration{
		0,
		1 * time.Microsecond,
		2 * time.Microsecond,
		5 * time.Microsecond,
		10 * time.Microsecond,
		20 * time.Microsecond,
		50 * time.Microsecond,
		100 * time.Microsecond,
	} {
		b.Run(delay.String(), func(b *testing.B) {
			if b.N == 0 {
				return
			}
			// Start two goroutines, which alternate between being
			// sender and receiver in the following protocol:
			//
			// - The receiver spins for `delay` and then does a
			// blocking receive on a channel.
			//
			// - The sender spins for `delay+wakeDelay` and then
			// sends to the same channel. (The addition of
			// `wakeDelay` improves the probability that the
			// receiver will be blocking when the send occurs when
			// the goroutines execute in parallel.)
			//
			// In each iteration of the benchmark, each goroutine
			// acts once as sender and once as receiver, so each
			// goroutine spins for delay twice.
			//
			// BenchmarkWakeupParallel is used to estimate how
			// efficiently the scheduler parallelizes goroutines in
			// the presence of blocking:
			//
			// - If both goroutines are executed on the same core,
			// an increase in delay by N will increase the time per
			// iteration by 4*N, because all 4 delays are
			// serialized.
			//
			// - Otherwise, an increase in delay by N will increase
			// the time per iteration by 2*N, and the time per
			// iteration is 2 * (runtime overhead + chan
			// send/receive pair + delay + wakeDelay). This allows
			// the runtime overhead, including the time it takes
			// for the unblocked goroutine to be scheduled, to be
			// estimated.
			ping, pong := make(chan struct{}), make(chan struct{})
			start := make(chan struct{})
			done := make(chan struct{})
			go func() {
				<-start
				for i := 0; i < b.N; i++ {
					// sender
					spin(delay + wakeDelay)
					ping <- struct{}{}
					// receiver
					spin(delay)
					<-pong
				}
				done <- struct{}{}
			}()
			go func() {
				for i := 0; i < b.N; i++ {
					// receiver
					spin(delay)
					<-ping
					// sender
					spin(delay + wakeDelay)
					pong <- struct{}{}
				}
				done <- struct{}{}
			}()
			b.ResetTimer()
			start <- struct{}{}
			<-done
			<-done
		})
	}
}

func BenchmarkWakeupParallelSpinning(b *testing.B) {
	benchmarkWakeupParallel(b, func(d time.Duration) {
		end := time.Now().Add(d)
		for time.Now().Before(end) {
			// do nothing
		}
	})
}

// sysNanosleep is defined by OS-specific files (such as runtime_linux_test.go)
// to sleep for the given duration. If nil, dependent tests are skipped.
// The implementation should invoke a blocking system call and not
// call time.Sleep, which would deschedule the goroutine.
var sysNanosleep func(d time.Duration)

func BenchmarkWakeupParallelSyscall(b *testing.B) {
	if sysNanosleep == nil {
		b.Skipf("skipping on %v; sysNanosleep not defined", runtime.GOOS)
	}
	benchmarkWakeupParallel(b, func(d time.Duration) {
		sysNanosleep(d)
	})
}

type Matrix [][]float64

func BenchmarkMatmult(b *testing.B) {
	b.StopTimer()
	// matmult is O(N**3) but testing expects O(b.N),
	// so we need to take cube root of b.N
	n := int(math.Cbrt(float64(b.N))) + 1
	A := makeMatrix(n)
	B := makeMatrix(n)
	C := makeMatrix(n)
	b.StartTimer()
	matmult(nil, A, B, C, 0, n, 0, n, 0, n, 8)
}

func makeMatrix(n int) Matrix {
	m := make(Matrix, n)
	for i := 0; i < n; i++ {
		m[i] = make([]float64, n)
		for j := 0; j < n; j++ {
			m[i][j] = float64(i*n + j)
		}
	}
	return m
}

func matmult(done chan<- struct{}, A, B, C Matrix, i0, i1, j0, j1, k0, k1, threshold int) {
	di := i1 - i0
	dj := j1 - j0
	dk := k1 - k0
	if di >= dj && di >= dk && di >= threshold {
		// divide in two by y axis
		mi := i0 + di/2
		done1 := make(chan struct{}, 1)
		go matmult(done1, A, B, C, i0, mi, j0, j1, k0, k1, threshold)
		matmult(nil, A, B, C, mi, i1, j0, j1, k0, k1, threshold)
		<-done1
	} else if dj >= dk && dj >= threshold {
		// divide in two by x axis
		mj := j0 + dj/2
		done1 := make(chan struct{}, 1)
		go matmult(done1, A, B, C, i0, i1, j0, mj, k0, k1, threshold)
		matmult(nil, A, B, C, i0, i1, mj, j1, k0, k1, threshold)
		<-done1
	} else if dk >= threshold {
		// divide in two by "k" axis
		// deliberately not parallel because of data races
		mk := k0 + dk/2
		matmult(nil, A, B, C, i0, i1, j0, j1, k0, mk, threshold)
		matmult(nil, A, B, C, i0, i1, j0, j1, mk, k1, threshold)
	} else {
		// the matrices are small enough, compute directly
		for i := i0; i < i1; i++ {
			for j := j0; j < j1; j++ {
				for k := k0; k < k1; k++ {
					C[i][j] += A[i][k] * B[k][j]
				}
			}
		}
	}
	if done != nil {
		done <- struct{}{}
	}
}

func TestStealOrder(t *testing.T) {
	runtime.RunStealOrderTest()
}

func TestLockOSThreadNesting(t *testing.T) {
	if runtime.GOARCH == "wasm" {
		t.Skip("no threads on wasm yet")
	}

	go func() {
		e, i := runtime.LockOSCounts()
		if e != 0 || i != 0 {
			t.Errorf("want locked counts 0, 0; got %d, %d", e, i)
			return
		}
		runtime.LockOSThread()
		runtime.LockOSThread()
		runtime.UnlockOSThread()
		e, i = runtime.LockOSCounts()
		if e != 1 || i != 0 {
			t.Errorf("want locked counts 1, 0; got %d, %d", e, i)
			return
		}
		runtime.UnlockOSThread()
		e, i = runtime.LockOSCounts()
		if e != 0 || i != 0 {
			t.Errorf("want locked counts 0, 0; got %d, %d", e, i)
			return
		}
	}()
}

func TestLockOSThreadExit(t *testing.T) {
	testLockOSThreadExit(t, "testprog")
}

func testLockOSThreadExit(t *testing.T, prog string) {
	output := runTestProg(t, prog, "LockOSThreadMain", "GOMAXPROCS=1")
	want := "OK\n"
	if output != want {
		t.Errorf("want %q, got %q", want, output)
	}

	output = runTestProg(t, prog, "LockOSThreadAlt")
	if output != want {
		t.Errorf("want %q, got %q", want, output)
	}
}

func TestLockOSThreadAvoidsStatePropagation(t *testing.T) {
	want := "OK\n"
	skip := "unshare not permitted\n"
	output := runTestProg(t, "testprog", "LockOSThreadAvoidsStatePropagation", "GOMAXPROCS=1")
	if output == skip {
		t.Skip("unshare syscall not permitted on this system")
	} else if output != want {
		t.Errorf("want %q, got %q", want, output)
	}
}

func TestLockOSThreadTemplateThreadRace(t *testing.T) {
	testenv.MustHaveGoRun(t)

	exe, err := buildTestProg(t, "testprog")
	if err != nil {
		t.Fatal(err)
	}

	iterations := 100
	if testing.Short() {
		// Reduce run time to ~100ms, with much lower probability of
		// catching issues.
		iterations = 5
	}
	for i := 0; i < iterations; i++ {
		want := "OK\n"
		output := runBuiltTestProg(t, exe, "LockOSThreadTemplateThreadRace")
		if output != want {
			t.Fatalf("run %d: want %q, got %q", i, want, output)
		}
	}
}

func TestLockOSThreadVgetrandom(t *testing.T) {
	if runtime.GOOS != "linux" {
		t.Skipf("vgetrandom only relevant on Linux")
	}
	output := runTestProg(t, "testprog", "LockOSThreadVgetrandom")
	want := "OK\n"
	if output != want {
		t.Errorf("want %q, got %q", want, output)
	}
}

// fakeSyscall emulates a system call.
//
//go:nosplit
func fakeSyscall(duration time.Duration) {
	runtime.Entersyscall()
	for start := runtime.Nanotime(); runtime.Nanotime()-start < int64(duration); {
	}
	runtime.Exitsyscall()
}

// Check that a goroutine will be preempted if it is calling short system calls.
func testPreemptionAfterSyscall(t *testing.T, syscallDuration time.Duration) {
	if runtime.GOARCH == "wasm" {
		t.Skip("no preemption on wasm yet")
	}

	defer runtime.GOMAXPROCS(runtime.GOMAXPROCS(2))

	iterations := 10
	if testing.Short() {
		iterations = 1
	}
	const (
		maxDuration = 5 * time.Second
		nroutines   = 8
	)

	for i := 0; i < iterations; i++ {
		c := make(chan bool, nroutines)
		stop := uint32(0)

		start := time.Now()
		for g := 0; g < nroutines; g++ {
			go func(stop *uint32) {
				c <- true
				for atomic.LoadUint32(stop) == 0 {
					fakeSyscall(syscallDuration)
				}
				c <- true
			}(&stop)
		}
		// wait until all goroutines have started.
		for g := 0; g < nroutines; g++ {
			<-c
		}
		atomic.StoreUint32(&stop, 1)
		// wait until all goroutines have finished.
		for g := 0; g < nroutines; g++ {
			<-c
		}
		duration := time.Since(start)

		if duration > maxDuration {
			t.Errorf("timeout exceeded: %v (%v)", duration, maxDuration)
		}
	}
}

func TestPreemptionAfterSyscall(t *testing.T) {
	if runtime.GOOS == "plan9" {
		testenv.SkipFlaky(t, 41015)
	}

	for _, i := range []time.Duration{10, 100, 1000} {
		d := i * time.Microsecond
		t.Run(fmt.Sprint(d), func(t *testing.T) {
			testPreemptionAfterSyscall(t, d)
		})
	}
}

func TestGetgThreadSwitch(t *testing.T) {
	runtime.RunGetgThreadSwitchTest()
}

// TestNetpollBreak tests that netpollBreak can break a netpoll.
// This test is not particularly safe since the call to netpoll
// will pick up any stray files that are ready, but it should work
// OK as long it is not run in parallel.
func TestNetpollBreak(t *testing.T) {
	if runtime.GOMAXPROCS(0) == 1 {
		t.Skip("skipping: GOMAXPROCS=1")
	}

	// Make sure that netpoll is initialized.
	runtime.NetpollGenericInit()

	start := time.Now()
	c := make(chan bool, 2)
	go func() {
		c <- true
		runtime.Netpoll(10 * time.Second.Nanoseconds())
		c <- true
	}()
	<-c
	// Loop because the break might get eaten by the scheduler.
	// Break twice to break both the netpoll we started and the
	// scheduler netpoll.
loop:
	for {
		runtime.Usleep(100)
		runtime.NetpollBreak()
		runtime.NetpollBreak()
		select {
		case <-c:
			break loop
		default:
		}
	}
	if dur := time.Since(start); dur > 5*time.Second {
		t.Errorf("netpollBreak did not interrupt netpoll: slept for: %v", dur)
	}
}

// TestBigGOMAXPROCS tests that setting GOMAXPROCS to a large value
// doesn't cause a crash at startup. See issue 38474.
func TestBigGOMAXPROCS(t *testing.T) {
	t.Parallel()
	output := runTestProg(t, "testprog", "NonexistentTest", "GOMAXPROCS=1024")
	// Ignore error conditions on small machines.
	for _, errstr := range []string{
		"failed to create new OS thread",
		"cannot allocate memory",
	} {
		if strings.Contains(output, errstr) {
			t.Skipf("failed to create 1024 threads")
		}
	}
	if !strings.Contains(output, "unknown function: NonexistentTest") {
		t.Errorf("output:\n%s\nwanted:\nunknown function: NonexistentTest", output)
	}
}

type goroutineState struct {
	G trace.GoID      // This goroutine.
	P trace.ProcID    // Most recent P this goroutine ran on.
	M trace.ThreadID  // Most recent M this goroutine ran on.
}

func newGoroutineState(g trace.GoID) *goroutineState {
	return &goroutineState{
		G: g,
		P: trace.NoProc,
		M: trace.NoThread,
	}
}

// TestTraceSTW verifies that goroutines continue running on the same M and P
// after a STW.
func TestTraceSTW(t *testing.T) {
	// Across STW, the runtime attempts to keep goroutines running on the
	// same P and the P running on the same M. It does this by keeping
	// goroutines in the P's local runq, and remembering which M the P ran
	// on before STW and preferring that M when restarting.
	//
	// This test verifies that affinity by analyzing a trace of testprog
	// TraceSTW.
	//
	// The affinity across STW is best-effort, so have to allow some
	// failure rate, thus we test many times and ensure the error rate is
	// low.
	//
	// The expected affinity can fail for a variety of reasons. The most
	// obvious is that while procresize assigns Ps back to their original
	// M, startTheWorldWithSema calls wakep to start a spinning M. The
	// spinning M may steal a goroutine from another P if that P is too
	// slow to start.

	if testing.Short() {
		t.Skip("skipping in -short mode")
	}

	if runtime.NumCPU() < 4 {
		t.Skip("This test sets GOMAXPROCS=4 and wants to avoid thread descheduling as much as possible. Skip on machines with less than 4 CPUs")
	}

	const runs = 50

	var errors int
	for i := range runs {
		err := runTestTracesSTW(t, i)
		if err != nil {
			t.Logf("Run %d failed: %v", i, err)
			errors++
		}
	}

	pct := float64(errors)/float64(runs)
	t.Logf("Errors: %d/%d = %f%%", errors, runs, 100*pct)
	if pct > 0.25 {
		t.Errorf("Error rate too high")
	}
}

func runTestTracesSTW(t *testing.T, run int) (err error) {
	t.Logf("Run %d", run)

	// By default, TSAN sleeps for 1s at exit to allow background
	// goroutines to race. This slows down execution for this test far too
	// much, since we are running 50 iterations, so disable the sleep.
	//
	// Outside of race mode, GORACE does nothing.
	buf := []byte(runTestProg(t, "testprog", "TraceSTW", "GORACE=atexit_sleep_ms=0"))

	// We locally "fail" the run (return an error) if the trace exhibits
	// unwanted scheduling. i.e., the target goroutines did not remain on
	// the same P/M.
	//
	// We fail the entire test (t.Fatal) for other cases that should never
	// occur, such as a trace parse error.
	defer func() {
		if err != nil || t.Failed() {
			testtrace.Dump(t, fmt.Sprintf("TestTraceSTW-run%d", run), []byte(buf), false)
		}
	}()

	br, err := trace.NewReader(bytes.NewReader(buf))
	if err != nil {
		t.Fatalf("NewReader got err %v want nil", err)
	}

	var targetGoroutines []*goroutineState
	findGoroutine := func(goid trace.GoID) *goroutineState {
		for _, gs := range targetGoroutines {
			if gs.G == goid {
				return gs
			}
		}
		return nil
	}
	findProc := func(pid trace.ProcID) *goroutineState {
		for _, gs := range targetGoroutines {
			if gs.P == pid {
				return gs
			}
		}
		return nil
	}

	// 1. Find the goroutine IDs for the target goroutines. This will be in
	// the StateTransition from NotExist.
	//
	// 2. Once found, track which M and P the target goroutines run on until...
	//
	// 3. Look for the "TraceSTW" "start" log message, where we commit the
	// target goroutines' "before" M and P.
	//
	// N.B. We must do (1) and (2) together because the first target
	// goroutine may start running before the second is created.
findStart:
	for {
		ev, err := br.ReadEvent()
		if err == io.EOF {
			// Reached the end of the trace without finding case (3).
			t.Fatalf("Trace missing start log message")
		}
		if err != nil {
			t.Fatalf("ReadEvent got err %v want nil", err)
		}
		t.Logf("Event: %s", ev.String())

		switch ev.Kind() {
		case trace.EventStateTransition:
			st := ev.StateTransition()
			if st.Resource.Kind != trace.ResourceGoroutine {
				continue
			}

			goid := st.Resource.Goroutine()
			from, to := st.Goroutine()

			// Potentially case (1): Goroutine creation.
			if from == trace.GoNotExist {
				for sf := range st.Stack.Frames() {
					if sf.Func == "main.traceSTWTarget" {
						targetGoroutines = append(targetGoroutines, newGoroutineState(goid))
						t.Logf("Identified target goroutine id %d", goid)
					}

					// Always break, the goroutine entrypoint is always the
					// first frame.
					break
				}
			}

			// Potentially case (2): Goroutine running.
			if to == trace.GoRunning {
				gs := findGoroutine(goid)
				if gs == nil {
					continue
				}
				gs.P = ev.Proc()
				gs.M = ev.Thread()
				t.Logf("G %d running on P %d M %d", gs.G, gs.P, gs.M)
			}
		case trace.EventLog:
			// Potentially case (3): Start log event.
			log := ev.Log()
			if log.Category != "TraceSTW" {
				continue
			}
			if log.Message != "start" {
				t.Fatalf("Log message got %s want start", log.Message)
			}

			// Found start point, move on to next stage.
			t.Logf("Found start message")
			break findStart
		}
	}

	t.Log("Target goroutines:")
	for _, gs := range targetGoroutines {
		t.Logf("%+v", gs)
	}

	if len(targetGoroutines) != 2 {
		t.Fatalf("len(targetGoroutines) got %d want 2", len(targetGoroutines))
	}

	for _, gs := range targetGoroutines {
		if gs.P == trace.NoProc {
			t.Fatalf("Goroutine %+v not running on a P", gs)
		}
		if gs.M == trace.NoThread {
			t.Fatalf("Goroutine %+v not running on an M", gs)
		}
	}

	// The test continues until we see the "end" log message.
	//
	// What we want to observe is that the target goroutines run only on
	// the original P and M.
	//
	// They will be stopped by STW [1], but should resume on the original P
	// and M.
	//
	// However, this is best effort. For example, startTheWorld wakep's a
	// spinning M. If the original M is slow to restart (e.g., due to poor
	// kernel scheduling), the spinning M may legally steal the goroutine
	// and run it instead.
	//
	// In practice, we see this occur frequently on builders, likely
	// because they are overcommitted on CPU. Thus, we instead check
	// slightly more constrained properties:
	// - The original P must run on the original M (if it runs at all).
	// - The original P must run the original G before anything else,
	//   unless that G has already run elsewhere.
	//
	// This allows a spinning M to steal the G from a slow-to-start M, but
	// does not allow the original P to just flat out run something
	// completely different from expected.
	//
	// Note this is still somewhat racy: the spinning M may steal the
	// target G, but before it marks the target G as running, the original
	// P runs an alternative G. This test will fail that case, even though
	// it is legitimate. We allow that failure because such a race should
	// be very rare, particularly because the test process usually has no
	// other runnable goroutines.
	//
	// [1] This is slightly fragile because there is a small window between
	// the "start" log and actual STW during which the target goroutines
	// could legitimately migrate.
	var stwSeen bool
	var pRunning []trace.ProcID
	var gRunning []trace.GoID
findEnd:
	for {
		ev, err := br.ReadEvent()
		if err == io.EOF {
			break
		}
		if err != nil {
			t.Fatalf("ReadEvent got err %v want nil", err)
		}
		t.Logf("Event: %s", ev.String())

		switch ev.Kind() {
		case trace.EventStateTransition:
			st := ev.StateTransition()
			switch st.Resource.Kind {
			case trace.ResourceProc:
				p := st.Resource.Proc()
				_, to := st.Proc()

				// Proc running. Ensure it didn't migrate.
				if to == trace.ProcRunning {
					gs := findProc(p)
					if gs == nil {
						continue
					}

					if slices.Contains(pRunning, p) {
						// Only check the first
						// transition to running.
						// Afterwards it is free to
						// migrate anywhere.
						continue
					}
					pRunning = append(pRunning, p)

					m := ev.Thread()
					if m != gs.M {
						t.Logf("Proc %d running on M %d want M %d", p, m, gs.M)
						return fmt.Errorf("P did not remain on M")
					}
				}
			case trace.ResourceGoroutine:
				goid := st.Resource.Goroutine()
				_, to := st.Goroutine()

				// Goroutine running. Ensure it didn't migrate.
				if to == trace.GoRunning {
					p := ev.Proc()
					m := ev.Thread()

					gs := findGoroutine(goid)
					if gs == nil {
						// This isn't a target
						// goroutine. Is it a target P?
						// That shouldn't run anything
						// other than the target G.
						gs = findProc(p)
						if gs == nil {
							continue
						}

						if slices.Contains(gRunning, gs.G) {
							// This P's target G ran elsewhere. This probably
							// means that this P was slow to start, so
							// another P stole it. That isn't ideal, but
							// we'll allow it.
							continue
						}

						t.Logf("Goroutine %d running on P %d M %d want this P to run G %d", goid, p, m, gs.G)
						return fmt.Errorf("P ran incorrect goroutine")
					}

					if !slices.Contains(gRunning, goid) {
						gRunning = append(gRunning, goid)
					}

					if p != gs.P || m != gs.M {
						t.Logf("Goroutine %d running on P %d M %d want P %d M %d", goid, p, m, gs.P, gs.M)
						// We don't want this to occur,
						// but allow it for cases of
						// bad kernel scheduling. See
						// "The test continues" comment
						// above.
					}
				}
			}
		case trace.EventLog:
			// Potentially end log event.
			log := ev.Log()
			if log.Category != "TraceSTW" {
				continue
			}
			if log.Message != "end" {
				t.Fatalf("Log message got %s want end", log.Message)
			}

			// Found end point.
			t.Logf("Found end message")
			break findEnd
		case trace.EventRangeBegin:
			r := ev.Range()
			if r.Name == "stop-the-world (read mem stats)" {
				// Note when we see the STW begin. This is not
				// load bearing; it's purpose is simply to fail
				// the test if we manage to remove the STW from
				// ReadMemStat, so we remember to change this
				// test to add some new source of STW.
				stwSeen = true
			}
		}
	}

	if !stwSeen {
		t.Fatal("No STW in the test trace")
	}

	return nil
}

func TestMexitSTW(t *testing.T) {
	got := runTestProg(t, "testprog", "mexitSTW")
	want := "OK\n"
	if got != want {
		t.Fatalf("expected %q, but got:\n%s", want, got)
	}
}
