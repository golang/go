// Copyright 2011 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

package runtime_test

import (
	"math"
	"net"
	"runtime"
	"runtime/debug"
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
			if err != nil {
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

		nstk := strings.Count(string(buf), "goroutine ")
		if n == nstk {
			break
		}
		if i >= 10 {
			t.Fatalf("NumGoroutine=%d, but found %d goroutines in stack dump: %s", n, nstk, buf)
		}
	}
}

func TestPingPongHog(t *testing.T) {
	if testing.Short() {
		t.Skip("skipping in -short mode")
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
	// 5, which indicates that both pairs of goroutines handed off
	// the P within a time-slice to their buddy. We can use a
	// fairly large factor here to make this robust: if the
	// scheduler isn't working right, the gap should be ~1000X.
	const factor = 5
	if hogCount > lightCount*factor || lightCount > hogCount*factor {
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

func stackGrowthRecursive(i int) {
	var pad [128]uint64
	if i != 0 && pad[0] == 0 {
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
			i := i
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
