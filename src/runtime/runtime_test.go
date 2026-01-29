// Copyright 2012 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

package runtime_test

import (
	"flag"
	"internal/asan"
	"internal/cpu"
	"internal/msan"
	"internal/race"
	"internal/runtime/atomic"
	"internal/testenv"
	"io"
	"math/bits"
	. "runtime"
	"runtime/debug"
	"slices"
	"strings"
	"sync"
	"testing"
	"time"
	"unsafe"
)

// flagQuick is set by the -quick option to skip some relatively slow tests.
// This is used by the cmd/dist test runtime:cpu124.
// The cmd/dist test passes both -test.short and -quick;
// there are tests that only check testing.Short, and those tests will
// not be skipped if only -quick is used.
var flagQuick = flag.Bool("quick", false, "skip slow tests, for cmd/dist test runtime:cpu124")

func init() {
	// We're testing the runtime, so make tracebacks show things
	// in the runtime. This only raises the level, so it won't
	// override GOTRACEBACK=crash from the user.
	SetTracebackEnv("system")
}

var errf error

func errfn() error {
	return errf
}

func errfn1() error {
	return io.EOF
}

func BenchmarkIfaceCmp100(b *testing.B) {
	for i := 0; i < b.N; i++ {
		for j := 0; j < 100; j++ {
			if errfn() == io.EOF {
				b.Fatal("bad comparison")
			}
		}
	}
}

func BenchmarkIfaceCmpNil100(b *testing.B) {
	for i := 0; i < b.N; i++ {
		for j := 0; j < 100; j++ {
			if errfn1() == nil {
				b.Fatal("bad comparison")
			}
		}
	}
}

var efaceCmp1 any
var efaceCmp2 any

func BenchmarkEfaceCmpDiff(b *testing.B) {
	x := 5
	efaceCmp1 = &x
	y := 6
	efaceCmp2 = &y
	for i := 0; i < b.N; i++ {
		for j := 0; j < 100; j++ {
			if efaceCmp1 == efaceCmp2 {
				b.Fatal("bad comparison")
			}
		}
	}
}

func BenchmarkEfaceCmpDiffIndirect(b *testing.B) {
	efaceCmp1 = [2]int{1, 2}
	efaceCmp2 = [2]int{1, 2}
	for i := 0; i < b.N; i++ {
		for j := 0; j < 100; j++ {
			if efaceCmp1 != efaceCmp2 {
				b.Fatal("bad comparison")
			}
		}
	}
}

func BenchmarkDefer(b *testing.B) {
	for i := 0; i < b.N; i++ {
		defer1()
	}
}

func defer1() {
	defer func(x, y, z int) {
		if recover() != nil || x != 1 || y != 2 || z != 3 {
			panic("bad recover")
		}
	}(1, 2, 3)
}

func BenchmarkDefer10(b *testing.B) {
	for i := 0; i < b.N/10; i++ {
		defer2()
	}
}

func defer2() {
	for i := 0; i < 10; i++ {
		defer func(x, y, z int) {
			if recover() != nil || x != 1 || y != 2 || z != 3 {
				panic("bad recover")
			}
		}(1, 2, 3)
	}
}

func BenchmarkDeferMany(b *testing.B) {
	for i := 0; i < b.N; i++ {
		defer func(x, y, z int) {
			if recover() != nil || x != 1 || y != 2 || z != 3 {
				panic("bad recover")
			}
		}(1, 2, 3)
	}
}

func BenchmarkPanicRecover(b *testing.B) {
	for i := 0; i < b.N; i++ {
		defer3()
	}
}

func defer3() {
	defer func(x, y, z int) {
		if recover() == nil {
			panic("failed recover")
		}
	}(1, 2, 3)
	panic("hi")
}

// golang.org/issue/7063
func TestStopCPUProfilingWithProfilerOff(t *testing.T) {
	SetCPUProfileRate(0)
}

// Addresses to test for faulting behavior.
// This is less a test of SetPanicOnFault and more a check that
// the operating system and the runtime can process these faults
// correctly. That is, we're indirectly testing that without SetPanicOnFault
// these would manage to turn into ordinary crashes.
// Note that these are truncated on 32-bit systems, so the bottom 32 bits
// of the larger addresses must themselves be invalid addresses.
// We might get unlucky and the OS might have mapped one of these
// addresses, but probably not: they're all in the first page, very high
// addresses that normally an OS would reserve for itself, or malformed
// addresses. Even so, we might have to remove one or two on different
// systems. We will see.

var faultAddrs = []uint64{
	// low addresses
	0,
	1,
	0xfff,
	// high (kernel) addresses
	// or else malformed.
	0xffffffffffffffff,
	0xfffffffffffff001,
	0xffffffffffff0001,
	0xfffffffffff00001,
	0xffffffffff000001,
	0xfffffffff0000001,
	0xffffffff00000001,
	0xfffffff000000001,
	0xffffff0000000001,
	0xfffff00000000001,
	0xffff000000000001,
	0xfff0000000000001,
	0xff00000000000001,
	0xf000000000000001,
	0x8000000000000001,
}

func TestSetPanicOnFault(t *testing.T) {
	old := debug.SetPanicOnFault(true)
	defer debug.SetPanicOnFault(old)

	nfault := 0
	for _, addr := range faultAddrs {
		testSetPanicOnFault(t, uintptr(addr), &nfault)
	}
	if nfault == 0 {
		t.Fatalf("none of the addresses faulted")
	}
}

// testSetPanicOnFault tests one potentially faulting address.
// It deliberately constructs and uses an invalid pointer,
// so mark it as nocheckptr.
//
//go:nocheckptr
func testSetPanicOnFault(t *testing.T, addr uintptr, nfault *int) {
	if GOOS == "js" || GOOS == "wasip1" {
		t.Skip(GOOS + " does not support catching faults")
	}

	defer func() {
		if err := recover(); err != nil {
			*nfault++
		}
	}()

	// The read should fault, except that sometimes we hit
	// addresses that have had C or kernel pages mapped there
	// readable by user code. So just log the content.
	// If no addresses fault, we'll fail the test.
	v := *(*byte)(unsafe.Pointer(addr))
	t.Logf("addr %#x: %#x\n", addr, v)
}

func eqstring_generic(s1, s2 string) bool {
	if len(s1) != len(s2) {
		return false
	}
	// optimization in assembly versions:
	// if s1.str == s2.str { return true }
	for i := 0; i < len(s1); i++ {
		if s1[i] != s2[i] {
			return false
		}
	}
	return true
}

func TestEqString(t *testing.T) {
	// This isn't really an exhaustive test of == on strings, it's
	// just a convenient way of documenting (via eqstring_generic)
	// what == does.
	s := []string{
		"",
		"a",
		"c",
		"aaa",
		"ccc",
		"cccc"[:3], // same contents, different string
		"1234567890",
	}
	for _, s1 := range s {
		for _, s2 := range s {
			x := s1 == s2
			y := eqstring_generic(s1, s2)
			if x != y {
				t.Errorf(`("%s" == "%s") = %t, want %t`, s1, s2, x, y)
			}
		}
	}
}

func TestTrailingZero(t *testing.T) {
	// make sure we add padding for structs with trailing zero-sized fields
	type T1 struct {
		n int32
		z [0]byte
	}
	if unsafe.Sizeof(T1{}) != 8 {
		t.Errorf("sizeof(%#v)==%d, want 8", T1{}, unsafe.Sizeof(T1{}))
	}
	type T2 struct {
		n int64
		z struct{}
	}
	if unsafe.Sizeof(T2{}) != 8+unsafe.Sizeof(uintptr(0)) {
		t.Errorf("sizeof(%#v)==%d, want %d", T2{}, unsafe.Sizeof(T2{}), 8+unsafe.Sizeof(uintptr(0)))
	}
	type T3 struct {
		n byte
		z [4]struct{}
	}
	if unsafe.Sizeof(T3{}) != 2 {
		t.Errorf("sizeof(%#v)==%d, want 2", T3{}, unsafe.Sizeof(T3{}))
	}
	// make sure padding can double for both zerosize and alignment
	type T4 struct {
		a int32
		b int16
		c int8
		z struct{}
	}
	if unsafe.Sizeof(T4{}) != 8 {
		t.Errorf("sizeof(%#v)==%d, want 8", T4{}, unsafe.Sizeof(T4{}))
	}
	// make sure we don't pad a zero-sized thing
	type T5 struct {
	}
	if unsafe.Sizeof(T5{}) != 0 {
		t.Errorf("sizeof(%#v)==%d, want 0", T5{}, unsafe.Sizeof(T5{}))
	}
}

func TestAppendGrowthHeap(t *testing.T) {
	var x []int64
	check := func(want int) {
		if cap(x) != want {
			t.Errorf("len=%d, cap=%d, want cap=%d", len(x), cap(x), want)
		}
	}

	check(0)
	want := 1
	for i := 1; i <= 100; i++ {
		x = append(x, 1)
		check(want)
		if i&(i-1) == 0 {
			want = 2 * i
		}
	}
	Escape(&x[0]) // suppress stack-allocated backing store
}

func TestAppendGrowthStack(t *testing.T) {
	if race.Enabled || asan.Enabled || msan.Enabled {
		t.Skip("instrumentation breaks this optimization")
	}
	var x []int64
	check := func(want int) {
		if cap(x) != want {
			t.Errorf("len=%d, cap=%d, want cap=%d", len(x), cap(x), want)
		}
	}

	check(0)
	want := 32 / 8 // 32 is the default for cmd/compile/internal/base.DebugFlags.VariableMakeThreshold
	if testenv.OptimizationOff() {
		want = 1
	}
	for i := 1; i <= 100; i++ {
		x = append(x, 1)
		check(want)
		if i&(i-1) == 0 {
			want = max(want, 2*i)
		}
	}
}

var One = []int64{1}

func TestAppendSliceGrowth(t *testing.T) {
	var x []int64
	check := func(want int) {
		if cap(x) != want {
			t.Errorf("len=%d, cap=%d, want cap=%d", len(x), cap(x), want)
		}
	}

	check(0)
	want := 1
	for i := 1; i <= 100; i++ {
		x = append(x, One...)
		check(want)
		if i&(i-1) == 0 {
			want = 2 * i
		}
	}
}

func TestGoroutineProfileTrivial(t *testing.T) {
	// Calling GoroutineProfile twice in a row should find the same number of goroutines,
	// but it's possible there are goroutines just about to exit, so we might end up
	// with fewer in the second call. Try a few times; it should converge once those
	// zombies are gone.
	for i := 0; ; i++ {
		n1, ok := GoroutineProfile(nil) // should fail, there's at least 1 goroutine
		if n1 < 1 || ok {
			t.Fatalf("GoroutineProfile(nil) = %d, %v, want >0, false", n1, ok)
		}
		n2, ok := GoroutineProfile(make([]StackRecord, n1))
		if n2 == n1 && ok {
			break
		}
		t.Logf("GoroutineProfile(%d) = %d, %v, want %d, true", n1, n2, ok, n1)
		if i >= 10 {
			t.Fatalf("GoroutineProfile not converging")
		}
	}
}

func BenchmarkGoroutineProfile(b *testing.B) {
	run := func(fn func() bool) func(b *testing.B) {
		runOne := func(b *testing.B) {
			latencies := make([]time.Duration, 0, b.N)

			b.ResetTimer()
			for i := 0; i < b.N; i++ {
				start := time.Now()
				ok := fn()
				if !ok {
					b.Fatal("goroutine profile failed")
				}
				latencies = append(latencies, time.Since(start))
			}
			b.StopTimer()

			// Sort latencies then report percentiles.
			slices.Sort(latencies)
			b.ReportMetric(float64(latencies[len(latencies)*50/100]), "p50-ns")
			b.ReportMetric(float64(latencies[len(latencies)*90/100]), "p90-ns")
			b.ReportMetric(float64(latencies[len(latencies)*99/100]), "p99-ns")
		}
		return func(b *testing.B) {
			b.Run("idle", runOne)

			b.Run("loaded", func(b *testing.B) {
				stop := applyGCLoad(b)
				runOne(b)
				// Make sure to stop the timer before we wait! The load created above
				// is very heavy-weight and not easy to stop, so we could end up
				// confusing the benchmarking framework for small b.N.
				b.StopTimer()
				stop()
			})
		}
	}

	// Measure the cost of counting goroutines
	b.Run("small-nil", run(func() bool {
		GoroutineProfile(nil)
		return true
	}))

	// Measure the cost with a small set of goroutines
	n := NumGoroutine()
	p := make([]StackRecord, 2*n+2*GOMAXPROCS(0))
	b.Run("small", run(func() bool {
		_, ok := GoroutineProfile(p)
		return ok
	}))

	// Measure the cost with a large set of goroutines
	ch := make(chan int)
	var ready, done sync.WaitGroup
	for i := 0; i < 5000; i++ {
		ready.Add(1)
		done.Add(1)
		go func() { ready.Done(); <-ch; done.Done() }()
	}
	ready.Wait()

	// Count goroutines with a large allgs list
	b.Run("large-nil", run(func() bool {
		GoroutineProfile(nil)
		return true
	}))

	n = NumGoroutine()
	p = make([]StackRecord, 2*n+2*GOMAXPROCS(0))
	b.Run("large", run(func() bool {
		_, ok := GoroutineProfile(p)
		return ok
	}))

	close(ch)
	done.Wait()

	// Count goroutines with a large (but unused) allgs list
	b.Run("sparse-nil", run(func() bool {
		GoroutineProfile(nil)
		return true
	}))

	// Measure the cost of a large (but unused) allgs list
	n = NumGoroutine()
	p = make([]StackRecord, 2*n+2*GOMAXPROCS(0))
	b.Run("sparse", run(func() bool {
		_, ok := GoroutineProfile(p)
		return ok
	}))
}

func TestVersion(t *testing.T) {
	// Test that version does not contain \r or \n.
	vers := Version()
	if strings.Contains(vers, "\r") || strings.Contains(vers, "\n") {
		t.Fatalf("cr/nl in version: %q", vers)
	}
}

func BenchmarkProcYield(b *testing.B) {
	benchN := func(n uint32) func(*testing.B) {
		return func(b *testing.B) {
			for i := 0; i < b.N; i++ {
				ProcYield(n)
			}
		}
	}

	b.Run("1", benchN(1))
	b.Run("10", benchN(10))
	b.Run("30", benchN(30)) // active_spin_cnt in lock_sema.go and lock_futex.go
	b.Run("100", benchN(100))
	b.Run("1000", benchN(1000))
}

func BenchmarkOSYield(b *testing.B) {
	for i := 0; i < b.N; i++ {
		OSYield()
	}
}

func BenchmarkMutexContention(b *testing.B) {
	// Measure throughput of a single mutex with all threads contending
	//
	// Share a single counter across all threads. Progress from any thread is
	// progress for the benchmark as a whole. We don't measure or give points
	// for fairness here, arbitrary delay to any given thread's progress is
	// invisible and allowed.
	//
	// The cache line that holds the count value will need to move between
	// processors, but not as often as the cache line that holds the mutex. The
	// mutex protects access to the count value, which limits contention on that
	// cache line. This is a simple design, but it helps to make the behavior of
	// the benchmark clear. Most real uses of mutex will protect some number of
	// cache lines anyway.

	var state struct {
		_     cpu.CacheLinePad
		lock  Mutex
		_     cpu.CacheLinePad
		count atomic.Int64
		_     cpu.CacheLinePad
	}

	procs := GOMAXPROCS(0)
	var wg sync.WaitGroup
	for range procs {
		wg.Add(1)
		go func() {
			defer wg.Done()
			for {
				Lock(&state.lock)
				ours := state.count.Add(1)
				Unlock(&state.lock)
				if ours >= int64(b.N) {
					return
				}
			}
		}()
	}
	wg.Wait()
}

func BenchmarkMutexCapture(b *testing.B) {

	// Measure mutex fairness.
	//
	// Have several threads contend for a single mutex value. Measure how
	// effectively a single thread is able to capture the lock and report the
	// duration of those "streak" events. Measure how long other individual
	// threads need to wait between their turns with the lock. Report the
	// duration of those "starve" events.
	//
	// Report in terms of wall clock time (assuming a constant time per
	// lock/unlock pair) rather than number of locks/unlocks. This keeps
	// timekeeping overhead out of the critical path, and avoids giving an
	// advantage to lock/unlock implementations that take less time per
	// operation.

	var state struct {
		_     cpu.CacheLinePad
		lock  Mutex
		_     cpu.CacheLinePad
		count atomic.Int64
		_     cpu.CacheLinePad
	}

	procs := GOMAXPROCS(0)
	var wg sync.WaitGroup
	histograms := make(chan [2][65]int)
	for range procs {
		wg.Add(1)
		go func() {
			var (
				prev      int64
				streak    int64
				histogram [2][65]int
			)
			for {
				Lock(&state.lock)
				ours := state.count.Add(1)
				Unlock(&state.lock)
				delta := ours - prev - 1
				prev = ours
				if delta == 0 {
					streak++
				} else {
					histogram[0][bits.LeadingZeros64(uint64(streak))]++
					histogram[1][bits.LeadingZeros64(uint64(delta))]++
					streak = 1
				}
				if ours >= int64(b.N) {
					wg.Done()
					if delta == 0 {
						histogram[0][bits.LeadingZeros64(uint64(streak))]++
						histogram[1][bits.LeadingZeros64(uint64(delta))]++
					}
					histograms <- histogram
					return
				}
			}
		}()
	}

	wg.Wait()
	b.StopTimer()

	var histogram [2][65]int
	for range procs {
		h := <-histograms
		for i := range h {
			for j := range h[i] {
				histogram[i][j] += h[i][j]
			}
		}
	}

	percentile := func(h [65]int, p float64) int {
		sum := 0
		for i, v := range h {
			bound := uint64(1<<63) >> i
			sum += int(bound) * v
		}

		// Imagine that the longest streak / starvation events were instead half
		// as long but twice in number. (Note that we've pre-multiplied by the
		// [lower] "bound" value.) Continue those splits until we meet the
		// percentile target.
		part := 0
		for i, v := range h {
			bound := uint64(1<<63) >> i
			part += int(bound) * v
			// have we trimmed off enough at the head to dip below the percentile goal
			if float64(sum-part) < float64(sum)*p {
				return int(bound)
			}
		}

		return 0
	}

	perOp := float64(b.Elapsed().Nanoseconds()) / float64(b.N)
	b.ReportMetric(perOp*float64(percentile(histogram[0], 1.0)), "ns/streak-p100")
	b.ReportMetric(perOp*float64(percentile(histogram[0], 0.9)), "ns/streak-p90")
	b.ReportMetric(perOp*float64(percentile(histogram[1], 1.0)), "ns/starve-p100")
	b.ReportMetric(perOp*float64(percentile(histogram[1], 0.9)), "ns/starve-p90")
}

func BenchmarkMutexHandoff(b *testing.B) {
	testcase := func(delay func(l *Mutex)) func(b *testing.B) {
		return func(b *testing.B) {
			if workers := 2; GOMAXPROCS(0) < workers {
				b.Skipf("requires GOMAXPROCS >= %d", workers)
			}

			// Measure latency of mutex handoff between threads.
			//
			// Hand off a runtime.mutex between two threads, one running a
			// "coordinator" goroutine and the other running a "worker"
			// goroutine. We don't override the runtime's typical
			// goroutine/thread mapping behavior.
			//
			// Measure the latency, starting when the coordinator enters a call
			// to runtime.unlock and ending when the worker's call to
			// runtime.lock returns. The benchmark can specify a "delay"
			// function to simulate the length of the mutex-holder's critical
			// section, including to arrange for the worker's thread to be in
			// either the "spinning" or "sleeping" portions of the runtime.lock2
			// implementation. Measurement starts after any such "delay".
			//
			// The two threads' goroutines communicate their current position to
			// each other in a non-blocking way via the "turn" state.

			var state struct {
				_    cpu.CacheLinePad
				lock Mutex
				_    cpu.CacheLinePad
				turn atomic.Int64
				_    cpu.CacheLinePad
			}

			var delta atomic.Int64
			var wg sync.WaitGroup

			// coordinator:
			//  - acquire the mutex
			//  - set the turn to 2 mod 4, instructing the worker to begin its Lock call
			//  - wait until the mutex is contended
			//  - wait a bit more so the worker can commit to its sleep
			//  - release the mutex and wait for it to be our turn (0 mod 4) again
			wg.Add(1)
			go func() {
				defer wg.Done()
				var t int64
				for range b.N {
					Lock(&state.lock)
					state.turn.Add(2)
					delay(&state.lock)
					t -= Nanotime() // start the timer
					Unlock(&state.lock)
					for state.turn.Load()&0x2 != 0 {
					}
				}
				state.turn.Add(1)
				delta.Add(t)
			}()

			// worker:
			//  - wait until its our turn (2 mod 4)
			//  - acquire and release the mutex
			//  - switch the turn counter back to the coordinator (0 mod 4)
			wg.Add(1)
			go func() {
				defer wg.Done()
				var t int64
				for {
					switch state.turn.Load() & 0x3 {
					case 0:
					case 1, 3:
						delta.Add(t)
						return
					case 2:
						Lock(&state.lock)
						t += Nanotime() // stop the timer
						Unlock(&state.lock)
						state.turn.Add(2)
					}
				}
			}()

			wg.Wait()
			b.ReportMetric(float64(delta.Load())/float64(b.N), "ns/op")
		}
	}

	b.Run("Solo", func(b *testing.B) {
		var lock Mutex
		for range b.N {
			Lock(&lock)
			Unlock(&lock)
		}
	})

	b.Run("FastPingPong", testcase(func(l *Mutex) {}))
	b.Run("SlowPingPong", testcase(func(l *Mutex) {
		// Wait for the worker to stop spinning and prepare to sleep
		for !MutexContended(l) {
		}
		// Wait a bit longer so the OS can finish committing the worker to its
		// sleep. Balance consistency against getting enough iterations.
		const extraNs = 10e3
		for t0 := Nanotime(); Nanotime()-t0 < extraNs; {
		}
	}))
}
