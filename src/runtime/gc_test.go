// Copyright 2011 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

package runtime_test

import (
	"fmt"
	"internal/asan"
	"internal/msan"
	"internal/race"
	"internal/testenv"
	"math/bits"
	"math/rand"
	"os"
	"reflect"
	"regexp"
	"runtime"
	"runtime/debug"
	"slices"
	"strings"
	"sync"
	"sync/atomic"
	"testing"
	"time"
	"unsafe"
	"weak"
)

func TestGcSys(t *testing.T) {
	t.Skip("skipping known-flaky test; golang.org/issue/37331")
	if os.Getenv("GOGC") == "off" {
		t.Skip("skipping test; GOGC=off in environment")
	}
	got := runTestProg(t, "testprog", "GCSys")
	want := "OK\n"
	if got != want {
		t.Fatalf("expected %q, but got %q", want, got)
	}
}

func TestGcDeepNesting(t *testing.T) {
	type T [2][2][2][2][2][2][2][2][2][2]*int
	a := new(T)

	// Prevent the compiler from applying escape analysis.
	// This makes sure new(T) is allocated on heap, not on the stack.
	t.Logf("%p", a)

	a[0][0][0][0][0][0][0][0][0][0] = new(int)
	*a[0][0][0][0][0][0][0][0][0][0] = 13
	runtime.GC()
	if *a[0][0][0][0][0][0][0][0][0][0] != 13 {
		t.Fail()
	}
}

func TestGcMapIndirection(t *testing.T) {
	defer debug.SetGCPercent(debug.SetGCPercent(1))
	runtime.GC()
	type T struct {
		a [256]int
	}
	m := make(map[T]T)
	for i := 0; i < 2000; i++ {
		var a T
		a.a[0] = i
		m[a] = T{}
	}
}

func TestGcArraySlice(t *testing.T) {
	type X struct {
		buf     [1]byte
		nextbuf []byte
		next    *X
	}
	var head *X
	for i := 0; i < 10; i++ {
		p := &X{}
		p.buf[0] = 42
		p.next = head
		if head != nil {
			p.nextbuf = head.buf[:]
		}
		head = p
		runtime.GC()
	}
	for p := head; p != nil; p = p.next {
		if p.buf[0] != 42 {
			t.Fatal("corrupted heap")
		}
	}
}

func TestGcRescan(t *testing.T) {
	type X struct {
		c     chan error
		nextx *X
	}
	type Y struct {
		X
		nexty *Y
		p     *int
	}
	var head *Y
	for i := 0; i < 10; i++ {
		p := &Y{}
		p.c = make(chan error)
		if head != nil {
			p.nextx = &head.X
		}
		p.nexty = head
		p.p = new(int)
		*p.p = 42
		head = p
		runtime.GC()
	}
	for p := head; p != nil; p = p.nexty {
		if *p.p != 42 {
			t.Fatal("corrupted heap")
		}
	}
}

func TestGcLastTime(t *testing.T) {
	ms := new(runtime.MemStats)
	t0 := time.Now().UnixNano()
	runtime.GC()
	t1 := time.Now().UnixNano()
	runtime.ReadMemStats(ms)
	last := int64(ms.LastGC)
	if t0 > last || last > t1 {
		t.Fatalf("bad last GC time: got %v, want [%v, %v]", last, t0, t1)
	}
	pause := ms.PauseNs[(ms.NumGC+255)%256]
	// Due to timer granularity, pause can actually be 0 on windows
	// or on virtualized environments.
	if pause == 0 {
		t.Logf("last GC pause was 0")
	} else if pause > 10e9 {
		t.Logf("bad last GC pause: got %v, want [0, 10e9]", pause)
	}
}

var hugeSink any

func TestHugeGCInfo(t *testing.T) {
	// The test ensures that compiler can chew these huge types even on weakest machines.
	// The types are not allocated at runtime.
	if hugeSink != nil {
		// 400MB on 32 bots, 4TB on 64-bits.
		const n = (400 << 20) + (unsafe.Sizeof(uintptr(0))-4)<<40
		hugeSink = new([n]*byte)
		hugeSink = new([n]uintptr)
		hugeSink = new(struct {
			x float64
			y [n]*byte
			z []string
		})
		hugeSink = new(struct {
			x float64
			y [n]uintptr
			z []string
		})
	}
}

func TestPeriodicGC(t *testing.T) {
	if runtime.GOARCH == "wasm" {
		t.Skip("no sysmon on wasm yet")
	}

	// Make sure we're not in the middle of a GC.
	runtime.GC()

	var ms1, ms2 runtime.MemStats
	runtime.ReadMemStats(&ms1)

	// Make periodic GC run continuously.
	orig := *runtime.ForceGCPeriod
	*runtime.ForceGCPeriod = 0

	// Let some periodic GCs happen. In a heavily loaded system,
	// it's possible these will be delayed, so this is designed to
	// succeed quickly if things are working, but to give it some
	// slack if things are slow.
	var numGCs uint32
	const want = 2
	for i := 0; i < 200 && numGCs < want; i++ {
		time.Sleep(5 * time.Millisecond)

		// Test that periodic GC actually happened.
		runtime.ReadMemStats(&ms2)
		numGCs = ms2.NumGC - ms1.NumGC
	}
	*runtime.ForceGCPeriod = orig

	if numGCs < want {
		t.Fatalf("no periodic GC: got %v GCs, want >= 2", numGCs)
	}
}

func TestGcZombieReporting(t *testing.T) {
	if asan.Enabled || msan.Enabled || race.Enabled {
		t.Skip("skipped test: checkptr mode catches the issue before getting to zombie reporting")
	}
	// This test is somewhat sensitive to how the allocator works.
	// Pointers in zombies slice may cross-span, thus we
	// add invalidptr=0 for avoiding the badPointer check.
	// See issue https://golang.org/issues/49613/
	got := runTestProg(t, "testprog", "GCZombie", "GODEBUG=invalidptr=0")
	want := "found pointer to free object"
	if !strings.Contains(got, want) {
		t.Fatalf("expected %q in output, but got %q", want, got)
	}
}

func TestGCTestMoveStackOnNextCall(t *testing.T) {
	if asan.Enabled {
		t.Skip("extra allocations with -asan causes this to fail; see #70079")
	}
	t.Parallel()
	var onStack int
	// GCTestMoveStackOnNextCall can fail in rare cases if there's
	// a preemption. This won't happen many times in quick
	// succession, so just retry a few times.
	for retry := 0; retry < 5; retry++ {
		runtime.GCTestMoveStackOnNextCall()
		if moveStackCheck(t, &onStack, uintptr(unsafe.Pointer(&onStack))) {
			// Passed.
			return
		}
	}
	t.Fatal("stack did not move")
}

// This must not be inlined because the point is to force a stack
// growth check and move the stack.
//
//go:noinline
func moveStackCheck(t *testing.T, new *int, old uintptr) bool {
	// new should have been updated by the stack move;
	// old should not have.

	// Capture new's value before doing anything that could
	// further move the stack.
	new2 := uintptr(unsafe.Pointer(new))

	t.Logf("old stack pointer %x, new stack pointer %x", old, new2)
	if new2 == old {
		// Check that we didn't screw up the test's escape analysis.
		if cls := runtime.GCTestPointerClass(unsafe.Pointer(new)); cls != "stack" {
			t.Fatalf("test bug: new (%#x) should be a stack pointer, not %s", new2, cls)
		}
		// This was a real failure.
		return false
	}
	return true
}

func TestGCTestMoveStackRepeatedly(t *testing.T) {
	// Move the stack repeatedly to make sure we're not doubling
	// it each time.
	for i := 0; i < 100; i++ {
		runtime.GCTestMoveStackOnNextCall()
		moveStack1(false)
	}
}

//go:noinline
func moveStack1(x bool) {
	// Make sure this function doesn't get auto-nosplit.
	if x {
		println("x")
	}
}

func TestGCTestIsReachable(t *testing.T) {
	var all, half []unsafe.Pointer
	var want uint64
	for i := 0; i < 16; i++ {
		// The tiny allocator muddies things, so we use a
		// scannable type.
		p := unsafe.Pointer(new(*int))
		all = append(all, p)
		if i%2 == 0 {
			half = append(half, p)
			want |= 1 << i
		}
	}

	got := runtime.GCTestIsReachable(all...)
	if got&want != want {
		// This is a serious bug - an object is live (due to the KeepAlive
		// call below), but isn't reported as such.
		t.Fatalf("live object not in reachable set; want %b, got %b", want, got)
	}
	if bits.OnesCount64(got&^want) > 1 {
		// Note: we can occasionally have a value that is retained even though
		// it isn't live, due to conservative scanning of stack frames.
		// See issue 67204. For now, we allow a "slop" of 1 unintentionally
		// retained object.
		t.Fatalf("dead object in reachable set; want %b, got %b", want, got)
	}
	runtime.KeepAlive(half)
}

var pointerClassBSS *int
var pointerClassData = 42

func TestGCTestPointerClass(t *testing.T) {
	if asan.Enabled {
		t.Skip("extra allocations cause this test to fail; see #70079")
	}
	t.Parallel()
	check := func(p unsafe.Pointer, want string) {
		t.Helper()
		got := runtime.GCTestPointerClass(p)
		if got != want {
			// Convert the pointer to a uintptr to avoid
			// escaping it.
			t.Errorf("for %#x, want class %s, got %s", uintptr(p), want, got)
		}
	}
	var onStack int
	var notOnStack int
	check(unsafe.Pointer(&onStack), "stack")
	check(unsafe.Pointer(runtime.Escape(&notOnStack)), "heap")
	check(unsafe.Pointer(&pointerClassBSS), "bss")
	check(unsafe.Pointer(&pointerClassData), "data")
	check(nil, "other")
}

func BenchmarkAllocation(b *testing.B) {
	type T struct {
		x, y *byte
	}
	ngo := runtime.GOMAXPROCS(0)
	work := make(chan bool, b.N+ngo)
	result := make(chan *T)
	for i := 0; i < b.N; i++ {
		work <- true
	}
	for i := 0; i < ngo; i++ {
		work <- false
	}
	for i := 0; i < ngo; i++ {
		go func() {
			var x *T
			for <-work {
				for i := 0; i < 1000; i++ {
					x = &T{}
				}
			}
			result <- x
		}()
	}
	for i := 0; i < ngo; i++ {
		<-result
	}
}

func TestPrintGC(t *testing.T) {
	if testing.Short() {
		t.Skip("Skipping in short mode")
	}
	defer runtime.GOMAXPROCS(runtime.GOMAXPROCS(2))
	done := make(chan bool)
	go func() {
		for {
			select {
			case <-done:
				return
			default:
				runtime.GC()
			}
		}
	}()
	for i := 0; i < 1e4; i++ {
		func() {
			defer print("")
		}()
	}
	close(done)
}

func testTypeSwitch(x any) error {
	switch y := x.(type) {
	case nil:
		// ok
	case error:
		return y
	}
	return nil
}

func testAssert(x any) error {
	if y, ok := x.(error); ok {
		return y
	}
	return nil
}

func testAssertVar(x any) error {
	var y, ok = x.(error)
	if ok {
		return y
	}
	return nil
}

var a bool

//go:noinline
func testIfaceEqual(x any) {
	if x == "abc" {
		a = true
	}
}

func TestPageAccounting(t *testing.T) {
	// Grow the heap in small increments. This used to drop the
	// pages-in-use count below zero because of a rounding
	// mismatch (golang.org/issue/15022).
	const blockSize = 64 << 10
	blocks := make([]*[blockSize]byte, (64<<20)/blockSize)
	for i := range blocks {
		blocks[i] = new([blockSize]byte)
	}

	// Check that the running page count matches reality.
	pagesInUse, counted := runtime.CountPagesInUse()
	if pagesInUse != counted {
		t.Fatalf("mheap_.pagesInUse is %d, but direct count is %d", pagesInUse, counted)
	}
}

func init() {
	// Enable ReadMemStats' double-check mode.
	*runtime.DoubleCheckReadMemStats = true
}

func TestReadMemStats(t *testing.T) {
	base, slow := runtime.ReadMemStatsSlow()
	if base != slow {
		logDiff(t, "MemStats", reflect.ValueOf(base), reflect.ValueOf(slow))
		t.Fatal("memstats mismatch")
	}
}

func logDiff(t *testing.T, prefix string, got, want reflect.Value) {
	typ := got.Type()
	switch typ.Kind() {
	case reflect.Array, reflect.Slice:
		if got.Len() != want.Len() {
			t.Logf("len(%s): got %v, want %v", prefix, got, want)
			return
		}
		for i := 0; i < got.Len(); i++ {
			logDiff(t, fmt.Sprintf("%s[%d]", prefix, i), got.Index(i), want.Index(i))
		}
	case reflect.Struct:
		for i := 0; i < typ.NumField(); i++ {
			gf, wf := got.Field(i), want.Field(i)
			logDiff(t, prefix+"."+typ.Field(i).Name, gf, wf)
		}
	case reflect.Map:
		t.Fatal("not implemented: logDiff for map")
	default:
		if got.Interface() != want.Interface() {
			t.Logf("%s: got %v, want %v", prefix, got, want)
		}
	}
}

func BenchmarkReadMemStats(b *testing.B) {
	var ms runtime.MemStats
	const heapSize = 100 << 20
	x := make([]*[1024]byte, heapSize/1024)
	for i := range x {
		x[i] = new([1024]byte)
	}

	b.ResetTimer()
	for i := 0; i < b.N; i++ {
		runtime.ReadMemStats(&ms)
	}

	runtime.KeepAlive(x)
}

func applyGCLoad(b *testing.B) func() {
	// We’ll apply load to the runtime with maxProcs-1 goroutines
	// and use one more to actually benchmark. It doesn't make sense
	// to try to run this test with only 1 P (that's what
	// BenchmarkReadMemStats is for).
	maxProcs := runtime.GOMAXPROCS(-1)
	if maxProcs == 1 {
		b.Skip("This benchmark can only be run with GOMAXPROCS > 1")
	}

	// Code to build a big tree with lots of pointers.
	type node struct {
		children [16]*node
	}
	var buildTree func(depth int) *node
	buildTree = func(depth int) *node {
		tree := new(node)
		if depth != 0 {
			for i := range tree.children {
				tree.children[i] = buildTree(depth - 1)
			}
		}
		return tree
	}

	// Keep the GC busy by continuously generating large trees.
	done := make(chan struct{})
	var wg sync.WaitGroup
	for i := 0; i < maxProcs-1; i++ {
		wg.Add(1)
		go func() {
			defer wg.Done()
			var hold *node
		loop:
			for {
				hold = buildTree(5)
				select {
				case <-done:
					break loop
				default:
				}
			}
			runtime.KeepAlive(hold)
		}()
	}
	return func() {
		close(done)
		wg.Wait()
	}
}

func BenchmarkReadMemStatsLatency(b *testing.B) {
	stop := applyGCLoad(b)

	// Spend this much time measuring latencies.
	latencies := make([]time.Duration, 0, 1024)

	// Run for timeToBench hitting ReadMemStats continuously
	// and measuring the latency.
	b.ResetTimer()
	var ms runtime.MemStats
	for i := 0; i < b.N; i++ {
		// Sleep for a bit, otherwise we're just going to keep
		// stopping the world and no one will get to do anything.
		time.Sleep(100 * time.Millisecond)
		start := time.Now()
		runtime.ReadMemStats(&ms)
		latencies = append(latencies, time.Since(start))
	}
	// Make sure to stop the timer before we wait! The load created above
	// is very heavy-weight and not easy to stop, so we could end up
	// confusing the benchmarking framework for small b.N.
	b.StopTimer()
	stop()

	// Disable the default */op metrics.
	// ns/op doesn't mean anything because it's an average, but we
	// have a sleep in our b.N loop above which skews this significantly.
	b.ReportMetric(0, "ns/op")
	b.ReportMetric(0, "B/op")
	b.ReportMetric(0, "allocs/op")

	// Sort latencies then report percentiles.
	slices.Sort(latencies)
	b.ReportMetric(float64(latencies[len(latencies)*50/100]), "p50-ns")
	b.ReportMetric(float64(latencies[len(latencies)*90/100]), "p90-ns")
	b.ReportMetric(float64(latencies[len(latencies)*99/100]), "p99-ns")
}

func TestUserForcedGC(t *testing.T) {
	// Test that runtime.GC() triggers a GC even if GOGC=off.
	defer debug.SetGCPercent(debug.SetGCPercent(-1))

	var ms1, ms2 runtime.MemStats
	runtime.ReadMemStats(&ms1)
	runtime.GC()
	runtime.ReadMemStats(&ms2)
	if ms1.NumGC == ms2.NumGC {
		t.Fatalf("runtime.GC() did not trigger GC")
	}
	if ms1.NumForcedGC == ms2.NumForcedGC {
		t.Fatalf("runtime.GC() was not accounted in NumForcedGC")
	}
}

func writeBarrierBenchmark(b *testing.B, f func()) {
	runtime.GC()
	var ms runtime.MemStats
	runtime.ReadMemStats(&ms)
	//b.Logf("heap size: %d MB", ms.HeapAlloc>>20)

	// Keep GC running continuously during the benchmark, which in
	// turn keeps the write barrier on continuously.
	var stop uint32
	done := make(chan bool)
	go func() {
		for atomic.LoadUint32(&stop) == 0 {
			runtime.GC()
		}
		close(done)
	}()
	defer func() {
		atomic.StoreUint32(&stop, 1)
		<-done
	}()

	b.ResetTimer()
	f()
	b.StopTimer()
}

func BenchmarkWriteBarrier(b *testing.B) {
	if runtime.GOMAXPROCS(-1) < 2 {
		// We don't want GC to take our time.
		b.Skip("need GOMAXPROCS >= 2")
	}

	// Construct a large tree both so the GC runs for a while and
	// so we have a data structure to manipulate the pointers of.
	type node struct {
		l, r *node
	}
	var wbRoots []*node
	var mkTree func(level int) *node
	mkTree = func(level int) *node {
		if level == 0 {
			return nil
		}
		n := &node{mkTree(level - 1), mkTree(level - 1)}
		if level == 10 {
			// Seed GC with enough early pointers so it
			// doesn't start termination barriers when it
			// only has the top of the tree.
			wbRoots = append(wbRoots, n)
		}
		return n
	}
	const depth = 22 // 64 MB
	root := mkTree(22)

	writeBarrierBenchmark(b, func() {
		var stack [depth]*node
		tos := -1

		// There are two write barriers per iteration, so i+=2.
		for i := 0; i < b.N; i += 2 {
			if tos == -1 {
				stack[0] = root
				tos = 0
			}

			// Perform one step of reversing the tree.
			n := stack[tos]
			if n.l == nil {
				tos--
			} else {
				n.l, n.r = n.r, n.l
				stack[tos] = n.l
				stack[tos+1] = n.r
				tos++
			}

			if i%(1<<12) == 0 {
				// Avoid non-preemptible loops (see issue #10958).
				runtime.Gosched()
			}
		}
	})

	runtime.KeepAlive(wbRoots)
}

func BenchmarkBulkWriteBarrier(b *testing.B) {
	if runtime.GOMAXPROCS(-1) < 2 {
		// We don't want GC to take our time.
		b.Skip("need GOMAXPROCS >= 2")
	}

	// Construct a large set of objects we can copy around.
	const heapSize = 64 << 20
	type obj [16]*byte
	ptrs := make([]*obj, heapSize/unsafe.Sizeof(obj{}))
	for i := range ptrs {
		ptrs[i] = new(obj)
	}

	writeBarrierBenchmark(b, func() {
		const blockSize = 1024
		var pos int
		for i := 0; i < b.N; i += blockSize {
			// Rotate block.
			block := ptrs[pos : pos+blockSize]
			first := block[0]
			copy(block, block[1:])
			block[blockSize-1] = first

			pos += blockSize
			if pos+blockSize > len(ptrs) {
				pos = 0
			}

			runtime.Gosched()
		}
	})

	runtime.KeepAlive(ptrs)
}

func BenchmarkScanStackNoLocals(b *testing.B) {
	var ready sync.WaitGroup
	teardown := make(chan bool)
	for j := 0; j < 10; j++ {
		ready.Add(1)
		go func() {
			x := 100000
			countpwg(&x, &ready, teardown)
		}()
	}
	ready.Wait()
	b.ResetTimer()
	for i := 0; i < b.N; i++ {
		b.StartTimer()
		runtime.GC()
		runtime.GC()
		b.StopTimer()
	}
	close(teardown)
}

func BenchmarkMSpanCountAlloc(b *testing.B) {
	// Allocate one dummy mspan for the whole benchmark.
	s := runtime.AllocMSpan()
	defer runtime.FreeMSpan(s)

	// n is the number of bytes to benchmark against.
	// n must always be a multiple of 8, since gcBits is
	// always rounded up 8 bytes.
	for _, n := range []int{8, 16, 32, 64, 128} {
		b.Run(fmt.Sprintf("bits=%d", n*8), func(b *testing.B) {
			// Initialize a new byte slice with pseudo-random data.
			bits := make([]byte, n)
			rand.Read(bits)

			b.ResetTimer()
			for i := 0; i < b.N; i++ {
				runtime.MSpanCountAlloc(s, bits)
			}
		})
	}
}

func countpwg(n *int, ready *sync.WaitGroup, teardown chan bool) {
	if *n == 0 {
		ready.Done()
		<-teardown
		return
	}
	*n--
	countpwg(n, ready, teardown)
}

func TestMemoryLimit(t *testing.T) {
	if testing.Short() {
		t.Skip("stress test that takes time to run")
	}
	if runtime.NumCPU() < 4 {
		t.Skip("want at least 4 CPUs for this test")
	}
	got := runTestProg(t, "testprog", "GCMemoryLimit")
	want := "OK\n"
	if got != want {
		t.Fatalf("expected %q, but got %q", want, got)
	}
}

func TestMemoryLimitNoGCPercent(t *testing.T) {
	if testing.Short() {
		t.Skip("stress test that takes time to run")
	}
	if runtime.NumCPU() < 4 {
		t.Skip("want at least 4 CPUs for this test")
	}
	got := runTestProg(t, "testprog", "GCMemoryLimitNoGCPercent")
	want := "OK\n"
	if got != want {
		t.Fatalf("expected %q, but got %q", want, got)
	}
}

func TestMyGenericFunc(t *testing.T) {
	runtime.MyGenericFunc[int]()
}

func TestWeakToStrongMarkTermination(t *testing.T) {
	testenv.MustHaveParallelism(t)

	type T struct {
		a *int
		b int
	}
	defer runtime.GOMAXPROCS(runtime.GOMAXPROCS(2))
	defer debug.SetGCPercent(debug.SetGCPercent(-1))
	w := make([]weak.Pointer[T], 2048)

	// Make sure there's no out-standing GC from a previous test.
	runtime.GC()

	// Create many objects with a weak pointers to them.
	for i := range w {
		x := new(T)
		x.a = new(int)
		w[i] = weak.Make(x)
	}

	// Reset the restart flag.
	runtime.GCMarkDoneResetRestartFlag()

	// Prevent mark termination from completing.
	runtime.SetSpinInGCMarkDone(true)

	// Start a GC, and wait a little bit to get something spinning in mark termination.
	// Simultaneously, fire off another goroutine to disable spinning. If everything's
	// working correctly, then weak.Value will block, so we need to make sure something
	// prevents the GC from continuing to spin.
	done := make(chan struct{})
	go func() {
		runtime.GC()
		done <- struct{}{}
	}()
	go func() {
		// Usleep here instead of time.Sleep. time.Sleep
		// can allocate, and if we get unlucky, then it
		// can end up stuck in gcMarkDone with nothing to
		// wake it.
		runtime.Usleep(100000) // 100ms

		// Let mark termination continue.
		runtime.SetSpinInGCMarkDone(false)
	}()
	time.Sleep(10 * time.Millisecond)

	// Perform many weak->strong conversions in the critical window.
	var wg sync.WaitGroup
	for _, wp := range w {
		wg.Add(1)
		go func() {
			defer wg.Done()
			wp.Value()
		}()
	}

	// Make sure the GC completes.
	<-done

	// Make sure all the weak->strong conversions finish.
	wg.Wait()

	// The bug is triggered if there's still mark work after gcMarkDone stops the world.
	//
	// This can manifest in one of two ways today:
	// - An exceedingly rare crash in mark termination.
	// - gcMarkDone restarts, as if issue #27993 is at play.
	//
	// Check for the latter. This is a fairly controlled environment, so #27993 is very
	// unlikely to happen (it's already rare to begin with) but we'll always _appear_ to
	// trigger the same bug if weak->strong conversions aren't properly coordinated with
	// mark termination.
	if runtime.GCMarkDoneRestarted() {
		t.Errorf("gcMarkDone restarted")
	}
}

func TestMSpanQueue(t *testing.T) {
	expectSize := func(t *testing.T, q *runtime.MSpanQueue, want int) {
		t.Helper()
		if got := q.Size(); got != want {
			t.Errorf("expected size %d, got %d", want, got)
		}
	}
	expectMSpan := func(t *testing.T, got, want *runtime.MSpan, op string) {
		t.Helper()
		if got != want {
			t.Errorf("expected mspan %p from %s, got %p", want, op, got)
		}
	}
	makeSpans := func(t *testing.T, n int) ([]*runtime.MSpan, func()) {
		t.Helper()
		spans := make([]*runtime.MSpan, 0, n)
		for range cap(spans) {
			spans = append(spans, runtime.AllocMSpan())
		}
		return spans, func() {
			for i, s := range spans {
				runtime.FreeMSpan(s)
				spans[i] = nil
			}
		}
	}
	t.Run("Empty", func(t *testing.T) {
		var q runtime.MSpanQueue
		expectSize(t, &q, 0)
		expectMSpan(t, q.Pop(), nil, "pop")
	})
	t.Run("PushPop", func(t *testing.T) {
		s := runtime.AllocMSpan()
		defer runtime.FreeMSpan(s)

		var q runtime.MSpanQueue
		q.Push(s)
		expectSize(t, &q, 1)
		expectMSpan(t, q.Pop(), s, "pop")
		expectMSpan(t, q.Pop(), nil, "pop")
	})
	t.Run("PushPopPushPop", func(t *testing.T) {
		s0 := runtime.AllocMSpan()
		defer runtime.FreeMSpan(s0)
		s1 := runtime.AllocMSpan()
		defer runtime.FreeMSpan(s1)

		var q runtime.MSpanQueue

		// Push and pop s0.
		q.Push(s0)
		expectSize(t, &q, 1)
		expectMSpan(t, q.Pop(), s0, "pop")
		expectMSpan(t, q.Pop(), nil, "pop")

		// Push and pop s1.
		q.Push(s1)
		expectSize(t, &q, 1)
		expectMSpan(t, q.Pop(), s1, "pop")
		expectMSpan(t, q.Pop(), nil, "pop")
	})
	t.Run("PushPushPopPop", func(t *testing.T) {
		s0 := runtime.AllocMSpan()
		defer runtime.FreeMSpan(s0)
		s1 := runtime.AllocMSpan()
		defer runtime.FreeMSpan(s1)

		var q runtime.MSpanQueue
		q.Push(s0)
		expectSize(t, &q, 1)
		q.Push(s1)
		expectSize(t, &q, 2)
		expectMSpan(t, q.Pop(), s0, "pop")
		expectMSpan(t, q.Pop(), s1, "pop")
		expectMSpan(t, q.Pop(), nil, "pop")
	})
	t.Run("EmptyTakeAll", func(t *testing.T) {
		var q runtime.MSpanQueue
		var p runtime.MSpanQueue
		expectSize(t, &p, 0)
		expectSize(t, &q, 0)
		p.TakeAll(&q)
		expectSize(t, &p, 0)
		expectSize(t, &q, 0)
		expectMSpan(t, q.Pop(), nil, "pop")
		expectMSpan(t, p.Pop(), nil, "pop")
	})
	t.Run("Push4TakeAll", func(t *testing.T) {
		spans, free := makeSpans(t, 4)
		defer free()

		var q runtime.MSpanQueue
		for i, s := range spans {
			expectSize(t, &q, i)
			q.Push(s)
			expectSize(t, &q, i+1)
		}

		var p runtime.MSpanQueue
		p.TakeAll(&q)
		expectSize(t, &p, 4)
		for i := range p.Size() {
			expectMSpan(t, p.Pop(), spans[i], "pop")
		}
		expectSize(t, &p, 0)
		expectMSpan(t, q.Pop(), nil, "pop")
		expectMSpan(t, p.Pop(), nil, "pop")
	})
	t.Run("Push4Pop3", func(t *testing.T) {
		spans, free := makeSpans(t, 4)
		defer free()

		var q runtime.MSpanQueue
		for i, s := range spans {
			expectSize(t, &q, i)
			q.Push(s)
			expectSize(t, &q, i+1)
		}
		p := q.PopN(3)
		expectSize(t, &p, 3)
		expectSize(t, &q, 1)
		for i := range p.Size() {
			expectMSpan(t, p.Pop(), spans[i], "pop")
		}
		expectMSpan(t, q.Pop(), spans[len(spans)-1], "pop")
		expectSize(t, &p, 0)
		expectSize(t, &q, 0)
		expectMSpan(t, q.Pop(), nil, "pop")
		expectMSpan(t, p.Pop(), nil, "pop")
	})
	t.Run("Push4Pop0", func(t *testing.T) {
		spans, free := makeSpans(t, 4)
		defer free()

		var q runtime.MSpanQueue
		for i, s := range spans {
			expectSize(t, &q, i)
			q.Push(s)
			expectSize(t, &q, i+1)
		}
		p := q.PopN(0)
		expectSize(t, &p, 0)
		expectSize(t, &q, 4)
		for i := range q.Size() {
			expectMSpan(t, q.Pop(), spans[i], "pop")
		}
		expectSize(t, &p, 0)
		expectSize(t, &q, 0)
		expectMSpan(t, q.Pop(), nil, "pop")
		expectMSpan(t, p.Pop(), nil, "pop")
	})
	t.Run("Push4Pop4", func(t *testing.T) {
		spans, free := makeSpans(t, 4)
		defer free()

		var q runtime.MSpanQueue
		for i, s := range spans {
			expectSize(t, &q, i)
			q.Push(s)
			expectSize(t, &q, i+1)
		}
		p := q.PopN(4)
		expectSize(t, &p, 4)
		expectSize(t, &q, 0)
		for i := range p.Size() {
			expectMSpan(t, p.Pop(), spans[i], "pop")
		}
		expectSize(t, &p, 0)
		expectMSpan(t, q.Pop(), nil, "pop")
		expectMSpan(t, p.Pop(), nil, "pop")
	})
	t.Run("Push4Pop5", func(t *testing.T) {
		spans, free := makeSpans(t, 4)
		defer free()

		var q runtime.MSpanQueue
		for i, s := range spans {
			expectSize(t, &q, i)
			q.Push(s)
			expectSize(t, &q, i+1)
		}
		p := q.PopN(5)
		expectSize(t, &p, 4)
		expectSize(t, &q, 0)
		for i := range p.Size() {
			expectMSpan(t, p.Pop(), spans[i], "pop")
		}
		expectSize(t, &p, 0)
		expectMSpan(t, q.Pop(), nil, "pop")
		expectMSpan(t, p.Pop(), nil, "pop")
	})
}

func TestDetectFinalizerAndCleanupLeaks(t *testing.T) {
	got := runTestProg(t, "testprog", "DetectFinalizerAndCleanupLeaks", "GODEBUG=checkfinalizers=1")
	sp := strings.SplitN(got, "detected possible issues with cleanups and/or finalizers", 2)
	if len(sp) != 2 {
		t.Fatalf("expected the runtime to throw, got:\n%s", got)
	}
	if strings.Count(sp[0], "is reachable from") != 2 {
		t.Fatalf("expected exactly two leaked cleanups and/or finalizers, got:\n%s", got)
	}
	// N.B. Disable in race mode and in asan mode. Both disable the tiny allocator.
	wantSymbolizedLocations := 2
	if !race.Enabled && !asan.Enabled {
		if strings.Count(sp[0], "is in a tiny block") != 1 {
			t.Fatalf("expected exactly one report for allocation in a tiny block, got:\n%s", got)
		}
		wantSymbolizedLocations++
	}
	if strings.Count(sp[0], "main.DetectFinalizerAndCleanupLeaks()") != wantSymbolizedLocations {
		t.Fatalf("expected %d symbolized locations, got:\n%s", wantSymbolizedLocations, got)
	}
}

// This tests the goroutine leak garbage collector.
func TestGoroutineLeakGC(t *testing.T) {
	// Goroutine leak test case.
	//
	// Test cases can be configured with test name, the name of the entry point function,
	// a set of expected leaks identified by regular expressions, and the number of times
	// the test should be repeated.
	//
	// Repetitions are used to amortize flakiness in some tests.
	type testCase struct {
		name          string
		simple        bool
		expectedLeaks map[*regexp.Regexp]bool

		// flakyLeaks are goroutine leaks that are too flaky to be reliably detected.
		// Still, they might pop up every once in a while.
		// If these occur, do not fail the test due to unexpected leaks.
		flakyLeaks map[*regexp.Regexp]struct{}
	}

	// makeAnyTest is a short-hand for creating test cases.
	// Each of the leaks in the list is identified by a regular expression.
	// If a leak is flaky, it is added to the flakyLeaks map.
	makeAnyTest := func(
		name string,
		flaky bool,
		leaks ...string) testCase {
		tc := testCase{
			name:          name,
			expectedLeaks: make(map[*regexp.Regexp]bool, len(leaks)),
			flakyLeaks:    make(map[*regexp.Regexp]struct{}, len(leaks)),
		}

		for _, leak := range leaks {
			if !flaky {
				tc.expectedLeaks[regexp.MustCompile(leak)] = false
			} else {
				tc.flakyLeaks[regexp.MustCompile(leak)] = struct{}{}
			}
		}

		return tc
	}

	// makeTest is a short-hand for creating non-flaky test cases.
	makeTest := func(name string, leaks ...string) testCase {
		tcase := makeAnyTest(name, false, leaks...)
		tcase.simple = true
		return tcase
	}

	// makeFlakyTest is a short-hand for creating flaky test cases.
	makeFlakyTest := func(name string, leaks ...string) testCase {
		return makeAnyTest(name, true, leaks...)
	}

	goroutineHeader := regexp.MustCompile(`goroutine \d+ \[`)

	// extractLeaks takes the output of a test and splits it into a
	// list of strings denoting goroutine leaks.
	//
	// If the input is:
	//
	// goroutine 1 [wait reason (leaked)]:
	// main.leaked()
	// 	./testgoroutineleakgc/foo.go:37 +0x100
	// created by main.main()
	// 	./testgoroutineleakgc/main.go:10 +0x20
	//
	// goroutine 2 [wait reason (leaked)]:
	// main.leaked2()
	// 	./testgoroutineleakgc/foo.go:37 +0x100
	// created by main.main()
	// 	./testgoroutineleakgc/main.go:10 +0x20
	//
	// The output is (as a list of strings):
	//
	// leaked() [wait reason]
	// leaked2() [wait reason]
	extractLeaks := func(output string) []string {
		stacks := strings.Split(output, "\n\ngoroutine")
		var leaks []string
		for _, stack := range stacks {
			lines := strings.Split(stack, "\n")
			if len(lines) < 5 {
				// Expecting at least the following lines (where n=len(lines)-1):
				//
				// [0] goroutine n [wait reason (leaked)]
				// ...
				// [n-3] bottom.leak.frame(...)
				// [n-2]  ./bottom/leak/frame/source.go:line
				// [n-1] created by go.instruction()
				// [n] 	  ./go/instruction/source.go:line
				continue
			}

			if !strings.Contains(lines[0], "(leaked)") {
				// Ignore non-leaked goroutines.
				continue
			}

			// Get the wait reason from the goroutine header.
			header := lines[0]
			waitReason := goroutineHeader.ReplaceAllString(header, "[")
			waitReason = strings.ReplaceAll(waitReason, " (leaked)", "")

			// Get the function name from the stack trace (should be two lines above `created by`).
			var funcName string
			for i := len(lines) - 1; i >= 0; i-- {
				if strings.Contains(lines[i], "created by") {
					funcName = strings.TrimPrefix(lines[i-2], "main.")
					break
				}
			}
			if funcName == "" {
				t.Fatalf("failed to extract function name from stack trace: %s", lines)
			}

			leaks = append(leaks, funcName+" "+waitReason)
		}
		return leaks
	}

	// Micro tests involve very simple leaks for each type of concurrency primitive operation.
	microTests := []testCase{
		makeTest("NilRecv",
			`NilRecv\.func1\(.* \[chan receive \(nil chan\)\]`,
		),
		makeTest("NilSend",
			`NilSend\.func1\(.* \[chan send \(nil chan\)\]`,
		),
		makeTest("SelectNoCases",
			`SelectNoCases\.func1\(.* \[select \(no cases\)\]`,
		),
		makeTest("ChanRecv",
			`ChanRecv\.func1\(.* \[chan receive\]`,
		),
		makeTest("ChanSend",
			`ChanSend\.func1\(.* \[chan send\]`,
		),
		makeTest("Select",
			`Select\.func1\(.* \[select\]`,
		),
		makeTest("WaitGroup",
			`WaitGroup\.func1\(.* \[sync\.WaitGroup\.Wait\]`,
		),
		makeTest("MutexStack",
			`MutexStack\.func1\(.* \[sync\.Mutex\.Lock\]`,
		),
		makeTest("MutexHeap",
			`MutexHeap\.func1.1\(.* \[sync\.Mutex\.Lock\]`,
		),
		makeTest("Cond",
			`Cond\.func1\(.* \[sync\.Cond\.Wait\]`,
		),
		makeTest("RWMutexRLock",
			`RWMutexRLock\.func1\(.* \[sync\.RWMutex\.RLock\]`,
		),
		makeTest("RWMutexLock",
			`RWMutexLock\.func1\(.* \[sync\.(RW)?Mutex\.Lock\]`,
		),
		makeTest("Mixed",
			`Mixed\.func1\(.* \[sync\.WaitGroup\.Wait\]`,
			`Mixed\.func1.1\(.* \[chan send\]`,
		),
		makeTest("NoLeakGlobal"),
	}

	// Stress tests are flaky and we do not strictly care about their output.
	// They are only intended to stress the goroutine leak detector and profiling
	// infrastructure in interesting ways.
	stressTestCases := []testCase{
		makeFlakyTest("SpawnGC",
			`spawnGC.func1\(.* \[chan receive\]`,
		),
	}

	// Common goroutine leak patterns.
	// Extracted from "Unveiling and Vanquishing Goroutine Leaks in Enterprise Microservices: A Dynamic Analysis Approach"
	// doi:10.1109/CGO57630.2024.10444835
	patternTestCases := []testCase{
		makeTest("NoCloseRange",
			`noCloseRange\(.* \[chan send\]`,
			`noCloseRange\.func1\(.* \[chan receive\]`,
		),
		makeTest("MethodContractViolation",
			`worker\.Start\.func1\(.* \[select\]`,
		),
		makeTest("DoubleSend",
			`DoubleSend\.func3\(.* \[chan send\]`,
		),
		makeTest("EarlyReturn",
			`earlyReturn\.func1\(.* \[chan send\]`,
		),
		makeTest("NCastLeak",
			`nCastLeak\.func1\(.* \[chan send\]`,
			`NCastLeak\.func2\(.* \[chan receive\]`,
		),
		makeTest("Timeout",
			// (vsaioc): Timeout is *theoretically* flaky, but the
			// pseudo-random choice for select case branches makes it
			// practically impossible for it to fail.
			`timeout\.func1\(.* \[chan send\]`,
		),
	}

	// GoKer tests from "GoBench: A Benchmark Suite of Real-World Go Concurrency Bugs".
	// White paper found at https://lujie.ac.cn/files/papers/GoBench.pdf
	// doi:10.1109/CGO51591.2021.9370317.
	//
	// This list is curated for tests that are not excessively flaky.
	// Some tests are also excluded because they are redundant.
	//
	// TODO(vsaioc): Some of these might be removable (their patterns may overlap).
	gokerTestCases := []testCase{
		makeTest("Cockroach584",
			`Cockroach584\.func2\(.* \[sync\.Mutex\.Lock\]`,
		),
		makeFlakyTest("Cockroach1055",
			`Cockroach1055\.func2\(.* \[chan receive\]`,
			`Cockroach1055\.func2\.2\(.* \[sync\.WaitGroup\.Wait\]`,
			`Cockroach1055\.func2\.1\(.* \[chan receive\]`,
			`Cockroach1055\.func2\.1\(.* \[sync\.Mutex\.Lock\]`,
		),
		makeFlakyTest("Cockroach1462",
			`\(\*Stopper_cockroach1462\)\.RunWorker\.func1\(.* \[chan send\]`,
			`Cockroach1462\.func2\(.* \[sync\.WaitGroup\.Wait\]`,
		),
		makeFlakyTest("Cockroach2448",
			`\(\*Store_cockroach2448\)\.processRaft\(.* \[select\]`,
			`\(\*state_cockroach2448\)\.start\(.* \[select\]`,
		),
		makeFlakyTest("Cockroach3710",
			`\(\*Store_cockroach3710\)\.ForceRaftLogScanAndProcess\(.* \[sync\.RWMutex\.RLock\]`,
			`\(\*Store_cockroach3710\)\.processRaft\.func1\(.* \[sync\.RWMutex\.Lock\]`,
		),
		makeFlakyTest("Cockroach6181",
			`testRangeCacheCoalescedRequests_cockroach6181\(.* \[sync\.WaitGroup\.Wait\]`,
			`testRangeCacheCoalescedRequests_cockroach6181\.func1\.1\(.* \[sync\.(RW)?Mutex\.Lock\]`,
			`testRangeCacheCoalescedRequests_cockroach6181\.func1\.1\(.* \[sync\.RWMutex\.RLock\]`,
		),
		makeFlakyTest("Cockroach7504",
			`Cockroach7504\.func2\.1.* \[sync\.Mutex\.Lock\]`,
			`Cockroach7504\.func2\.2.* \[sync\.Mutex\.Lock\]`,
		),
		makeFlakyTest("Cockroach9935",
			`\(\*loggingT_cockroach9935\)\.outputLogEntry\(.* \[sync\.Mutex\.Lock\]`,
		),
		makeFlakyTest("Cockroach10214",
			`Cockroach10214\.func2\.1\(.* \[sync\.Mutex\.Lock\]`,
			`Cockroach10214\.func2\.2\(.* \[sync\.Mutex\.Lock\]`,
		),
		makeTest("Cockroach10790",
			`\(\*Replica_cockroach10790\)\.beginCmds\.func1\(.* \[chan receive\]`,
		),
		makeTest("Cockroach13197",
			`\(\*Tx_cockroach13197\)\.awaitDone\(.* \[chan receive\]`,
		),
		makeTest("Cockroach13755",
			`\(\*Rows_cockroach13755\)\.awaitDone\(.* \[chan receive\]`,
		),
		makeFlakyTest("Cockroach16167",
			`Cockroach16167\.func2\(.* \[sync\.RWMutex\.RLock\]`,
			`\(\*Executor_cockroach16167\)\.Start\(.* \[sync\.RWMutex\.Lock\]`,
		),
		makeFlakyTest("Cockroach18101",
			`restore_cockroach18101\.func1\(.* \[chan send\]`,
		),
		makeTest("Cockroach24808",
			`Cockroach24808\.func2\(.* \[chan send\]`,
		),
		makeTest("Cockroach25456",
			`Cockroach25456\.func2\(.* \[chan receive\]`,
		),
		makeTest("Cockroach35073",
			`Cockroach35073\.func2.1\(.* \[chan send\]`,
			`Cockroach35073\.func2\(.* \[chan send\]`,
		),
		makeTest("Cockroach35931",
			`Cockroach35931\.func2\(.* \[chan send\]`,
		),
		makeTest("Etcd5509",
			`Etcd5509\.func2\(.* \[sync\.RWMutex\.Lock\]`,
		),
		makeTest("Etcd6708",
			`Etcd6708\.func2\(.* \[sync\.RWMutex\.RLock\]`,
		),
		makeFlakyTest("Etcd6857",
			`\(\*node_etcd6857\)\.Status\(.* \[chan send\]`,
		),
		makeFlakyTest("Etcd6873",
			`\(\*watchBroadcasts_etcd6873\)\.stop\(.* \[chan receive\]`,
			`newWatchBroadcasts_etcd6873\.func1\(.* \[sync\.Mutex\.Lock\]`,
		),
		makeFlakyTest("Etcd7492",
			`Etcd7492\.func2\(.* \[sync\.WaitGroup\.Wait\]`,
			`Etcd7492\.func2\.1\(.* \[chan send\]`,
			`\(\*simpleTokenTTLKeeper_etcd7492\)\.run\(.* \[sync\.Mutex\.Lock\]`,
		),
		makeFlakyTest("Etcd7902",
			`doRounds_etcd7902\.func1\(.* \[chan receive\]`,
			`doRounds_etcd7902\.func1\(.* \[sync\.Mutex\.Lock\]`,
			`runElectionFunc_etcd7902\(.* \[sync\.WaitGroup\.Wait\]`,
		),
		makeTest("Etcd10492",
			`Etcd10492\.func2\(.* \[sync\.Mutex\.Lock\]`,
		),
		makeTest("Grpc660",
			`\(\*benchmarkClient_grpc660\)\.doCloseLoopUnary\.func1\(.* \[chan send\]`,
		),
		makeFlakyTest("Grpc795",
			`\(\*Server_grpc795\)\.Serve\(.* \[sync\.Mutex\.Lock\]`,
			`testServerGracefulStopIdempotent_grpc795\(.* \[sync\.Mutex\.Lock\]`,
		),
		makeTest("Grpc862",
			`DialContext_grpc862\.func2\(.* \[chan receive\]`),
		makeTest("Grpc1275",
			`testInflightStreamClosing_grpc1275\.func1\(.* \[chan receive\]`),
		makeTest("Grpc1424",
			`DialContext_grpc1424\.func1\(.* \[chan receive\]`),
		makeFlakyTest("Grpc1460",
			`\(\*http2Client_grpc1460\)\.keepalive\(.* \[chan receive\]`,
			`\(\*http2Client_grpc1460\)\.NewStream\(.* \[sync\.Mutex\.Lock\]`,
		),
		makeFlakyTest("Grpc3017",
			// grpc/3017 involves a goroutine leak that also simultaneously engages many GC assists.
			`Grpc3017\.func2\(.* \[chan receive\]`,
			`Grpc3017\.func2\.1\(.* \[sync\.Mutex\.Lock\]`,
			`\(\*lbCacheClientConn_grpc3017\)\.RemoveSubConn\.func1\(.* \[sync\.Mutex\.Lock\]`,
		),
		makeFlakyTest("Hugo3251",
			`Hugo3251\.func2\(.* \[sync\.WaitGroup\.Wait\]`,
			`Hugo3251\.func2\.1\(.* \[sync\.Mutex\.Lock\]`,
			`Hugo3251\.func2\.1\(.* \[sync\.RWMutex\.RLock\]`,
		),
		makeFlakyTest("Istio16224",
			`Istio16224\.func2\(.* \[sync\.Mutex\.Lock\]`,
			`\(\*controller_istio16224\)\.Run\(.* \[chan send\]`,
			`\(\*controller_istio16224\)\.Run\(.* \[chan receive\]`,
		),
		makeFlakyTest("Istio17860",
			`\(\*agent_istio17860\)\.runWait\(.* \[chan send\]`,
		),
		makeFlakyTest("Istio18454",
			`\(\*Worker_istio18454\)\.Start\.func1\(.* \[chan receive\]`,
			`\(\*Worker_istio18454\)\.Start\.func1\(.* \[chan send\]`,
		),
		// NOTE(vsaioc):
		// Kubernetes/1321 is excluded due to a race condition in the original program
		// that may, in extremely rare cases, lead to nil pointer dereference crashes.
		// (Reproducible even with regular GC). Only kept here for posterity.
		//
		// makeTest(testCase{name: "Kubernetes1321"},
		// 	`NewMux_kubernetes1321\.gowrap1\(.* \[chan send\]`,
		// 	`testMuxWatcherClose_kubernetes1321\(.* \[sync\.Mutex\.Lock\]`),
		makeTest("Kubernetes5316",
			`finishRequest_kubernetes5316\.func1\(.* \[chan send\]`,
		),
		makeFlakyTest("Kubernetes6632",
			`\(\*idleAwareFramer_kubernetes6632\)\.monitor\(.* \[sync\.Mutex\.Lock\]`,
			`\(\*idleAwareFramer_kubernetes6632\)\.WriteFrame\(.* \[chan send\]`,
		),
		makeFlakyTest("Kubernetes10182",
			`\(\*statusManager_kubernetes10182\)\.Start\.func1\(.* \[sync\.Mutex\.Lock\]`,
			`\(\*statusManager_kubernetes10182\)\.SetPodStatus\(.* \[chan send\]`,
		),
		makeFlakyTest("Kubernetes11298",
			`After_kubernetes11298\.func1\(.* \[chan receive\]`,
			`After_kubernetes11298\.func1\(.* \[sync\.Cond\.Wait\]`,
			`Kubernetes11298\.func2\(.* \[chan receive\]`,
		),
		makeFlakyTest("Kubernetes13135",
			`Util_kubernetes13135\(.* \[sync\.Mutex\.Lock\]`,
			`\(\*WatchCache_kubernetes13135\)\.Add\(.* \[sync\.Mutex\.Lock\]`,
		),
		makeTest("Kubernetes25331",
			`\(\*watchChan_kubernetes25331\)\.run\(.* \[chan send\]`,
		),
		makeFlakyTest("Kubernetes26980",
			`Kubernetes26980\.func2\(.* \[chan receive\]`,
			`Kubernetes26980\.func2\.1\(.* \[sync\.Mutex\.Lock\]`,
			`\(\*processorListener_kubernetes26980\)\.pop\(.* \[chan receive\]`,
		),
		makeFlakyTest("Kubernetes30872",
			`\(\*DelayingDeliverer_kubernetes30872\)\.StartWithHandler\.func1\(.* \[sync\.Mutex\.Lock\]`,
			`\(\*Controller_kubernetes30872\)\.Run\(.* \[sync\.Mutex\.Lock\]`,
			`\(\*NamespaceController_kubernetes30872\)\.Run\.func1\(.* \[sync\.Mutex\.Lock\]`,
		),
		makeTest("Kubernetes38669",
			`\(\*cacheWatcher_kubernetes38669\)\.process\(.* \[chan send\]`,
		),
		makeFlakyTest("Kubernetes58107",
			`\(\*ResourceQuotaController_kubernetes58107\)\.worker\(.* \[sync\.Cond\.Wait\]`,
			`\(\*ResourceQuotaController_kubernetes58107\)\.worker\(.* \[sync\.RWMutex\.RLock\]`,
			`\(\*ResourceQuotaController_kubernetes58107\)\.Sync\(.* \[sync\.RWMutex\.Lock\]`,
		),
		makeFlakyTest("Kubernetes62464",
			`\(\*manager_kubernetes62464\)\.reconcileState\(.* \[sync\.RWMutex\.RLock\]`,
			`\(\*staticPolicy_kubernetes62464\)\.RemoveContainer\(.* \[sync\.(RW)?Mutex\.Lock\]`,
		),
		makeFlakyTest("Kubernetes70277",
			`Kubernetes70277\.func2\(.* \[chan receive\]`,
		),
		makeFlakyTest("Moby4951",
			`\(\*DeviceSet_moby4951\)\.DeleteDevice\(.* \[sync\.Mutex\.Lock\]`,
		),
		makeTest("Moby7559",
			`\(\*UDPProxy_moby7559\)\.Run\(.* \[sync\.Mutex\.Lock\]`,
		),
		makeTest("Moby17176",
			`testDevmapperLockReleasedDeviceDeletion_moby17176\.func1\(.* \[sync\.Mutex\.Lock\]`,
		),
		makeFlakyTest("Moby21233",
			`\(\*Transfer_moby21233\)\.Watch\.func1\(.* \[chan send\]`,
			`\(\*Transfer_moby21233\)\.Watch\.func1\(.* \[select\]`,
			`testTransfer_moby21233\(.* \[chan receive\]`,
		),
		makeTest("Moby25348",
			`\(\*Manager_moby25348\)\.init\(.* \[sync\.WaitGroup\.Wait\]`,
		),
		makeFlakyTest("Moby27782",
			`\(\*JSONFileLogger_moby27782\)\.readLogs\(.* \[sync\.Cond\.Wait\]`,
			`\(\*Watcher_moby27782\)\.readEvents\(.* \[select\]`,
		),
		makeFlakyTest("Moby28462",
			`monitor_moby28462\(.* \[sync\.Mutex\.Lock\]`,
			`\(\*Daemon_moby28462\)\.StateChanged\(.* \[chan send\]`,
		),
		makeTest("Moby29733",
			`Moby29733\.func2\(.* \[chan receive\]`,
			`testActive_moby29733\.func1\(.* \[sync\.Cond\.Wait\]`,
		),
		makeTest("Moby30408",
			`Moby30408\.func2\(.* \[chan receive\]`,
			`testActive_moby30408\.func1\(.* \[sync\.Cond\.Wait\]`,
		),
		makeFlakyTest("Moby33781",
			`monitor_moby33781\.func1\(.* \[chan send\]`,
		),
		makeFlakyTest("Moby36114",
			`\(\*serviceVM_moby36114\)\.hotAddVHDsAtStart\(.* \[sync\.Mutex\.Lock\]`,
		),
		makeFlakyTest("Serving2137",
			`\(\*Breaker_serving2137\)\.concurrentRequest\.func1\(.* \[chan send\]`,
			`\(\*Breaker_serving2137\)\.concurrentRequest\.func1\(.* \[sync\.Mutex\.Lock\]`,
			`Serving2137\.func2\(.* \[chan receive\]`,
		),
		makeTest("Syncthing4829",
			`Syncthing4829\.func2\(.* \[sync\.RWMutex\.RLock\]`,
		),
		makeTest("Syncthing5795",
			`\(\*rawConnection_syncthing5795\)\.Start\.func1.* \[chan receive\]`,
			`Syncthing5795\.func2.* \[chan receive\]`,
		),
	}

	// Combine all test cases into a single list.
	testCases := append(microTests, stressTestCases...)
	testCases = append(testCases, patternTestCases...)
	testCases = append(testCases, gokerTestCases...)

	// Test cases must not panic or cause fatal exceptions.
	failStates := regexp.MustCompile(`fatal|panic`)

	// Build the test program once.
	exe, err := buildTestProg(t, "testgoroutineleakgc")
	if err != nil {
		t.Fatal(fmt.Sprintf("building testgoroutineleakgc failed: %v", err))
	}

	for _, tcase := range testCases {
		t.Run(tcase.name, func(t *testing.T) {
			t.Parallel()

			cmdEnv := []string{
				"GODEBUG=asyncpreemptoff=1",
				"GOEXPERIMENT=greenteagc",
			}

			if tcase.simple {
				// If the test is simple, set GOMAXPROCS=1 in order to better
				// control the behavior of the scheduler.
				cmdEnv = append(cmdEnv, "GOMAXPROCS=1")
			}

			// Run program and get output trace.
			output := runBuiltTestProg(t, exe, tcase.name, cmdEnv...)
			if len(output) == 0 {
				t.Fatalf("Test produced no output. Is the goroutine leak profile collected?")
			}

			// Zero tolerance policy for fatal exceptions or panics.
			if failStates.MatchString(output) {
				t.Errorf("unexpected fatal exception or panic!\noutput:\n%s\n\n", output)
				return
			}

			// Extract all the goroutine leaks
			foundLeaks := extractLeaks(output)

			// If the test case was not expected to produce leaks, but some were reported,
			// stop the test immediately. Zero tolerance policy for false positives.
			if len(tcase.expectedLeaks)+len(tcase.flakyLeaks) == 0 && len(foundLeaks) > 0 {
				t.Errorf("output:\n%s\n\ngoroutines leaks detected in case with no leaks", output)
			}

			unexpectedLeaks := make([]string, 0, len(foundLeaks))

			// Parse every leak and check if it is expected (maybe as a flaky leak).
		LEAKS:
			for _, leak := range foundLeaks {
				// Check if the leak is expected.
				// If it is, check whether it has been encountered before.
				var foundNew bool
				var leakPattern *regexp.Regexp

				for expectedLeak, ok := range tcase.expectedLeaks {
					if expectedLeak.MatchString(leak) {
						if !ok {
							foundNew = true
						}

						leakPattern = expectedLeak
						break
					}
				}

				if foundNew {
					// Only bother writing if we found a new leak.
					tcase.expectedLeaks[leakPattern] = true
				}

				if leakPattern == nil {
					// We are dealing with a leak not marked as expected.
					// Check if it is a flaky leak.
					for flakyLeak := range tcase.flakyLeaks {
						if flakyLeak.MatchString(leak) {
							// The leak is flaky. Carry on to the next line.
							continue LEAKS
						}
					}

					unexpectedLeaks = append(unexpectedLeaks, leak)
				}
			}

			missingLeakStrs := make([]string, 0, len(tcase.expectedLeaks))
			for expectedLeak, found := range tcase.expectedLeaks {
				if !found {
					missingLeakStrs = append(missingLeakStrs, expectedLeak.String())
				}
			}

			var errors []error
			if len(unexpectedLeaks) > 0 {
				errors = append(errors, fmt.Errorf("unexpected goroutine leaks:\n%s\n", strings.Join(unexpectedLeaks, "\n")))
			}
			if len(missingLeakStrs) > 0 {
				errors = append(errors, fmt.Errorf("missing expected leaks:\n%s\n", strings.Join(missingLeakStrs, ", ")))
			}
			if len(errors) > 0 {
				t.Fatalf("Failed with the following errors:\n%s\n\noutput:\n%s", errors, output)
			}
		})
	}
}
