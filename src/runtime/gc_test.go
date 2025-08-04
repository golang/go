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
	"strconv"
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
		repetitions   int
		expectedLeaks map[*regexp.Regexp]bool

		// flakyLeaks are goroutine leaks that are too flaky to be reliably detected.
		// Still, they might pop up every once in a while.
		// If these occur, do not fail the test due to unexpected leaks.
		flakyLeaks map[*regexp.Regexp]struct{}
	}

	// makeTest is a short-hand for creating test cases.
	// Each of the leaks in the list is identified by a regular expression.
	//
	// If a leak is the string "FLAKY", it notifies makeTest that any remaining
	// leak patterns should be added to the flakyLeaks map.
	makeTest := func(
		cfg testCase,
		leaks ...string) testCase {
		tc := testCase{
			name:          cfg.name,
			expectedLeaks: make(map[*regexp.Regexp]bool, len(leaks)),
			flakyLeaks:    make(map[*regexp.Regexp]struct{}, len(leaks)),
		}
		// Default to 1 repetition if not specified.
		// One extra rep for configured tests is irrelevant.
		tc.repetitions = cfg.repetitions | 1

		const (
			EXPECTED int = iota
			FLAKY
		)

		mode := EXPECTED
		for _, leak := range leaks {
			if leak == "FLAKY" {
				mode = FLAKY
				continue
			}

			switch mode {
			case EXPECTED:
				tc.expectedLeaks[regexp.MustCompile(leak)] = false
			case FLAKY:
				tc.flakyLeaks[regexp.MustCompile(leak)] = struct{}{}
			}
		}
		return tc
	}

	// Micro tests involve very simple leaks for each type of concurrency primitive operation.
	microTests := []testCase{
		makeTest(testCase{name: "NilRecv"}, `\[chan receive \(nil chan\)\]`),
		makeTest(testCase{name: "NilSend"}, `\[chan send \(nil chan\)\]`),
		makeTest(testCase{name: "SelectNoCases"}, `\[select \(no cases\)\]`),
		makeTest(testCase{name: "ChanRecv"}, `\[chan receive\]`),
		makeTest(testCase{name: "ChanSend"}, `\[chan send\]`),
		makeTest(testCase{name: "Select"}, `\[select\]`),
		makeTest(testCase{name: "WaitGroup"}, `\[sync\.WaitGroup\.Wait\]`),
		makeTest(testCase{name: "MutexStack"}, `\[sync\.Mutex\.Lock\]`),
		makeTest(testCase{name: "MutexHeap"}, `\[sync\.Mutex\.Lock\]`),
		makeTest(testCase{name: "Cond"}, `\[sync\.Cond\.Wait\]`),
		makeTest(testCase{name: "RWMutexRLock"}, `\[sync\.RWMutex\.RLock\]`),
		makeTest(testCase{name: "RWMutexLock"}, `\[sync\.(RW)?Mutex\.Lock\]`),
		makeTest(testCase{name: "Mixed"}, `\[sync\.WaitGroup\.Wait\]`, `\[chan send\]`),
		makeTest(testCase{name: "NoLeakGlobal"}),
	}

	// Common goroutine leak patterns.
	// Extracted from "Unveiling and Vanquishing Goroutine Leaks in Enterprise Microservices: A Dynamic Analysis Approach"
	// doi:10.1109/CGO57630.2024.10444835
	patternTestCases := []testCase{
		makeTest(testCase{name: "NoCloseRange"},
			`main\.NoCloseRange\.gowrap1 .* \[chan send\]`,
			`main\.noCloseRange\.func1 .* \[chan receive\]`),
		makeTest(testCase{name: "MethodContractViolation"},
			`main\.worker\.Start\.func1 .* \[select\]`),
		makeTest(testCase{name: "DoubleSend"},
			`main\.DoubleSend\.func3 .* \[chan send\]`),
		makeTest(testCase{name: "EarlyReturn"},
			`main\.earlyReturn\.func1 .* \[chan send\]`),
		makeTest(testCase{name: "NCastLeak"},
			`main\.nCastLeak\.func1 .* \[chan send\]`,
			`main\.NCastLeak\.func2 .* \[chan receive\]`),
		makeTest(testCase{name: "Timeout"},
			`main\.timeout\.func1 .* \[chan send\]`),
	}

	// GoKer tests from "GoBench: A Benchmark Suite of Real-World Go Concurrency Bugs".
	// White paper found at https://lujie.ac.cn/files/papers/GoBench.pdf
	// doi:10.1109/CGO51591.2021.9370317.
	//
	// This list is curated for tests that are not excessively flaky.
	gokerTestCases := []testCase{
		makeTest(testCase{name: "Cockroach584"},
			`main\.Cockroach584\.func2\.1 .* \[sync\.Mutex\.Lock\]`),
		makeTest(testCase{name: "Cockroach1055"},
			`main\.Cockroach1055\.func2 .* \[chan receive\]`,
			`main\.Cockroach1055\.func2\.1 .* \[chan receive\]`,
			`main\.Cockroach1055\.func2\.1 .* \[sync\.Mutex\.Lock\]`,
			`main\.Cockroach1055\.func2\.2 .* \[sync\.WaitGroup\.Wait\]`),
		makeTest(testCase{name: "Cockroach1462"},
			`main\.\(\*Stopper_cockroach1462\)\.RunWorker\.func1 .* \[chan send\]`,
			`main\.Cockroach1462\.func2 .* \[sync\.WaitGroup\.Wait\]`),
		makeTest(testCase{name: "Cockroach2448"},
			`main\.Cockroach2448\.func2\.gowrap1 .* \[select\]`,
			`main\.Cockroach2448\.func2\.gowrap2 .* \[select\]`),
		makeTest(testCase{name: "Cockroach3710"},
			`main\.Cockroach3710\.func2\.gowrap1 .* \[sync\.RWMutex\.RLock\]`,
			`main\.\(\*Store_cockroach3710\)\.processRaft\.func1 .* \[sync\.RWMutex\.Lock\]`),
		makeTest(testCase{name: "Cockroach6181", repetitions: 50},
			`main\.testRangeCacheCoalescedRequests_cockroach6181 .* \[sync\.WaitGroup\.Wait\]`,
			`main\.testRangeCacheCoalescedRequests_cockroach6181\.func1\.1 .* \[sync\.Mutex\.Lock\]`,
			`main\.testRangeCacheCoalescedRequests_cockroach6181\.func1\.1 .* \[sync\.RWMutex\.Lock\]`,
			`main\.testRangeCacheCoalescedRequests_cockroach6181\.func1\.1 .* \[sync\.RWMutex\.RLock\]`),
		makeTest(testCase{name: "Cockroach7504", repetitions: 100},
			`main\.Cockroach7504\.func2\.1 .* \[sync\.Mutex\.Lock\]`,
			`main\.Cockroach7504\.func2\.2 .* \[sync\.Mutex\.Lock\]`),
		makeTest(testCase{name: "Cockroach9935"},
			`main\.Cockroach9935\.func2\.gowrap1 .* \[sync\.Mutex\.Lock\]`),
		makeTest(testCase{name: "Cockroach10214"},
			`main\.Cockroach10214\.func2\.1 .* \[sync\.Mutex\.Lock\]`,
			`main\.Cockroach10214\.func2\.2 .* \[sync\.Mutex\.Lock\]`),
		makeTest(testCase{name: "Cockroach10790"},
			`main\.\(\*Replica_cockroach10790\)\.beginCmds\.func1 .* \[chan receive\]`),
		makeTest(testCase{name: "Cockroach13197"},
			`main\.\(\*DB_cockroach13197\)\.begin\.gowrap1 .* \[chan receive\]`),
		makeTest(testCase{name: "Cockroach13755"},
			`main\.\(\*Rows_cockroach13755\)\.initContextClose\.gowrap1 .* \[chan receive\]`),
		makeTest(testCase{name: "Cockroach16167"},
			`main\.Cockroach16167\.func2 .* \[sync\.RWMutex\.RLock\]`,
			`main\.Cockroach16167\.func2\.gowrap1 .* \[sync\.RWMutex\.Lock\]`),
		makeTest(testCase{name: "Cockroach10790"},
			`main\.\(\*Replica_cockroach10790\)\.beginCmds\.func1 .* \[chan receive\]`),
		makeTest(testCase{name: "Cockroach13197"},
			`main\.\(\*DB_cockroach13197\)\.begin\.gowrap1 .* \[chan receive\]`),
		makeTest(testCase{name: "Cockroach13755"},
			`main\.\(\*Rows_cockroach13755\)\.initContextClose\.gowrap1 .* \[chan receive\]`),
		makeTest(testCase{name: "Cockroach16167"},
			`main\.Cockroach16167\.func2 .* \[sync\.RWMutex\.RLock\]`,
			`main\.Cockroach16167\.func2\.gowrap1 .* \[sync\.RWMutex\.Lock\]`),
		makeTest(testCase{name: "Cockroach18101"},
			`main\.restore_cockroach18101\.func1 .* \[chan send\]`),
		makeTest(testCase{name: "Cockroach24808"},
			`main\.Cockroach24808\.func2 .* \[chan send\]`),
		makeTest(testCase{name: "Cockroach25456"},
			`main\.Cockroach25456\.func2 .* \[chan receive\]`),
		makeTest(testCase{name: "Cockroach35073"},
			`main\.Cockroach35073\.func2.1 .* \[chan send\]`,
			`main\.Cockroach35073\.func2 .* \[chan send\]`),
		makeTest(testCase{name: "Cockroach35931"},
			`main\.Cockroach35931\.func2 .* \[chan send\]`),
		makeTest(testCase{name: "Etcd5509"},
			`main\.Etcd5509\.func2 .* \[sync\.RWMutex\.Lock\]`),
		makeTest(testCase{name: "Etcd6857"},
			`main\.Etcd6857\.func2\.gowrap2 .* \[chan send\]`),
		makeTest(testCase{name: "Etcd6873"},
			`main\.Etcd6873\.func2\.gowrap1 .* \[chan receive\]`,
			`main\.newWatchBroadcasts_etcd6873\.func1 .* \[sync\.Mutex\.Lock\]`),
		makeTest(testCase{name: "Etcd7492"},
			`main\.Etcd7492\.func2 .* \[sync\.WaitGroup\.Wait\]`,
			`main\.Etcd7492\.func2\.1 .* \[chan send\]`,
			`main\.NewSimpleTokenTTLKeeper_etcd7492\.gowrap1 .* \[sync\.Mutex\.Lock\]`),
		makeTest(testCase{name: "Etcd7902"},
			`main\.doRounds_etcd7902\.gowrap1 .* \[chan receive\]`,
			`main\.doRounds_etcd7902\.gowrap1 .* \[sync\.Mutex\.Lock\]`,
			`main\.runElectionFunc_etcd7902 .* \[sync\.WaitGroup\.Wait\]`),
		makeTest(testCase{name: "Etcd10492"},
			`main\.Etcd10492\.func2 .* \[sync\.Mutex\.Lock\]`),
		makeTest(testCase{name: "Grpc660"},
			`main\.\(\*benchmarkClient_grpc660\)\.doCloseLoopUnary\.func1 .* \[chan send\]`),
		makeTest(testCase{name: "Grpc795"},
			`main\.\(\*test_grpc795\)\.startServer\.gowrap1 .* \[sync\.Mutex\.Lock\]`,
			`main\.testServerGracefulStopIdempotent_grpc795 .* \[sync\.Mutex\.Lock\]`),
		makeTest(testCase{name: "Grpc862"},
			`main\.DialContext_grpc862\.func2 .* \[chan receive\]`),
		makeTest(testCase{name: "Grpc1275"},
			`main\.testInflightStreamClosing_grpc1275\.func1 .* \[chan receive\]`),
		makeTest(testCase{name: "Grpc1424"},
			`main\.DialContext_grpc1424\.func1 .* \[chan receive\]`),
		makeTest(testCase{name: "Grpc1460"},
			`main\.Grpc1460\.func2\.gowrap1 .* \[chan receive\]`,
			`main\.Grpc1460\.func2\.gowrap2 .* \[sync\.Mutex\.Lock\]`),
		makeTest(testCase{name: "Grpc3017", repetitions: 50},
			// grpc/3017 involves a goroutine leak that also simultaneously engages many GC assists.
			// Testing runtime behaviour when pivoting between regular and goroutine leak detection modes.
			`main\.Grpc3017\.func2 .* \[chan receive\]`,
			`main\.Grpc3017\.func2\.1 .* \[sync\.Mutex\.Lock\]`,
			`main\.\(\*lbCacheClientConn_grpc3017\)\.RemoveSubConn\.func1 .* \[sync\.Mutex\.Lock\]`),
		makeTest(testCase{name: "Hugo3251", repetitions: 20},
			`main\.Hugo3251\.func2 .* \[sync\.WaitGroup\.Wait\]`,
			`main\.Hugo3251\.func2\.gowrap1 .* \[sync\.Mutex\.Lock\]`,
			`main\.Hugo3251\.func2\.gowrap1 .* \[sync\.RWMutex\.RLock\]`),
		makeTest(testCase{name: "Hugo5379"},
			`main\.\(\*Page_hugo5379\)\.initContent\.func1\.1 .* \[sync\.Mutex\.Lock\]`,
			`main\.\(\*Site_hugo5379\)\.renderPages\.gowrap1 .* \[sync\.Mutex\.Lock\]`,
			`main\.Hugo5379\.func2 .* \[sync\.WaitGroup\.Wait\]`),
		makeTest(testCase{name: "Istio16224"},
			`main\.Istio16224\.func2 .* \[sync\.Mutex\.Lock\]`,
			`main\.Istio16224\.func2\.gowrap1 .* \[chan send\]`,
			// This is also a leak, but it is too flaky to be reliably detected.
			`FLAKY`,
			`main\.Istio16224\.func2\.gowrap1 .* \[chan receive\]`),
		makeTest(testCase{name: "Istio17860"},
			`main\.\(\*agent_istio17860\)\.Restart\.gowrap2 .* \[chan send\]`),
		makeTest(testCase{name: "Istio18454"},
			`main\.\(\*Worker_istio18454\)\.Start\.func1 .* \[chan receive\]`,
			`main\.\(\*Worker_istio18454\)\.Start\.func1 .* \[chan send\]`),
		makeTest(testCase{name: "Kubernetes1321"},
			`main\.NewMux_kubernetes1321\.gowrap1 .* \[chan send\]`,
			`main\.testMuxWatcherClose_kubernetes1321 .* \[sync\.Mutex\.Lock\]`),
		makeTest(testCase{name: "Kubernetes5316"},
			`main\.finishRequest_kubernetes5316\.func1 .* \[chan send\]`),
		makeTest(testCase{name: "Kubernetes6632"},
			`main\.Kubernetes6632\.func2\.gowrap1 .* \[sync\.Mutex\.Lock\]`,
			`main\.Kubernetes6632\.func2\.gowrap2 .* \[chan send\]`),
		makeTest(testCase{name: "Kubernetes10182"},
			`main\.\(\*statusManager_kubernetes10182\)\.Start\.func1 .* \[sync\.Mutex\.Lock\]`,
			`main\.Kubernetes10182\.func2\.gowrap2 .* \[chan send\]`,
			`main\.Kubernetes10182\.func2\.gowrap3 .* \[chan send\]`),
		makeTest(testCase{name: "Kubernetes11298"},
			`main\.After_kubernetes11298\.func1 .* \[chan receive\]`,
			`main\.After_kubernetes11298\.func1 .* \[sync\.Cond\.Wait\]`,
			`main\.Kubernetes11298\.func2 .* \[chan receive\]`),
		makeTest(testCase{name: "Kubernetes13135"},
			`main\.Kubernetes13135\.func2 .* \[sync\.WaitGroup\.Wait\]`),
		makeTest(testCase{name: "Kubernetes25331"},
			`main\.Kubernetes25331\.func2\.gowrap1 .* \[chan send\]`),
		makeTest(testCase{name: "Kubernetes26980"},
			`main\.Kubernetes26980\.func2 .* \[chan receive\]`,
			`main\.Kubernetes26980\.func2\.1 .* \[sync\.Mutex\.Lock\]`,
			`main\.Kubernetes26980\.func2\.gowrap2 .* \[chan receive\]`),
		makeTest(testCase{name: "Kubernetes30872"},
			`main\.\(\*DelayingDeliverer_kubernetes30872\)\.StartWithHandler\.func1 .* \[sync\.Mutex\.Lock\]`,
			`main\.\(\*federatedInformerImpl_kubernetes30872\)\.Start\.gowrap2 .* \[sync\.Mutex\.Lock\]`,
			`main\.\(\*NamespaceController_kubernetes30872\)\.Run\.func1 .* \[sync\.Mutex\.Lock\]`),
		makeTest(testCase{name: "Kubernetes38669"},
			`main\.newCacheWatcher_kubernetes38669\.gowrap1 .* \[chan send\]`),
		makeTest(testCase{name: "Kubernetes58107"},
			`main\.\(\*ResourceQuotaController_kubernetes58107\)\.Run\.gowrap1 .* \[sync\.Cond\.Wait\]`,
			`main\.\(\*ResourceQuotaController_kubernetes58107\)\.Run\.gowrap1 .* \[sync\.RWMutex\.RLock\]`,
			`main\.\(\*ResourceQuotaController_kubernetes58107\)\.Run\.gowrap2 .* \[sync\.Cond\.Wait\]`,
			`main\.\(\*ResourceQuotaController_kubernetes58107\)\.Run\.gowrap2 .* \[sync\.RWMutex\.RLock\]`,
			`main\.startResourceQuotaController_kubernetes58107\.gowrap2 .* \[sync\.RWMutex\.Lock\]`),
		makeTest(testCase{name: "Kubernetes62464"},
			`main\.Kubernetes62464\.func2\.gowrap1 .* \[sync\.RWMutex\.RLock\]`,
			`main\.Kubernetes62464\.func2\.gowrap2 .* \[sync\.RWMutex\.Lock\]`),
		makeTest(testCase{name: "Kubernetes70277"},
			`main\.Kubernetes70277\.func2 .* \[chan receive\]`),
		makeTest(testCase{name: "Moby4395"},
			`main\.Go_moby4395\.func1 .* \[chan send\]`),
		makeTest(testCase{name: "Moby4951"},
			`main\.Moby4951\.func2\.gowrap1 .* \[sync\.Mutex\.Lock\]`,
			`main\.Moby4951\.func2\.gowrap2 .* \[sync\.Mutex\.Lock\]`),
		makeTest(testCase{name: "Moby7559"},
			`main\.Moby7559\.func2\.gowrap1 .* \[sync\.Mutex\.Lock\]`),
		makeTest(testCase{name: "Moby17176"},
			`main\.testDevmapperLockReleasedDeviceDeletion_moby17176\.func1 .* \[sync\.Mutex\.Lock\]`),
		makeTest(testCase{name: "Moby21233"},
			`main\.\(\*Transfer_moby21233\)\.Watch\.func1 .* \[chan send\]`,
			`main\.\(\*Transfer_moby21233\)\.Watch\.func1 .* \[select\]`,
			`main\.testTransfer_moby21233 .* \[chan receive\]`),
		makeTest(testCase{name: "Moby25348"},
			`main\.Moby25348\.func2\.gowrap1 .* \[sync\.WaitGroup\.Wait\]`),
		makeTest(testCase{name: "Moby27782"},
			`main\.\(\*JSONFileLogger_moby27782\)\.ReadLogs\.gowrap1 .* \[sync\.Cond\.Wait\]`,
			`main\.NewWatcher_moby27782\.gowrap1 .* \[select\]`),
		makeTest(testCase{name: "Moby28462"},
			`main\.Moby28462\.func2\.gowrap1 .* \[sync\.Mutex\.Lock\]`,
			`main\.Moby28462\.func2\.gowrap2 .* \[chan send\]`),
		makeTest(testCase{name: "Moby29733"},
			`main\.Moby29733\.func2 .* \[chan receive\]`,
			`main\.testActive_moby29733\.func1 .* \[sync\.Cond\.Wait\]`),
		makeTest(testCase{name: "Moby30408"},
			`main\.Moby30408\.func2 .* \[chan receive\]`,
			`main\.testActive_moby30408\.func1 .* \[sync\.Cond\.Wait\]`),
		makeTest(testCase{name: "Moby33781"},
			`main\.monitor_moby33781\.func1 .* \[chan send\]`),
		makeTest(testCase{name: "Moby36114"},
			`main\.Moby36114\.func2\.gowrap1 .* \[sync\.Mutex\.Lock\]`),
		makeTest(testCase{name: "Serving2137"},
			`main\.\(\*Breaker_serving2137\)\.concurrentRequest\.func1 .* \[chan send\]`,
			`main\.\(\*Breaker_serving2137\)\.concurrentRequest\.func1 .* \[sync\.Mutex\.Lock\]`,
			`main\.Serving2137\.func2 .* \[chan receive\]`),
		makeTest(testCase{name: "Syncthing4829"},
			`main\.Syncthing4829\.func2 .* \[sync\.RWMutex\.RLock\]`),
		makeTest(testCase{name: "Syncthing5795"},
			`main\.\(\*rawConnection_syncthing5795\)\.Start\.func1 .* \[chan receive\]`,
			`main\.Syncthing5795\.func2 .* \[chan receive\]`),
	}

	// Combine all test cases into a single list.
	testCases := append(microTests, patternTestCases...)
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
			// Run tests in parallel.
			t.Parallel()

			// Default to 1 repetition if not specified.
			// One extra rep for tests with a specified number of repetitions
			// is irrelevant.
			repetitions := tcase.repetitions | 1

			// Output trace. Aggregated across all repetitions.
			var output string
			// Output and trace are protected by separate mutexes to reduce contention.
			var outputMu sync.Mutex
			var traceMu sync.RWMutex
			// Wait group coordinates across all repetitions.
			var wg sync.WaitGroup

			wg.Add(repetitions)
			for i := 0; i < repetitions; i++ {
				go func() {
					defer wg.Done()

					// FIXME: Use GODEBUG flag only temporarily until we can use pprof/goroutineleaks.
					repOutput := runBuiltTestProg(t, exe, tcase.name, "GODEBUG=gctrace=1,gcgoroutineleaks=1")

					// If the test case was not expected to produce leaks, but some were reported,
					// stop the test immediately. Zero tolerance policy for false positives.
					if len(tcase.expectedLeaks)+len(tcase.flakyLeaks) == 0 && strings.Contains(repOutput, "goroutine leak!") {
						t.Errorf("output:\n%s\n\ngoroutines leaks detected in case with no leaks", repOutput)
					}

					// Zero tolerance policy for fatal exceptions or panics.
					if failStates.MatchString(repOutput) {
						t.Errorf("output:\n%s\n\nunexpected fatal exception or panic", repOutput)
					}

					// Parse the output line by line and look for the `goroutine leak!` message.
				LINES:
					for _, line := range strings.Split(repOutput, "\n") {
						// We are not interested in anything else.
						if !strings.Contains(line, "goroutine leak!") {
							continue
						}

						// Check if the leak is expected.
						// If it is, check whether it has been encountered before.
						var foundNew bool
						var leakPattern *regexp.Regexp
						traceMu.RLock()
						for expectedLeak, ok := range tcase.expectedLeaks {
							if expectedLeak.MatchString(line) {
								if !ok {
									foundNew = true
								}

								leakPattern = expectedLeak
								break
							}
						}
						traceMu.RUnlock()

						if foundNew {
							// Only bother writing if we found a new leak.
							traceMu.Lock()
							tcase.expectedLeaks[leakPattern] = true
							traceMu.Unlock()
						}

						if leakPattern == nil {
							// We are dealing with a leak not marked as expected.
							// Check if it is a flaky leak.
							for flakyLeak := range tcase.flakyLeaks {
								if flakyLeak.MatchString(line) {
									// The leak is flaky. Carry on to the next line.
									continue LINES
								}
							}

							t.Errorf("output:\n%s\n\nunexpected goroutine leak: %s", repOutput, line)
						}
					}

					outputMu.Lock()
					output += "\nRepetition " + strconv.Itoa(i) + ":\n" + repOutput + "\n--------------------------\n"
					outputMu.Unlock()
				}()
			}

			// Coordinate across all repetitions.
			wg.Wait()
			missingLeakStrs := make([]string, 0, len(tcase.expectedLeaks))
			for expectedLeak, found := range tcase.expectedLeaks {
				if !found {
					missingLeakStrs = append(missingLeakStrs, expectedLeak.String())
				}
			}

			if len(missingLeakStrs) > 0 {
				t.Fatalf("output:\n%s\n\nnot enough goroutines leaks detected. Missing:\n%s", output, strings.Join(missingLeakStrs, ", "))
			}
		})
	}
}
