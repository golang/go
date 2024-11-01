// Copyright 2011 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

package runtime_test

import (
	"fmt"
	"internal/testenv"
	"internal/weak"
	"math/bits"
	"math/rand"
	"os"
	"reflect"
	"runtime"
	"runtime/debug"
	"slices"
	"strings"
	"sync"
	"sync/atomic"
	"testing"
	"time"
	"unsafe"
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
			// Initialize a new byte slice with pseduo-random data.
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
	// working correctly, then weak.Strong will block, so we need to make sure something
	// prevents the GC from continuing to spin.
	done := make(chan struct{})
	go func() {
		runtime.GC()
		done <- struct{}{}
	}()
	go func() {
		time.Sleep(100 * time.Millisecond)

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
			wp.Strong()
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
