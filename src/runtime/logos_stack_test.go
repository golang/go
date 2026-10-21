// Copyright 2026 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

package runtime_test

import (
	"internal/goexperiment"
	"internal/race"
	"math"
	"runtime"
	"runtime/debug"
	"sync"
	"sync/atomic"
	"testing"
	"time"
)

// M17-S2 attack tests: GOEXPERIMENT=logosstack gives every go statement
// its own initial goroutine stack size, the moving average of the final
// stack sizes of the goroutines it created. Red before the slice lands:
// the experiment flag, the table, and the hooks do not exist.

const logosStackTarget = 64 << 10

func skipUnlessLogosStack(t *testing.T) {
	t.Helper()
	if !goexperiment.LogosStack {
		t.Skip("needs GOEXPERIMENT=logosstack")
	}
}

// logosStackStartIs reports whether a goroutine that observed a stack of
// got bytes at its first instruction started with want bytes. Under the
// race detector the first frames of a goroutine can outgrow the smallest
// stacks before the observation, so there it reports whether got is want
// or at most three doublings above it.
func logosStackStartIs(got, want int) bool {
	if race.Enabled {
		return got >= want && got <= want<<3
	}
	return got == want
}

// logosStackSpawn and logosStackSpawnOther are two go statements with
// identical bodies. They must not inline, or each caller would get its
// own program counter, and they must not use wg.Go, which would give
// every spawn the one go statement inside package sync.
//
//go:noinline
func logosStackSpawn(wg *sync.WaitGroup, body func()) {
	wg.Add(1)
	go func() {
		defer wg.Done()
		body()
	}()
}

//go:noinline
func logosStackSpawnOther(wg *sync.WaitGroup, body func()) {
	wg.Add(1)
	go func() {
		defer wg.Done()
		body()
	}()
}

const logosStackPad = 64 * 8

// logosStackUse recurses until it has used about bytes of stack.
//
//go:noinline
func logosStackUse(bytes int) uint64 {
	if bytes <= 0 {
		return 1
	}
	var pad [64]uint64
	pad[bytes%len(pad)] = 1
	return pad[0] + logosStackUse(bytes-logosStackPad)
}

// logosStackStarts runs n goroutines through spawn and reports the stack
// size each one starts with and the go statement pc they share.
func logosStackStarts(spawn func(*sync.WaitGroup, func()), n int) (starts []int, pc uintptr) {
	var wg sync.WaitGroup
	var site atomic.Uintptr
	sizes := make(chan int, n)
	for range n {
		spawn(&wg, func() {
			sizes <- runtime.LogosCurStackSize()
			site.Store(runtime.LogosGoPC())
		})
	}
	wg.Wait()
	close(sizes)
	for size := range sizes {
		starts = append(starts, size)
	}
	return starts, site.Load()
}

// TestLogosStackSiteStartsAtLearnedSize: after 1000 goroutines from one
// go statement each end with a 64 KB stack, the next goroutines from that
// statement start at 64 KB, and goroutines from another statement keep
// the global size.
func TestLogosStackSiteStartsAtLearnedSize(t *testing.T) {
	skipUnlessLogosStack(t)
	defer debug.SetGCPercent(debug.SetGCPercent(-1))
	runtime.LogosStackReset()

	var site atomic.Uintptr
	deep := func() {
		site.Store(runtime.LogosGoPC())
		logosStackUse(logosStackTarget * 3 / 4)
	}
	for range 10 {
		var wg sync.WaitGroup
		for range 100 {
			logosStackSpawn(&wg, deep)
		}
		wg.Wait()
	}
	pc := site.Load()
	if got := runtime.LogosStartSize(pc); got != logosStackTarget {
		t.Fatalf("LogosStartSize(site) = %d after 1000 deep goroutines, want %d", got, logosStackTarget)
	}
	if ema, ok := runtime.LogosStackEMA(pc); !ok || ema != logosStackTarget {
		t.Fatalf("LogosStackEMA(site) = %d, %v, want %d, true", ema, ok, logosStackTarget)
	}

	starts, again := logosStackStarts(logosStackSpawn, 8)
	if again != pc {
		t.Fatalf("second batch came from pc %#x, want the same go statement %#x", again, pc)
	}
	for _, size := range starts {
		if size != logosStackTarget {
			t.Fatalf("goroutine from the learned statement started with %d bytes, want %d", size, logosStackTarget)
		}
	}

	// The global starting size can move at the end of any GC cycle.
	// Finish the cycles before sampling it for the other statement, so
	// the batch cannot race the sample.
	runtime.GC()
	runtime.GC()
	debug.SetGCPercent(-1)
	global := runtime.LogosGlobalStartSize()
	if global == logosStackTarget {
		t.Skip("global starting size equals the learned size; the other statement cannot tell them apart")
	}
	if got := runtime.LogosStartSize(^uintptr(0)); got != global {
		t.Fatalf("LogosStartSize without history = %d, want the global size %d", got, global)
	}
	starts, other := logosStackStarts(logosStackSpawnOther, 8)
	if other == pc {
		t.Fatal("the two spawn helpers share a go statement pc")
	}
	for _, size := range starts {
		if !logosStackStartIs(size, global) {
			t.Fatalf("goroutine from another statement started with %d bytes, want the global size %d", size, global)
		}
	}
}

// TestLogosStackOffRecordsNothing pins the default build: without the
// experiment no history is recorded and every start size is the global
// one. With one P a goroutine finishes its exit path before the
// goroutine that waits for it runs again, so the check needs no sleep.
func TestLogosStackOffRecordsNothing(t *testing.T) {
	if goexperiment.LogosStack {
		t.Skip("covers the build without GOEXPERIMENT=logosstack")
	}
	defer runtime.GOMAXPROCS(runtime.GOMAXPROCS(1))
	runtime.LogosStackReset()

	// The global starting size is the start size the runtime promises a
	// goroutine with no call-site history on the recycled-g path: a
	// fresh g from malg starts at the smallest stack, and
	// gcComputeStartingStackSize may raise the global size at the end of
	// any GC cycle. The test finishes all GC cycles first, so the
	// global it samples can no longer move, and fills the free-g pool so
	// the batch is served from it, because only a recycled g is resized
	// to the global size at gfget.
	defer debug.SetGCPercent(debug.SetGCPercent(-1))
	runtime.GC()
	runtime.GC()
	debug.SetGCPercent(-1)
	var warm sync.WaitGroup
	for range 64 {
		logosStackSpawn(&warm, func() {})
	}
	warm.Wait()
	runtime.Gosched()
	global := runtime.LogosGlobalStartSize()

	starts, pc := logosStackStarts(logosStackSpawn, 16)
	var wg sync.WaitGroup
	logosStackSpawn(&wg, func() { logosStackUse(logosStackTarget * 3 / 4) })
	wg.Wait()
	if ema, ok := runtime.LogosStackEMA(pc); ok {
		t.Fatalf("history %d recorded without the experiment", ema)
	}
	if n := runtime.LogosStackOccupied(); n != 0 {
		t.Fatalf("%d table slots hold history without the experiment", n)
	}
	if got := runtime.LogosStartSize(^uintptr(0)); got != global {
		t.Fatalf("LogosStartSize without history = %d, want the global size %d", got, global)
	}
	for _, size := range starts {
		if !logosStackStartIs(size, global) {
			t.Fatalf("goroutine started with %d bytes, want the global size %d", size, global)
		}
	}
}

var (
	logosStackLeafDone = make(chan struct{}, 1)
	logosStackLeafSize atomic.Int64
	logosStackLeafPC   atomic.Uintptr
)

func logosStackLeaf() {
	logosStackLeafSize.Store(int64(runtime.LogosCurStackSize()))
	logosStackLeafPC.Store(runtime.LogosGoPC())
	logosStackLeafDone <- struct{}{}
}

//go:noinline
func logosStackLeafSpawn() {
	go logosStackLeaf()
	<-logosStackLeafDone
}

// logosStackAwait waits until the history of pc satisfies done. A
// goroutine records its exit after its WaitGroup is released, so a test
// that reads the table right after Wait must wait for the record.
func logosStackAwait(t *testing.T, pc uintptr, done func(ema int, ok bool) bool) {
	t.Helper()
	deadline := time.Now().Add(10 * time.Second)
	for {
		ema, ok := runtime.LogosStackEMA(pc)
		if done(ema, ok) {
			return
		}
		if time.Now().After(deadline) {
			t.Fatalf("history of pc %#x never reached the awaited state (ema %d, %v)", pc, ema, ok)
		}
		runtime.Gosched()
	}
}

// logosStackLearn waits until the exit of the goroutine that just ran
// from pc has landed in the table, then replaces the history of pc by
// size.
func logosStackLearn(t *testing.T, pc uintptr, size int) {
	t.Helper()
	logosStackAwait(t, pc, func(_ int, ok bool) bool { return ok })
	runtime.LogosStackReset()
	runtime.LogosStackRecord(pc, size)
}

// TestLogosStackSpawnDoesNotAllocate: choosing the size, replacing the
// stack, and recording the exit allocate nothing on the heap.
func TestLogosStackSpawnDoesNotAllocate(t *testing.T) {
	skipUnlessLogosStack(t)
	runtime.LogosStackReset()
	logosStackLeafSpawn()
	pc := logosStackLeafPC.Load()
	logosStackLearn(t, pc, logosStackTarget)

	if allocs := testing.AllocsPerRun(100, logosStackLeafSpawn); allocs != 0 {
		t.Fatalf("spawn with history allocated %v times per run, want 0", allocs)
	}
	if got := logosStackLeafSize.Load(); got != logosStackTarget {
		t.Fatalf("leaf started with %d bytes, want the learned %d", got, logosStackTarget)
	}
}

// TestLogosStackTableBounded: 10,000 distinct go statement pcs fit the
// 1024-entry table by replacement, with no allocation, and the newest
// entry always survives its own insertion.
func TestLogosStackTableBounded(t *testing.T) {
	if runtime.LogosStackSlots != 1024 {
		t.Fatalf("table capacity = %d, want 1024", runtime.LogosStackSlots)
	}
	runtime.LogosStackReset()
	const sites = 10000
	pcOf := func(i int) uintptr { return 0x401000 + uintptr(i)*16 }
	sizeOf := func(i int) int { return runtime.LogosFixedStack << (i % 10) }
	feed := func() {
		for i := range sites {
			runtime.LogosStackRecord(pcOf(i), sizeOf(i))
			if ema, ok := runtime.LogosStackEMA(pcOf(i)); !ok || ema != sizeOf(i) {
				t.Errorf("site %d: newest entry has ema %d, %v, want %d, true", i, ema, ok, sizeOf(i))
				return
			}
		}
	}
	if allocs := testing.AllocsPerRun(1, feed); allocs != 0 {
		t.Fatalf("recording %d sites allocated %v times, want 0", sites, allocs)
	}

	retained := 0
	for i := range sites {
		if _, ok := runtime.LogosStackEMA(pcOf(i)); ok {
			retained++
		}
	}
	occupied := runtime.LogosStackOccupied()
	if occupied == 0 || occupied > runtime.LogosStackSlots {
		t.Fatalf("occupied slots = %d, want 1..%d", occupied, runtime.LogosStackSlots)
	}
	if retained != occupied {
		t.Fatalf("%d sites retrievable but %d slots occupied", retained, occupied)
	}
}

// TestLogosStackLargeEntriesOutliveSmall: a flood of statements whose
// goroutines end with the smallest stack replaces small entries and
// never the entry of a statement whose goroutines need a large stack.
func TestLogosStackLargeEntriesOutliveSmall(t *testing.T) {
	runtime.LogosStackReset()
	const big = 0x405000
	runtime.LogosStackRecord(big, 1<<20)
	for i := range 10000 {
		runtime.LogosStackRecord(0x500000+uintptr(i)*16, runtime.LogosFixedStack)
	}
	if ema, ok := runtime.LogosStackEMA(big); !ok || ema != 1<<20 {
		t.Fatalf("large entry after the flood: ema %d, %v, want %d, true", ema, ok, 1<<20)
	}
}

// logosStackLearnSite runs one goroutine through logosStackSpawn and
// returns the pc of its go statement once its history has size bytes.
func logosStackLearnSite(t *testing.T, size int) uintptr {
	t.Helper()
	var wg sync.WaitGroup
	var site atomic.Uintptr
	logosStackSpawn(&wg, func() { site.Store(runtime.LogosGoPC()) })
	wg.Wait()
	pc := site.Load()
	logosStackLearn(t, pc, size)
	return pc
}

// TestLogosStackStartBoundedByMaxStack: a statement whose history exceeds
// the maximum stack starts its goroutines at the largest power of two
// within the maximum stack.
func TestLogosStackStartBoundedByMaxStack(t *testing.T) {
	skipUnlessLogosStack(t)
	defer debug.SetGCPercent(debug.SetGCPercent(-1))
	runtime.LogosStackReset()
	logosStackLearnSite(t, 1<<22)

	defer debug.SetMaxStack(debug.SetMaxStack(3 << 18))
	starts, _ := logosStackStarts(logosStackSpawn, 4)
	for _, size := range starts {
		if size != 1<<19 {
			t.Fatalf("goroutine started with %d bytes under a %d-byte maximum stack, want %d", size, 3<<18, 1<<19)
		}
	}
}

// TestLogosStackSurvivesCollection: goroutines that start at a learned
// size stay correct across collections, which scan their stacks and
// shrink the ones they barely use.
func TestLogosStackSurvivesCollection(t *testing.T) {
	skipUnlessLogosStack(t)
	runtime.LogosStackReset()
	logosStackLearnSite(t, logosStackTarget)

	const parked = 200
	release := make(chan struct{})
	releaseAll := sync.OnceFunc(func() { close(release) })
	defer releaseAll()
	var wg sync.WaitGroup
	var ready, sum atomic.Int64
	for range parked {
		logosStackSpawn(&wg, func() {
			ready.Add(1)
			<-release
			sum.Add(int64(logosStackUse(8 << 10)))
		})
	}
	deadline := time.Now().Add(10 * time.Second)
	for ready.Load() < parked {
		if time.Now().After(deadline) {
			t.Fatalf("%d of %d goroutines started", ready.Load(), parked)
		}
		runtime.Gosched()
	}
	for range 3 {
		runtime.GC()
	}
	releaseAll()
	wg.Wait()
	if want := parked * int64(logosStackUse(8<<10)); sum.Load() != want {
		t.Fatalf("goroutines computed %d, want %d", sum.Load(), want)
	}
}

// TestLogosStackEMAWeight pins the moving average: the first sample seeds
// it, each later sample moves it one eighth of the way, and it settles
// exactly on a steady sample from either side.
func TestLogosStackEMAWeight(t *testing.T) {
	runtime.LogosStackReset()
	const pc = 0x402000
	unit := runtime.LogosFixedStack
	ema := func() int {
		got, ok := runtime.LogosStackEMA(pc)
		if !ok {
			t.Fatal("no history after a sample")
		}
		return got
	}

	runtime.LogosStackRecord(pc, unit)
	if got := ema(); got != unit {
		t.Fatalf("seed = %d, want %d", got, unit)
	}
	runtime.LogosStackRecord(pc, 32*unit)
	if got, want := ema(), 4*unit; got != want {
		t.Fatalf("after 1 and 32 units = %d, want %d (7/8 of 1 plus 1/8 of 32, rounded down)", got, want)
	}
	for range 200 {
		runtime.LogosStackRecord(pc, 32*unit)
	}
	if got, want := ema(), 32*unit; got != want {
		t.Fatalf("after steady 32 units = %d, want %d", got, want)
	}
	for range 200 {
		runtime.LogosStackRecord(pc, unit)
	}
	if got := ema(); got != unit {
		t.Fatalf("after steady 1 unit = %d, want %d", got, unit)
	}
}

// TestLogosStackPickBounds: the start size is a power of two, at least
// the smallest stack, at most the largest power of two within the
// maximum stack (or the smallest stack when the maximum is smaller), and
// the moving average rounded up to a power of two when that fits.
func TestLogosStackPickBounds(t *testing.T) {
	unit := runtime.LogosFixedStack
	floorPow2 := func(n int) int {
		p := 1
		for p <= n/2 {
			p *= 2
		}
		return p
	}
	ceilPow2 := func(n int) int {
		p := 1
		for p < n {
			p *= 2
		}
		return p
	}
	limits := []int{0, 1, unit - 1, unit, unit + 1, 3 * unit, 64 << 10, 3 << 20, 1 << 30, math.MaxInt}
	for _, limit := range limits {
		for ema := unit; ema <= 1<<30; ema += (ema/(3*unit) + 1) * unit {
			got := runtime.LogosStackPick(ema, limit)
			ceiling := max(unit, floorPow2(limit))
			if got < unit || got&(got-1) != 0 || got > ceiling {
				t.Fatalf("pick(ema=%d, limit=%d) = %d, want a power of two in [%d, %d]", ema, limit, got, unit, ceiling)
			}
			if want := min(ceilPow2(ema), ceiling); got != want {
				t.Fatalf("pick(ema=%d, limit=%d) = %d, want %d", ema, limit, got, want)
			}
		}
	}
}

// TestLogosStackRecordClampsSamples: a sample of any size, including
// sizes no stack can have, keeps the stored average valid.
func TestLogosStackRecordClampsSamples(t *testing.T) {
	runtime.LogosStackReset()
	const pc = 0x403000
	for _, size := range []int{0, 1, runtime.LogosFixedStack, 1 << 30, math.MaxInt32, math.MaxInt} {
		for range 64 {
			runtime.LogosStackRecord(pc, size)
		}
		ema, ok := runtime.LogosStackEMA(pc)
		if !ok || ema < runtime.LogosFixedStack || uint64(ema) > 1<<31 {
			t.Fatalf("after samples of %d: ema = %d, %v", size, ema, ok)
		}
		start := runtime.LogosStartSize(pc)
		if start < runtime.LogosFixedStack || start&(start-1) != 0 {
			t.Fatalf("after samples of %d: start size %d is not a valid stack size", size, start)
		}
	}
}

// TestLogosStackConcurrentTable: Ps that record and look up the same
// slots at once never observe an invalid size.
func TestLogosStackConcurrentTable(t *testing.T) {
	runtime.LogosStackReset()
	unit := runtime.LogosFixedStack
	var wg sync.WaitGroup
	for seed := range 8 {
		wg.Go(func() {
			for i := range 20000 {
				pc := 0x404000 + uintptr((i*7+seed)%96)*16
				runtime.LogosStackRecord(pc, unit<<((i+seed)%12))
				got := runtime.LogosStartSize(pc)
				if got < unit || got&(got-1) != 0 {
					t.Errorf("start size %d is not a valid stack size", got)
					return
				}
			}
		})
	}
	wg.Wait()
}

// logosStackFrames recurses with a 4096-byte frame buffer per level, the
// size of the O3 stack-first buffer, stores the stack size at the bottom
// in deepest, and reports whether every buffer survived the stack moves
// of its callees.
//
//go:noinline
func logosStackFrames(depth int, seed byte, deepest *int) bool {
	var frame [4096]byte
	for i := range frame {
		frame[i] = seed ^ byte(i)
	}
	ok := true
	if depth > 0 {
		ok = logosStackFrames(depth-1, seed+1, deepest)
	} else {
		*deepest = runtime.LogosCurStackSize()
	}
	for i := range frame {
		if frame[i] != seed^byte(i) {
			return false
		}
	}
	return ok
}

// logosStackMakes recurses with a variable-size make that does not
// escape, so sizes up to the stack-first threshold live in the frame.
//
//go:noinline
func logosStackMakes(depth, n int, seed byte) bool {
	b := make([]byte, n)
	for i := range b {
		b[i] = seed + byte(i)
	}
	ok := true
	if depth > 0 {
		ok = logosStackMakes(depth-1, n, seed+1)
	}
	for i := range b {
		if b[i] != seed+byte(i) {
			return false
		}
	}
	return ok
}

// TestLogosStackFirstBuffersSurviveStartSizes (#33, the part S2 can
// reach): buffers stay correct in goroutines that start at the global
// size, at a learned small size, and at a learned large size, through the
// stack growth a deep recursion forces. The 4096-byte frame arrays are
// real stack frames of the O3 buffer size. The variable-size makes sit on
// both sides of the O2 and O3 thresholds, but this build uses the
// default 32-byte threshold, so only the sizes up to 32 take the stack
// buffer and the rest take makeslice; the thresholds themselves are
// M17-S1's.
func TestLogosStackFirstBuffersSurviveStartSizes(t *testing.T) {
	defer debug.SetGCPercent(debug.SetGCPercent(-1))
	var wg sync.WaitGroup
	var site atomic.Uintptr
	logosStackSpawn(&wg, func() { site.Store(runtime.LogosGoPC()) })
	wg.Wait()
	pc := site.Load()

	sizes := []int{runtime.LogosFixedStack, 16 << 10, logosStackTarget}
	for _, size := range sizes {
		if goexperiment.LogosStack {
			logosStackLearn(t, pc, size)
		}
		var ok atomic.Bool
		var grown, start atomic.Int64
		logosStackSpawn(&wg, func() {
			start.Store(int64(runtime.LogosCurStackSize()))
			var deepest int
			good := logosStackFrames(24, 1, &deepest)
			for _, n := range []int{31, 32, 33, 255, 256, 257, 4095, 4096, 4097} {
				good = good && logosStackMakes(200, n, 7)
			}
			ok.Store(good)
			grown.Store(int64(deepest))
		})
		wg.Wait()
		if goexperiment.LogosStack {
			logosStackAwait(t, pc, func(ema int, _ bool) bool { return ema != size })
		}
		if goexperiment.LogosStack && !logosStackStartIs(int(start.Load()), size) {
			t.Fatalf("goroutine started with %d bytes, want the learned %d", start.Load(), size)
		}
		if !ok.Load() {
			t.Fatalf("a stack-first buffer lost its contents in a goroutine starting at %d bytes", size)
		}
		if got := grown.Load(); got < 24*4096 {
			t.Fatalf("goroutine starting at %d bytes ended with a %d-byte stack; the recursion did not grow it", size, got)
		}
	}
}
