// Copyright 2026 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

//go:build goexperiment.greenteagc && (!js || !wasm)

package runtime_test

import (
	"internal/goexperiment"
	"math/rand/v2"
	"runtime"
	"slices"
	"testing"
	"unsafe"
	"weak"
)

// M19-S6 attack (SPEC Sec 19.1, T-301.2). Under -gc=greentea a pointer into
// a span sets the object's mark bit and marks the span dirty without
// following the pointer; workers pop dirty spans in monotonic address order;
// a single-page 8 KB span scans its grey objects sequentially in one pass;
// pointer-keyed maps keep the standard hash because objects never move.

const spanDeckPage = 8192

type spanDeckParent struct {
	child *spanDeckChild
	pad   [3]uintptr
}

type spanDeckChild struct {
	next *int
	pad  [5]uintptr
}

var spanDeckSink [2]unsafe.Pointer

// A pointer store into span S sets the bit in S's mark bits and marks S dirty
// without dereferencing the pointer: the child the parent points to stays
// unmarked until the span is scanned. The test reads the bitmaps only. The
// marking path is shared, so this runs in both the deck and the FIFO builds.
func TestLogosSpanDeckMarkSetsBitAndDirtyWithoutScanning(t *testing.T) {
	checkMarkProbe(t)
}

func checkMarkProbe(t *testing.T) {
	t.Helper()
	const attempts = 5
	for range attempts {
		parent, child := new(spanDeckParent), new(spanDeckChild)
		parent.child = child
		spanDeckSink[0], spanDeckSink[1] = unsafe.Pointer(parent), unsafe.Pointer(child)
		runtime.GC()
		r := runtime.LogosSpanMarkProbe(unsafe.Pointer(parent), unsafe.Pointer(child))
		if !r.Fresh {
			continue
		}
		if r.ParentMarkedBefore || r.ParentDirtyBefore {
			t.Fatalf("before marking: marked=%v dirty=%v, want both false", r.ParentMarkedBefore, r.ParentDirtyBefore)
		}
		if !r.ParentMarked || !r.ParentDirty {
			t.Fatalf("after marking: marked=%v dirty=%v, want both true", r.ParentMarked, r.ParentDirty)
		}
		if r.Queued != 1 {
			t.Fatalf("marking queued %d spans, want 1", r.Queued)
		}
		if r.ChildMarkedAfterMark {
			t.Fatal("marking the parent marked its child: the pointer was followed")
		}
		if !r.ChildMarkedAfterScan || !r.ChildDirtyAfterScan {
			t.Fatalf("after scanning the span: child marked=%v dirty=%v, want both true", r.ChildMarkedAfterScan, r.ChildDirtyAfterScan)
		}
		return
	}
	t.Skipf("no clean span after %d attempts", attempts)
}

// span returns the page address of a span number k of a fake arena.
func span(k uintptr) uintptr { return 0x40000000 + k*spanDeckPage }

// checkSweeps asserts the pop sequence ascends within every sweep and that a
// new sweep starts only when the previous one moved to a lower address.
func checkSweeps(t *testing.T, bases []uintptr, sweeps []uint32) {
	t.Helper()
	for i := 1; i < len(bases); i++ {
		switch {
		case sweeps[i] == sweeps[i-1] && bases[i] <= bases[i-1]:
			t.Fatalf("pop %d: base %#x after %#x inside sweep %d", i, bases[i], bases[i-1], sweeps[i])
		case sweeps[i] == sweeps[i-1]+1 && bases[i] >= bases[i-1]:
			t.Fatalf("pop %d: sweep %d began at %#x although %#x was still above the cursor", i, sweeps[i], bases[i], bases[i-1])
		case sweeps[i] != sweeps[i-1] && sweeps[i] != sweeps[i-1]+1:
			t.Fatalf("pop %d: sweep jumped from %d to %d", i, sweeps[i-1], sweeps[i])
		}
	}
}

// Workers pop dirty spans in monotonic address order. The test records the
// pop sequence of a queue filled out of order and asserts it is sorted.
func TestLogosSpanDeckPopsInAddressOrder(t *testing.T) {
	order := []uintptr{9, 3, 7, 1, 8, 2, 6, 4, 5, 11, 10}
	script := make([]uintptr, len(order))
	want := make([]uintptr, len(order))
	for i, k := range order {
		script[i], want[i] = span(k), span(k)
	}
	slices.Sort(want)

	bases, sweeps, lost, inversions := runtime.LogosSpanQueueScript(script)
	if !goexperiment.LogosSpanDeck {
		if !slices.Equal(bases, script) {
			t.Fatalf("FIFO pop sequence\n got %#x\nwant %#x", bases, script)
		}
		if lost != 0 || inversions != 0 {
			t.Fatalf("lost=%d inversions=%d", lost, inversions)
		}
		return
	}
	if !slices.Equal(bases, want) {
		t.Fatalf("pop sequence\n got %#x\nwant %#x", bases, want)
	}
	if !slices.IsSorted(bases) || lost != 0 || inversions != 0 {
		t.Fatalf("sorted=%v lost=%d inversions=%d", slices.IsSorted(bases), lost, inversions)
	}
	checkSweeps(t, bases, sweeps)
	if sweeps[len(sweeps)-1] != 0 {
		t.Fatalf("one fill and one drain took %d sweeps, want 1", sweeps[len(sweeps)-1]+1)
	}
}

// A span marked behind the cursor waits for the next sweep; a span marked
// ahead of it joins the current one.
func TestLogosSpanDeckSpanBehindTheCursorWaitsForNextSweep(t *testing.T) {
	script := []uintptr{span(5), span(3), span(8), 0, span(1), span(6)}
	bases, sweeps, lost, inversions := runtime.LogosSpanQueueScript(script)
	if !goexperiment.LogosSpanDeck {
		wantFIFO := []uintptr{span(5), span(3), span(8), span(1), span(6)}
		if !slices.Equal(bases, wantFIFO) || lost != 0 || inversions != 0 {
			t.Fatalf("FIFO bases %#x lost %d inversions %d\nwant %#x", bases, lost, inversions, wantFIFO)
		}
		return
	}
	wantBases := []uintptr{span(3), span(5), span(6), span(8), span(1)}
	wantSweeps := []uint32{0, 0, 0, 0, 1}
	if !slices.Equal(bases, wantBases) || !slices.Equal(sweeps, wantSweeps) || lost != 0 || inversions != 0 {
		t.Fatalf("bases %#x sweeps %v lost %d inversions %d\nwant bases %#x sweeps %v", bases, sweeps, lost, inversions, wantBases, wantSweeps)
	}
}

// Spans that spill to the shared chain and come back by stealing from the
// owner's own chain are neither lost nor duplicated, and the order holds in
// every sweep. The script keeps more than a ring of spans outstanding.
func TestLogosSpanDeckSpillAndStealKeepEverySpanOnce(t *testing.T) {
	rng := rand.New(rand.NewPCG(19, 1))
	perm := rng.Perm(2000)
	var script []uintptr
	var puts []uintptr
	next := 0
	for len(puts) < len(perm) {
		if len(puts) >= 600 && rng.IntN(3) == 0 {
			script = append(script, 0)
			continue
		}
		k := span(uintptr(perm[next]))
		next++
		script = append(script, k)
		puts = append(puts, k)
	}

	bases, sweeps, lost, inversions := runtime.LogosSpanQueueScript(script)
	if !goexperiment.LogosSpanDeck {
		if lost != 0 || inversions != 0 {
			t.Fatalf("lost=%d inversions=%d", lost, inversions)
		}
		got := slices.Clone(bases)
		slices.Sort(got)
		slices.Sort(puts)
		if !slices.Equal(got, puts) {
			t.Fatalf("popped %d spans, put %d: sets differ", len(got), len(puts))
		}
		return
	}
	checkSweeps(t, bases, sweeps)
	if lost != 0 || inversions != 0 {
		t.Fatalf("lost=%d inversions=%d", lost, inversions)
	}
	got := slices.Clone(bases)
	slices.Sort(got)
	slices.Sort(puts)
	if !slices.Equal(got, puts) {
		t.Fatalf("popped %d spans, put %d: sets differ", len(got), len(puts))
	}
}

// A ring spills its highest spans: the owner keeps the low end, which its
// sweep reaches first, and the spilled spans stay reachable for thieves. The
// queue spills 16 spans on the 64th put (spillPeriod and spillMax in
// spanQueue.put), so after 64 ascending puts those are spans 48 to 63.
func TestLogosSpanDeckSpillKeepsTheHighestSpansForThieves(t *testing.T) {
	const puts, spilledFrom, spilledTo = 100, 48, 64
	script := make([]uintptr, puts)
	for i := range script {
		script[i] = span(uintptr(i))
	}
	bases, sweeps, lost, inversions := runtime.LogosSpanQueueScript(script)
	if lost != 0 || inversions != 0 {
		t.Fatalf("lost=%d inversions=%d", lost, inversions)
	}
	if !goexperiment.LogosSpanDeck {
		var wantFIFO []uintptr
		for k := uintptr(16); k < puts; k++ {
			wantFIFO = append(wantFIFO, span(k))
		}
		for k := uintptr(0); k < 16; k++ {
			wantFIFO = append(wantFIFO, span(k))
		}
		if !slices.Equal(bases, wantFIFO) {
			t.Fatalf("FIFO spill order\n got %#x\nwant %#x", bases, wantFIFO)
		}
		return
	}
	checkSweeps(t, bases, sweeps)
	var first, second []uintptr
	for i, b := range bases {
		if sweeps[i] == 0 {
			first = append(first, b)
		} else {
			second = append(second, b)
		}
	}
	var wantSecond []uintptr
	for k := uintptr(spilledFrom); k < spilledTo; k++ {
		wantSecond = append(wantSecond, span(k))
	}
	if !slices.Equal(second, wantSecond) || len(first) != puts-len(wantSecond) {
		t.Fatalf("spilled spans %#x, want %#x; %d spans in the first sweep", second, wantSecond, len(first))
	}
}

type spanDeckRec struct {
	a   *int
	pad [2]uintptr
	b   *int
}

// A single-page 8 KB span scans its grey objects sequentially in one pass:
// one call collects the pointer slots of every grey object, in address order,
// with no slot of a white object.
func TestLogosSpanDeckScansSinglePageSpanInOnePass(t *testing.T) {
	recs := make([]*spanDeckRec, 1024)
	for i := range recs {
		recs[i] = new(spanDeckRec)
	}
	byPage := map[uintptr][]*spanDeckRec{}
	for _, r := range recs {
		page := uintptr(unsafe.Pointer(r)) &^ (spanDeckPage - 1)
		byPage[page] = append(byPage[page], r)
	}
	var objs []*spanDeckRec
	for _, group := range byPage {
		if len(group) > len(objs) {
			objs = group
		}
	}
	if len(objs) < 30 {
		t.Skipf("the allocator scattered the records: %d in the largest span", len(objs))
	}
	slices.SortFunc(objs, func(x, y *spanDeckRec) int {
		return int(uintptr(unsafe.Pointer(x)) - uintptr(unsafe.Pointer(y)))
	})
	var grey []unsafe.Pointer
	var want []uintptr
	for i, o := range objs {
		if i%3 != 0 {
			continue
		}
		grey = append(grey, unsafe.Pointer(o))
		base := uintptr(unsafe.Pointer(o))
		want = append(want, base+unsafe.Offsetof(o.a), base+unsafe.Offsetof(o.b))
	}

	slots, base, size, elemsize := runtime.LogosSpanGatherSlots(grey)
	if size != spanDeckPage || base&(spanDeckPage-1) != 0 {
		t.Fatalf("span base %#x size %d, want one page-aligned %d-byte page", base, size, spanDeckPage)
	}
	if elemsize != unsafe.Sizeof(spanDeckRec{}) {
		t.Fatalf("object size %d, want %d", elemsize, unsafe.Sizeof(spanDeckRec{}))
	}
	if !slices.Equal(slots, want) {
		t.Fatalf("slots\n got %#x\nwant %#x", slots, want)
	}
	runtime.KeepAlive(recs)
}

type spanDeckNode struct {
	next        *spanDeckNode
	left, right *spanDeckNode
	v           int
}

var spanDeckRoot *spanDeckNode

func spanDeckGarbage(n int) []weak.Pointer[spanDeckNode] {
	ws := make([]weak.Pointer[spanDeckNode], n)
	for i := range ws {
		ws[i] = weak.Make(&spanDeckNode{v: i})
	}
	return ws
}

// Whole collections keep every reachable object and free the unreachable
// ones; in the deck build the deck itself reports that it ran and that no
// pop broke the order inside a sweep.
func TestLogosSpanDeckCollectsWithTheDeckSelected(t *testing.T) {
	const live, dead = 40000, 4000
	liveWeak := make([]weak.Pointer[spanDeckNode], live)
	var head *spanDeckNode
	sum := 0
	for i := range live {
		n := &spanDeckNode{next: head, v: i}
		if i > 2 {
			n.left, n.right = head.next, head.next.next
		}
		head = n
		liveWeak[i] = weak.Make(n)
		sum += i
	}
	spanDeckRoot = head
	deadWeak := spanDeckGarbage(dead)

	popsBefore, _ := runtime.LogosSpanDeckStats()
	for range 3 {
		runtime.GC()
	}
	pops, inversions := runtime.LogosSpanDeckStats()

	got, walked := 0, 0
	for n := spanDeckRoot; n != nil; n = n.next {
		got += n.v
		walked++
	}
	if walked != live || got != sum {
		t.Fatalf("list walked %d nodes with sum %d, want %d with %d", walked, got, live, sum)
	}
	for i, w := range liveWeak {
		if w.Value() == nil {
			t.Fatalf("reachable node %d was freed", i)
		}
	}
	left := 0
	for _, w := range deadWeak {
		if w.Value() != nil {
			left++
		}
	}
	if left*100 > dead*5 {
		t.Fatalf("%d of %d unreachable nodes survived three collections", left, dead)
	}
	if goexperiment.LogosSpanDeck {
		if pops == popsBefore {
			t.Fatal("the collections never popped a span from the deck")
		}
		if inversions != 0 {
			t.Fatalf("%d pops broke the address order inside a sweep", inversions)
		}
	}
	spanDeckRoot = nil
}

// A heap of half a million small records, wired in allocation order with
// random long edges, survives four collections: every edge is intact, and in
// the deck build the deck popped spans without breaking its order.
func TestLogosSpanDeckLargeHeapKeepsEveryEdge(t *testing.T) {
	recs := spanOrderHeap(1 << 19)
	near, far := spanOrderSnapshot(recs)
	popsBefore, _ := runtime.LogosSpanDeckStats()
	for range 4 {
		runtime.GC()
	}
	pops, inversions := runtime.LogosSpanDeckStats()
	checkSpanOrderHeap(t, recs, near, far)
	if goexperiment.LogosSpanDeck && (pops == popsBefore || inversions != 0) {
		t.Fatalf("pops %d -> %d, inversions %d", popsBefore, pops, inversions)
	}
	runtime.KeepAlive(recs)
}

// Pointer-keyed maps hash with the standard memhash: the collector never
// moves objects, so the address is a stable key in both builds.
func TestLogosSpanDeckPointerMapsKeepStandardMemhash(t *testing.T) {
	hasher := runtime.LogosMapKeyHasher(map[*int]int{})
	keys := make([]*int, 256)
	addrs := make([]uintptr, len(keys))
	m := make(map[*int]int, len(keys))
	for i := range keys {
		keys[i] = new(int)
		addrs[i] = uintptr(unsafe.Pointer(keys[i]))
		m[keys[i]] = i
	}
	for _, k := range keys {
		for _, seed := range []uintptr{0, 1, 0xdeadbeef} {
			got, want := hasher(unsafe.Pointer(&k), seed), runtime.MemHash64(unsafe.Pointer(&k), seed)
			if got != want {
				t.Fatalf("hash of %p seed %#x = %#x, want memhash64 %#x", k, seed, got, want)
			}
		}
	}
	for range 3 {
		runtime.GC()
	}
	for i, k := range keys {
		if uintptr(unsafe.Pointer(k)) != addrs[i] {
			t.Fatalf("key %d moved from %#x to %p", i, addrs[i], k)
		}
		if v, ok := m[k]; !ok || v != i {
			t.Fatalf("key %d lost across collections: %d %v", i, v, ok)
		}
	}
}
