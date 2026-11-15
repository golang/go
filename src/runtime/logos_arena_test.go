// Copyright 2026 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

//go:build goexperiment.logosarena

package runtime_test

import (
	"fmt"
	"internal/goexperiment"
	"math"
	"math/bits"
	"runtime"
	"runtime/debug"
	"sync"
	"testing"
	"time"
	"unsafe"
	"weak"
)

var arenaSink any

func inFreshGoroutine(t *testing.T, fn func()) {
	t.Helper()
	goroutines := runtime.NumGoroutine()
	done := make(chan any)
	go func() {
		defer func() { done <- recover() }()
		fn()
	}()
	r := <-done
	waitGoroutines(t, goroutines)
	if r != nil {
		t.Fatalf("panic in goroutine: %v", r)
	}
}

func recoverMsg(fn func()) (msg string, panicked bool) {
	defer func() {
		if r := recover(); r != nil {
			msg, panicked = fmt.Sprint(r), true
		}
	}()
	fn()
	return
}

func waitLeased(t *testing.T, want int) int {
	t.Helper()
	deadline := time.Now().Add(20 * time.Second)
	for {
		_, leased, _ := runtime.LogosArenaCounts()
		if leased == want || time.Now().After(deadline) {
			return leased
		}
		time.Sleep(time.Millisecond)
	}
}

func waitGoroutines(t *testing.T, want int) {
	t.Helper()
	deadline := time.Now().Add(20 * time.Second)
	for runtime.NumGoroutine() > want && time.Now().Before(deadline) {
		time.Sleep(time.Millisecond)
	}
	if got := runtime.NumGoroutine(); got > want {
		t.Fatalf("goroutines = %d, want %d", got, want)
	}
}

func fillBytes(p unsafe.Pointer, size uintptr, v byte) {
	b := unsafe.Slice((*byte)(p), size)
	for i := range b {
		b[i] = v
	}
}

func checkBytes(p unsafe.Pointer, size uintptr, v byte) bool {
	for _, x := range unsafe.Slice((*byte)(p), size) {
		if x != v {
			return false
		}
	}
	return true
}

func fillWords(p unsafe.Pointer, size uintptr, v *int) {
	words := unsafe.Slice((**int)(p), size/unsafe.Sizeof(v))
	for i := range words {
		words[i] = v
	}
}

func TestLogosArenaParkReturnsChunk(t *testing.T) {
	const (
		waves   = 25
		perWave = 4000
	)
	goroutines := runtime.NumGoroutine()
	_, leasedBase, _ := runtime.LogosArenaCounts()
	for w := range waves {
		release := make(chan struct{})
		var started sync.WaitGroup
		started.Add(perWave)
		for range perWave {
			go func() {
				m := runtime.LogosArenaSave()
				runtime.LogosArenaAlloc(byte(0), 64)
				runtime.LogosArenaAllocScan((*int)(nil), 48)
				runtime.LogosArenaRestore(m)
				started.Done()
				<-release
			}()
		}
		started.Wait()
		if leased := waitLeased(t, leasedBase); leased != leasedBase {
			close(release)
			waitGoroutines(t, goroutines)
			t.Fatalf("wave %d: leased chunks = %d with %d goroutines parked, want baseline %d", w, leased, perWave, leasedBase)
		}
		close(release)
		waitGoroutines(t, goroutines)
	}
	runtime.GC()
	total, leased, pooled := runtime.LogosArenaCounts()
	if pooled != 0 || total != leased {
		t.Errorf("after GC: total=%d leased=%d pooled=%d, want every cached chunk freed", total, leased, pooled)
	}
}

func TestLogosArenaParkKeepsLiveChunk(t *testing.T) {
	goroutines := runtime.NumGoroutine()
	_, leasedBase, _ := runtime.LogosArenaCounts()
	parked := make(chan struct{})
	wake := make(chan struct{})
	wake2 := make(chan struct{})
	done := make(chan struct{})
	go func() {
		defer close(done)
		m := runtime.LogosArenaSave()
		runtime.LogosArenaAllocScan((*int)(nil), 32)
		close(parked)
		<-wake
		runtime.LogosArenaRestore(m)
		<-wake2
	}()
	<-parked
	for range 100 {
		runtime.Gosched()
	}
	if _, leased, _ := runtime.LogosArenaCounts(); leased != leasedBase+1 {
		t.Fatalf("leased = %d while a goroutine is parked with live arena memory, want %d", leased, leasedBase+1)
	}
	wake <- struct{}{}
	if leased := waitLeased(t, leasedBase); leased != leasedBase {
		t.Errorf("leased = %d after the goroutine parked with empty regions, want %d", leased, leasedBase)
	}
	wake2 <- struct{}{}
	<-done
	waitGoroutines(t, goroutines)
}

func TestLogosArenaMarkSurvivesPark(t *testing.T) {
	goroutines := runtime.NumGoroutine()
	_, leasedBase, _ := runtime.LogosArenaCounts()
	park := make(chan struct{})
	wake := make(chan struct{})
	done := make(chan struct{})
	go func() {
		defer close(done)
		m0 := runtime.LogosArenaSave()
		runtime.LogosArenaAlloc(byte(0), 16)
		runtime.LogosArenaRestore(m0)
		if st := runtime.LogosArenaStat(); st.Chunks != 1 || st.NoscanUsed != 0 {
			t.Errorf("after restore: %+v, want one empty chunk", st)
		}
		m := runtime.LogosArenaSave()
		park <- struct{}{}
		<-wake
		if st := runtime.LogosArenaStat(); st.Chunks != 0 {
			t.Errorf("chunks = %d after parking with empty regions, want 0", st.Chunks)
		}
		runtime.LogosArenaAlloc(byte(0), 16)
		runtime.LogosArenaRestore(m)
		if st := runtime.LogosArenaStat(); st.Chunks != 1 || st.NoscanUsed != 0 {
			t.Errorf("after restoring a mark from before the park: %+v, want one empty chunk", st)
		}
	}()
	<-park
	if leased := waitLeased(t, leasedBase); leased != leasedBase {
		t.Errorf("leased = %d with the goroutine parked at an empty mark, want %d", leased, leasedBase)
	}
	wake <- struct{}{}
	<-done
	waitGoroutines(t, goroutines)
}

const arenaParkBig = runtime.LogosArenaCapacity - runtime.LogosArenaUnit

func arenaEmptyMarks(addr uintptr) (noscan, scan uintptr) {
	return addr + runtime.LogosArenaChunkBytes - runtime.LogosArenaCapacity,
		addr + runtime.LogosArenaChunkBytes
}

func checkWords(p unsafe.Pointer, size uintptr, v *int) bool {
	words := unsafe.Slice((**int)(p), size/unsafe.Sizeof(v))
	for _, w := range words {
		if w != v {
			return false
		}
	}
	return true
}

const arenaParkMax = 4

type arenaParkBuilt struct {
	addrs [arenaParkMax]uintptr
	objs  [arenaParkMax]unsafe.Pointer
	goid  uint64
}

func arenaParkCase(t *testing.T, live []bool, empty func(addr uintptr) bool) {
	t.Helper()
	if len(live) == 0 || len(live) > arenaParkMax {
		t.Fatalf("arenaParkCase: %d chunks, want 1 to %d", len(live), arenaParkMax)
	}
	_, leasedBase, _ := runtime.LogosArenaCounts()
	x := new(int)
	built := make(chan arenaParkBuilt)
	wake := make(chan struct{})
	done := make(chan struct{})
	go func() {
		defer close(done)
		m := runtime.LogosArenaSave()
		var b arenaParkBuilt
		b.goid = runtime.LogosArenaGoid()
		for i := range live {
			var p unsafe.Pointer
			if i%2 == 0 {
				p = runtime.LogosArenaAlloc(byte(0), arenaParkBig)
				fillBytes(p, arenaParkBig, byte(0x40+i))
			} else {
				p = runtime.LogosArenaAllocScan((*int)(nil), arenaParkBig)
				fillWords(p, arenaParkBig, x)
			}
			head, _, _ := runtime.LogosArenaMarkSave()
			b.addrs[i] = head
			b.objs[i] = p
		}
		for i, keep := range live {
			if !keep {
				b.objs[i] = nil
				if !empty(b.addrs[i]) {
					t.Errorf("chunk %d: emptying it in place failed", i)
				}
			}
		}
		built <- b
		<-wake
		runtime.LogosArenaRestore(m)
	}()

	b := <-built
	liveCount := 0
	for _, keep := range live {
		if keep {
			liveCount++
		}
	}
	if leased := waitLeased(t, leasedBase+liveCount); leased != leasedBase+liveCount {
		close(wake)
		<-done
		t.Fatalf("leased = %d with the goroutine parked, want %d", leased, leasedBase+liveCount)
	}
	total, leased, pooled := runtime.LogosArenaCounts()
	if total != leased+pooled {
		t.Errorf("total = %d chunks, want leased %d + pooled %d", total, leased, pooled)
	}
	arenaParkChainCheck(t, b, live)
	for j, keep := range live {
		if !keep {
			continue
		}
		if j%2 == 0 && !checkBytes(b.objs[j], arenaParkBig, byte(0x40+j)) {
			t.Errorf("chunk %d: noscan object lost its contents across the park", j)
		}
		if j%2 == 1 && !checkWords(b.objs[j], arenaParkBig, x) {
			t.Errorf("chunk %d: scan object lost its contents across the park", j)
		}
	}
	close(wake)
	<-done
	arenaWaitLeased(t, leasedBase)
}

func arenaParkChainCheck(t *testing.T, b arenaParkBuilt, live []bool) {
	t.Helper()
	deadline, ok := t.Deadline()
	if !ok {
		deadline = time.Now().Add(time.Minute)
	}
	var chain [arenaParkMax + 1]runtime.LogosArenaChunk
	var n int
	for {
		var waiting bool
		n, waiting = runtime.LogosArenaChainOf(b.goid, chain[:])
		if waiting {
			break
		}
		if !time.Now().Before(deadline) {
			t.Fatalf("goroutine %d is not parked", b.goid)
		}
		runtime.Gosched()
	}
	held := make(map[uintptr]bool)
	for _, c := range chain[:min(n, len(chain))] {
		held[c.Addr] = true
	}
	want := 0
	for j, keep := range live {
		if keep {
			want++
			if !held[b.addrs[j]] {
				t.Errorf("chunk %d at %#x left the parked goroutine's chain", j, b.addrs[j])
			}
		}
	}
	if n != want {
		t.Errorf("parked chain holds %d chunks, want %d live ones", n, want)
	}
}

func arenaWaitLeased(t *testing.T, want int) {
	t.Helper()
	deadline := time.Now().Add(20 * time.Second)
	for {
		_, leased, _ := runtime.LogosArenaCounts()
		if leased == want || time.Now().After(deadline) {
			if leased != want {
				t.Errorf("leased = %d after cleanup, want %d", leased, want)
			}
			return
		}
		runtime.Gosched()
	}
}

func TestLogosArenaParkReleasesEmptyHeadOverLiveTail(t *testing.T) {
	arenaParkCase(t, []bool{true, false}, func(addr uintptr) bool {
		noscan, scan := arenaEmptyMarks(addr)
		runtime.LogosArenaMarkRestore(addr, noscan, scan)
		return true
	})
}

func TestLogosArenaParkReleasesEmptyChunksBehindLiveHead(t *testing.T) {
	for _, tc := range []struct {
		name string
		live []bool
	}{
		{"all live", []bool{true, true}},
		{"one empty", []bool{false}},
		{"all empty", []bool{false, false}},
		{"live head over empty tail", []bool{false, true}},
		{"empty head over live tail", []bool{true, false}},
		{"empty middle", []bool{true, false, true}},
		{"alternating", []bool{false, true, false, true}},
	} {
		t.Run(tc.name, func(t *testing.T) {
			arenaParkCase(t, tc.live, runtime.LogosArenaEmptyChunk)
		})
	}
}

func TestLogosArenaExitReturnsChunk(t *testing.T) {
	_, leasedBase, _ := runtime.LogosArenaCounts()
	const size = 4096
	filled := make(chan int)
	defer debug.SetGCPercent(debug.SetGCPercent(-1))
	runtime.GC()
	go func() {
		runtime.LogosArenaAllocScan((*int)(nil), size)
		runtime.LogosArenaAlloc(byte(0), 64)
		_, leasedHeld, _ := runtime.LogosArenaCounts()
		filled <- leasedHeld
	}()
	if held := <-filled; held <= leasedBase {
		t.Fatalf("leased = %d while the exiting goroutine held arena memory, want above %d (the exit probe never leased a chunk)", held, leasedBase)
	}
	if leased := waitLeased(t, leasedBase); leased != leasedBase {
		t.Fatalf("leased = %d after the goroutine exited, want %d", leased, leasedBase)
	}
}

func TestLogosArenaSetFinalizerPanics(t *testing.T) {
	type T struct{ a, b uintptr }
	type S struct {
		p *int
		n uintptr
	}
	inFreshGoroutine(t, func() {
		m := runtime.LogosArenaSave()
		defer runtime.LogosArenaRestore(m)
		noscan := (*T)(runtime.LogosArenaAlloc(uintptr(0), unsafe.Sizeof(T{})))
		scan := (*S)(runtime.LogosArenaAllocScan((*int)(nil), unsafe.Sizeof(S{})))
		const want = "runtime: SetFinalizer on loGo/s arena object"
		for name, obj := range map[string]any{"noscan": noscan, "scan": scan} {
			msg, panicked := recoverMsg(func() { runtime.SetFinalizer(obj, nil) })
			if !panicked || msg != want {
				t.Errorf("SetFinalizer on %s arena object: panicked=%v msg=%q, want %q", name, panicked, msg, want)
			}
		}
		msg, panicked := recoverMsg(func() { runtime.SetFinalizer(noscan, func(*T) {}) })
		if !panicked || msg != want {
			t.Errorf("SetFinalizer with a finalizer: panicked=%v msg=%q, want %q", panicked, msg, want)
		}
		wantCleanup := "runtime: AddCleanup on loGo/s arena object"
		msg, panicked = recoverMsg(func() { runtime.AddCleanup(noscan, func(int) {}, 0) })
		if !panicked || msg != wantCleanup {
			t.Errorf("AddCleanup on arena object: panicked=%v msg=%q, want %q", panicked, msg, wantCleanup)
		}
		heap := new(T)
		if msg, panicked := recoverMsg(func() { runtime.SetFinalizer(heap, func(*T) {}) }); panicked {
			t.Errorf("SetFinalizer on a heap object panicked: %s", msg)
		}
		big := runtime.LogosArenaAlloc(byte(0), 1<<20)
		if runtime.LogosArenaInChunk(big) {
			t.Fatal("oversize request was served from a chunk")
		}
		if msg, panicked := recoverMsg(func() { runtime.SetFinalizer((*byte)(big), func(*byte) {}) }); panicked {
			t.Errorf("SetFinalizer on a mallocgc fallback object panicked: %s", msg)
		}
	})
}

func TestLogosArenaCgoChecksAcceptArenaAddresses(t *testing.T) {
	type holder struct{ p *int }
	inFreshGoroutine(t, func() {
		m := runtime.LogosArenaSave()
		defer runtime.LogosArenaRestore(m)
		leaf := (*int)(runtime.LogosArenaAlloc(0, 8))
		h := (*holder)(runtime.LogosArenaAllocScan((*int)(nil), unsafe.Sizeof(holder{})))
		h.p = leaf
		bytes := unsafe.Slice((*byte)(runtime.LogosArenaAlloc(byte(0), 32)), 32)
		heapLeaf := new(int)
		if !runtime.LogosArenaIsGoPointer(unsafe.Pointer(leaf)) || !runtime.LogosArenaIsGoPointer(unsafe.Pointer(h)) {
			t.Error("an arena address is not a Go pointer for the cgo checks")
		}
		cases := []struct {
			name string
			fn   func()
		}{
			{"pointer to noscan object", func() { runtime.LogosArenaCgoCheckPointer(leaf, true) }},
			{"pointer to scan object", func() { runtime.LogosArenaCgoCheckPointer(h, true) }},
			{"unsafe.Pointer to arena", func() { runtime.LogosArenaCgoCheckPointer(unsafe.Pointer(h), true) }},
			{"slice over arena bytes", func() { runtime.LogosArenaCgoCheckPointer(&bytes[0], bytes) }},
			{"arena struct holding an arena pointer", func() { runtime.LogosArenaCgoCheckPointer(*h, nil) }},
			{"result holding an arena pointer", func() { runtime.LogosArenaCgoCheckResult(h) }},
			{"heap pointer stored into arena memory", func() {
				runtime.LogosArenaCgoCheckPtrWrite((*unsafe.Pointer)(unsafe.Pointer(&h.p)), unsafe.Pointer(heapLeaf))
			}},
			{"arena pointer stored into heap memory", func() {
				var slot *int
				runtime.LogosArenaCgoCheckPtrWrite((*unsafe.Pointer)(unsafe.Pointer(&slot)), unsafe.Pointer(leaf))
			}},
		}
		for _, c := range cases {
			if msg, panicked := recoverMsg(c.fn); panicked {
				t.Errorf("%s: cgo check panicked: %s", c.name, msg)
			}
		}
	})
}

func TestLogosArenaNoscanGrowsUp(t *testing.T) {
	inFreshGoroutine(t, func() {
		m := runtime.LogosArenaSave()
		defer runtime.LogosArenaRestore(m)
		var prev uintptr
		for i, size := range []uintptr{8, 24, 40, 128} {
			p := runtime.LogosArenaAlloc(byte(0), size)
			if p == nil {
				t.Fatalf("noscan alloc %d of %d bytes returned nil", i, size)
			}
			if addr := uintptr(p); i == 0 {
				prev = addr
			} else if addr <= prev {
				t.Errorf("noscan object %d at %#x, want above %#x", i, addr, prev)
			} else {
				prev = addr
			}
		}
	})
}

func TestLogosArenaScanGrowsDown(t *testing.T) {
	inFreshGoroutine(t, func() {
		m := runtime.LogosArenaSave()
		defer runtime.LogosArenaRestore(m)
		x := new(int)
		first := runtime.LogosArenaAllocScan((*int)(nil), 128)
		if first == nil {
			t.Fatal("first scan alloc returned nil")
		}
		fillWords(first, 128, x)
		var chunks [4]runtime.LogosArenaChunk
		n, _ := runtime.LogosArenaChainOf(runtime.LogosArenaGoid(), chunks[:])
		if n < 1 {
			t.Fatal("no arena chunk holds the first scan object")
		}
		top := chunks[0].Addr + uintptr(runtime.LogosArenaChunkBytes)
		if addr := uintptr(first); addr+128 != top {
			t.Errorf("first scan object ends at %#x, want the chunk upper bound %#x", addr+128, top)
		}
		prev := uintptr(first)
		for i, size := range []uintptr{64, 32, 16} {
			p := runtime.LogosArenaAllocScan((*int)(nil), size)
			if p == nil {
				t.Fatalf("scan alloc %d of %d bytes returned nil", i, size)
			}
			fillWords(p, size, x)
			if addr := uintptr(p); addr >= prev {
				t.Errorf("scan object %d at %#x, want below %#x", i, addr, prev)
			} else {
				prev = addr
			}
		}
	})
}

func TestLogosArenaOneByteOverflowLinksChunk(t *testing.T) {
	inFreshGoroutine(t, func() {
		m := runtime.LogosArenaSave()
		const capacity = runtime.LogosArenaCapacity
		a := runtime.LogosArenaAlloc(byte(0), capacity-8)
		fillBytes(a, capacity-8, 0x11)
		b := runtime.LogosArenaAlloc(byte(0), 9)
		fillBytes(b, 9, 0x22)
		st := runtime.LogosArenaStat()
		if st.Chunks != 2 || st.NoscanUsed != 9 {
			t.Errorf("after the overflow: %+v, want two chunks and 9 bytes in the newest", st)
		}
		if !runtime.LogosArenaInChunk(b) {
			t.Error("the overflowing object was not served from a chunk")
		}
		if !checkBytes(a, capacity-8, 0x11) || !checkBytes(b, 9, 0x22) {
			t.Error("an object was damaged by the chunk link")
		}
		runtime.LogosArenaRestore(m)
		if st := runtime.LogosArenaStat(); st.Chunks != 1 || st.NoscanUsed != 0 || st.ScanUsed != 0 {
			t.Errorf("after restore: %+v, want the first chunk empty and the overflow chunk returned", st)
		}
	})
}

func TestLogosArenaScanOverflowLinksChunk(t *testing.T) {
	inFreshGoroutine(t, func() {
		m := runtime.LogosArenaSave()
		defer runtime.LogosArenaRestore(m)
		x := new(int)
		first := uintptr(runtime.LogosArenaCapacity) - runtime.LogosArenaScanHeaderBytes
		a := runtime.LogosArenaAllocScan((*int)(nil), first)
		fillWords(a, first, x)
		b := runtime.LogosArenaAllocScan((*int)(nil), 8)
		fillWords(b, 8, x)
		if st := runtime.LogosArenaStat(); st.Chunks != 2 {
			t.Errorf("chunks = %d after a scan overflow, want 2", st.Chunks)
		}
		words := unsafe.Slice((**int)(a), first/8)
		for i, w := range words {
			if w != x {
				t.Fatalf("word %d of the first scan object was damaged", i)
			}
		}
		if *(**int)(b) != x {
			t.Error("the second scan object was damaged")
		}
	})
}

func TestLogosArenaExactFitBoundaryKeepsChunk(t *testing.T) {
	for _, row := range []struct {
		name string
		side string
	}{
		{name: "noscan", side: "noscan"},
		{name: "scan", side: "scan"},
	} {
		t.Run(row.name, func(t *testing.T) {
			inFreshGoroutine(t, func() {
				m := runtime.LogosArenaSave()
				x := new(int)
				fill := uintptr(runtime.LogosArenaCapacity)
				exact := uintptr(0)
				over := uintptr(0)
				if row.side == "noscan" {
					fill -= 8
					exact = 8
					over = 9
				} else {
					fill -= 2*uintptr(runtime.LogosArenaScanHeaderBytes) + 16
					exact = 16
					over = uintptr(runtime.LogosArenaScanHeaderBytes) + 16 + 1
				}
				var a unsafe.Pointer
				if row.side == "noscan" {
					a = runtime.LogosArenaAlloc(byte(0), fill)
					fillBytes(a, fill, 0x11)
				} else {
					a = runtime.LogosArenaAllocScan((*int)(nil), fill)
					fillWords(a, fill, x)
				}
				_ = a
				before := runtime.LogosArenaStat().Chunks
				var got unsafe.Pointer
				if row.side == "noscan" {
					got = runtime.LogosArenaAlloc(byte(0), exact)
					fillBytes(got, exact, 0x22)
				} else {
					got = runtime.LogosArenaAllocScan((*int)(nil), exact)
					fillWords(got, exact, x)
				}
				if got == nil {
					t.Fatal("exact-fit allocation did not succeed")
				}
				if st := runtime.LogosArenaStat(); st.Chunks != before {
					t.Errorf("exact fit linked a chunk: chunks %d, want %d", st.Chunks, before)
				}
				runtime.LogosArenaRestore(m)
				m = runtime.LogosArenaSave()
				if row.side == "noscan" {
					a := runtime.LogosArenaAlloc(byte(0), fill)
					fillBytes(a, fill, 0x11)
				} else {
					a := runtime.LogosArenaAllocScan((*int)(nil), fill)
					fillWords(a, fill, x)
				}
				before = runtime.LogosArenaStat().Chunks
				if row.side == "noscan" {
					got = runtime.LogosArenaAlloc(byte(0), over)
					fillBytes(got, over, 0x33)
				} else {
					got = runtime.LogosArenaAllocScan((*int)(nil), over)
					fillWords(got, over, x)
				}
				if got == nil {
					t.Fatal("overflow allocation did not succeed")
				}
				if st := runtime.LogosArenaStat(); st.Chunks != before+1 {
					t.Errorf("one byte over the gap did not link a chunk: chunks %d, want %d", st.Chunks, before+1)
				}
				runtime.LogosArenaRestore(m)
			})
		})
	}
}

func TestLogosArenaNoscanReuseIsZeroed(t *testing.T) {
	inFreshGoroutine(t, func() {
		m := runtime.LogosArenaSave()
		p := runtime.LogosArenaAlloc(byte(0), 256)
		fillBytes(p, 256, 0xFF)
		runtime.LogosArenaRestore(m)
		m = runtime.LogosArenaSave()
		defer runtime.LogosArenaRestore(m)
		q := runtime.LogosArenaAlloc(byte(0), 256)
		if !checkBytes(q, 256, 0) {
			t.Error("reused noscan memory is not zero")
		}
	})
}

func TestLogosArenaScanLayoutAndZeroing(t *testing.T) {
	inFreshGoroutine(t, func() {
		m := runtime.LogosArenaSave()
		x := new(int)
		sizes := []uintptr{8, 24, 40, 128}
		objs := make([]unsafe.Pointer, len(sizes))
		for i, size := range sizes {
			objs[i] = runtime.LogosArenaAllocScan((*int)(nil), size)
			fillWords(objs[i], size, x)
		}
		typ := runtime.LogosArenaType((*int)(nil))
		for i, p := range objs {
			gotTyp, gotSize := runtime.LogosArenaScanHeader(p)
			if gotTyp != typ || gotSize != sizes[i] {
				t.Errorf("object %d header = (%p, %d), want (%p, %d)", i, gotTyp, gotSize, typ, sizes[i])
			}
		}
		runtime.LogosArenaRestore(m)
		if st := runtime.LogosArenaStat(); st.ScanUsed != 0 {
			t.Errorf("ScanUsed = %d after restore, want 0", st.ScanUsed)
		}
		m = runtime.LogosArenaSave()
		defer runtime.LogosArenaRestore(m)
		for i, size := range sizes {
			q := runtime.LogosArenaAllocScan((*int)(nil), size)
			if !checkBytes(q, size, 0) {
				t.Errorf("reused scan object %d memory is not zero", i)
			}
		}
	})
}

func TestLogosArenaRestorePopsLinkedChunks(t *testing.T) {
	inFreshGoroutine(t, func() {
		m := runtime.LogosArenaSave()
		x := new(int)
		var objs []unsafe.Pointer
		const size = 30000
		for range 4 {
			p := runtime.LogosArenaAllocScan((*int)(nil), size)
			fillWords(p, size, x)
			objs = append(objs, p)
		}
		if st := runtime.LogosArenaStat(); st.Chunks < 2 {
			t.Fatalf("chunks = %d, want the scan objects to span several chunks", st.Chunks)
		}
		runtime.LogosArenaRestore(m)
		if st := runtime.LogosArenaStat(); st.Chunks != 1 || st.ScanUsed != 0 || st.NoscanUsed != 0 {
			t.Errorf("after restore: %+v, want one empty chunk", st)
		}
		var qs []unsafe.Pointer
		for range 4 {
			q := runtime.LogosArenaAllocScan((*int)(nil), size)
			qs = append(qs, q)
		}
		for i, q := range qs {
			if !checkBytes(q, size, 0) {
				t.Errorf("reused scan object %d memory is not zero after restore", i)
			}
		}
	})
}

func TestLogosArenaGCFreesCachedChunks(t *testing.T) {
	const n = 512
	goroutines := runtime.NumGoroutine()
	runtime.GC()
	_, leasedBase, _ := runtime.LogosArenaCounts()
	var before runtime.MemStats
	runtime.ReadMemStats(&before)
	hold := make(chan struct{})
	var held, finished sync.WaitGroup
	held.Add(n)
	finished.Add(n)
	for range n {
		go func() {
			defer finished.Done()
			m := runtime.LogosArenaSave()
			runtime.LogosArenaAlloc(byte(0), 100)
			held.Done()
			<-hold
			runtime.LogosArenaRestore(m)
		}()
	}
	held.Wait()
	var during runtime.MemStats
	runtime.ReadMemStats(&during)
	if got, want := during.StackInuse-before.StackInuse, uint64(n)*runtime.LogosArenaChunkBytes; got < want {
		t.Errorf("StackInuse grew by %d with %d chunks leased, want at least %d", got, n, want)
	}
	if _, leased, _ := runtime.LogosArenaCounts(); leased < leasedBase+n {
		t.Errorf("leased = %d with %d goroutines holding arena memory, want at least %d", leased, n, leasedBase+n)
	}
	close(hold)
	finished.Wait()
	waitGoroutines(t, goroutines)
	if leased := waitLeased(t, leasedBase); leased != leasedBase {
		t.Errorf("leased = %d after parked goroutines exited, want %d", leased, leasedBase)
	}
	runtime.GC()
	total, leased, pooled := runtime.LogosArenaCounts()
	if pooled != 0 || total != leased {
		t.Errorf("after GC: total=%d leased=%d pooled=%d, want every cached chunk freed", total, leased, pooled)
	}
}

func TestLogosArenaProcResizeFreesCachedChunks(t *testing.T) {
	procs := runtime.GOMAXPROCS(0)
	if procs < 2 {
		t.Skip("needs at least two Ps")
	}
	goroutines := runtime.NumGoroutine()
	const n = 256
	_, leasedBase, _ := runtime.LogosArenaCounts()
	hold := make(chan struct{})
	var held, finished sync.WaitGroup
	held.Add(n)
	finished.Add(n)
	for range n {
		go func() {
			defer finished.Done()
			m := runtime.LogosArenaSave()
			runtime.LogosArenaAlloc(byte(0), 100)
			held.Done()
			<-hold
			runtime.LogosArenaRestore(m)
		}()
	}
	held.Wait()
	if _, leased, _ := runtime.LogosArenaCounts(); leased < leasedBase+n {
		t.Errorf("leased = %d with %d goroutines holding arena memory, want at least %d", leased, n, leasedBase+n)
	}
	close(hold)
	finished.Wait()
	waitGoroutines(t, goroutines)
	before := runtime.LogosArenaPoolCounts()
	old := runtime.GOMAXPROCS(1)
	defer runtime.GOMAXPROCS(old)
	total, leased, pooled := runtime.LogosArenaCounts()
	if total != leased+pooled {
		t.Errorf("after shrinking to one P: total=%d leased=%d pooled=%d, want total = leased + pooled", total, leased, pooled)
	}
	if pooled > before[0] {
		t.Errorf("pooled = %d after shrinking to one P, which held %d: a chunk outlived its destroyed P", pooled, before[0])
	}
}

//go:noinline
func arenaEligibleMake(n int) uintptr {
	b := make([]byte, n)
	if len(b) > 0 {
		b[0] = 42
	}
	during := runtime.LogosArenaStat().NoscanUsed
	if len(b) > 0 && b[0] != 42 {
		panic("memory corrupted")
	}
	return during
}

func TestLogosArenaDynamicMakeWatermark(t *testing.T) {
	if !goexperiment.LogosArena {
		t.Skip("skipping test: GOEXPERIMENT=logosarena not enabled")
	}
	inFreshGoroutine(t, func() {
		before := runtime.LogosArenaStat().NoscanUsed

		during1 := arenaEligibleMake(1024)
		after1 := runtime.LogosArenaStat().NoscanUsed

		if during1 <= before {
			t.Errorf("call 1: watermark during make = %d, want > %d (watermark did not advance)", during1, before)
		}
		if after1 != before {
			t.Errorf("call 1: watermark after frame exit = %d, want %d (watermark did not fall back)", after1, before)
		}

		during2 := arenaEligibleMake(2048)
		after2 := runtime.LogosArenaStat().NoscanUsed

		if during2 <= before {
			t.Errorf("call 2: watermark during make = %d, want > %d (watermark did not advance)", during2, before)
		}
		if after2 != before {
			t.Errorf("call 2: watermark after frame exit = %d, want %d (watermark did not fall back)", after2, before)
		}
	})
}

var arenaPanicWatermark uintptr

//go:noinline
func arenaMakeAndPanic(n int) {
	b := make([]byte, n)
	if len(b) > 0 {
		b[0] = 7
	}
	arenaPanicWatermark = runtime.LogosArenaStat().NoscanUsed
	panic("arena panic unwind test")
}

func TestLogosArenaPanicUnwindRecovery(t *testing.T) {
	if !goexperiment.LogosArena {
		t.Skip("skipping test: GOEXPERIMENT=logosarena not enabled")
	}
	inFreshGoroutine(t, func() {
		before := runtime.LogosArenaStat().NoscanUsed

		func() {
			defer func() {
				_ = recover()
			}()
			arenaMakeAndPanic(1024)
		}()

		after := runtime.LogosArenaStat().NoscanUsed
		if arenaPanicWatermark <= before {
			t.Fatalf("panic probe allocated nothing in the arena: watermark during make = %d, want above %d (the unwind restore is not exercised)", arenaPanicWatermark, before)
		}
		if after != before {
			t.Errorf("watermark leaked across recovered panic: before=%d, after=%d (G3 panic-unwind restore missing)", before, after)
		}
	})
}

//go:noinline
func arenaMakeLoop(iters, allocSize int) uintptr {
	var lastWatermark uintptr
	for i := range iters {
		b := make([]byte, allocSize)
		if len(b) > 0 {
			b[0] = byte(i)
		}
		lastWatermark = runtime.LogosArenaStat().NoscanUsed
	}
	return lastWatermark
}

func TestLogosArenaLoopWatermarkBounded(t *testing.T) {
	if !goexperiment.LogosArena {
		t.Skip("skipping test: GOEXPERIMENT=logosarena not enabled")
	}
	inFreshGoroutine(t, func() {
		before := runtime.LogosArenaStat().NoscanUsed

		wm1 := arenaMakeLoop(1, 1024)
		used1 := wm1 - before
		if used1 == 0 {
			t.Fatalf("loop probe allocated nothing in the arena: 1-iter used 0 B, the attack is not live")
		}

		wm10 := arenaMakeLoop(10, 1024)
		used10 := wm10 - before

		if used10 > used1*2 {
			t.Errorf("loop arena watermark grew unboundedly: 1-iter used %d B, 10-iter used %d B (G4 loop back-edge restore missing)", used1, used10)
		}
	})
}

//go:noinline
func arenaMakeSwitchBreakLoop(outerIters, allocSize int) uintptr {
	var lastWatermark uintptr
	for o := range outerIters {
	SwitchOuter:
		switch o >= 0 {
		case true:
			for i := range 5 {
				b := make([]byte, allocSize)
				if len(b) > 0 {
					b[0] = byte(i + o)
				}
				lastWatermark = runtime.LogosArenaStat().NoscanUsed
				if i == 2 {
					break SwitchOuter
				}
			}
		case false:
		}
	}
	return lastWatermark
}

func TestLogosArenaSwitchBreakWatermarkBounded(t *testing.T) {
	if !goexperiment.LogosArena {
		t.Skip("skipping test: GOEXPERIMENT=logosarena not enabled")
	}
	if runtime.GOARCH != "amd64" {
		t.Skip("skipping test: arena make lowering is amd64-only")
	}
	inFreshGoroutine(t, func() {
		before := runtime.LogosArenaStat().NoscanUsed

		wm1 := arenaMakeSwitchBreakLoop(1, 256)
		used1 := wm1 - before
		if used1 == 0 {
			t.Fatalf("switch-break probe allocated nothing in the arena: 1-iter used 0 B, the attack is not live")
		}

		wm10 := arenaMakeSwitchBreakLoop(10, 256)
		used10 := wm10 - before

		if used10 > used1*2 {
			t.Errorf("switch-break arena watermark grew unboundedly: 1-iter used %d B, 10-iter used %d B (enclosing-switch exit restore missing)", used1, used10)
		}
	})
}

//go:noinline
func arenaMakeSelectBreakLoop(outerIters, allocSize int, ch chan int) uintptr {
	var lastWatermark uintptr
	for o := range outerIters {
	SelectOuter:
		select {
		case <-ch:
			for i := range 5 {
				b := make([]byte, allocSize)
				if len(b) > 0 {
					b[0] = byte(i + o)
				}
				lastWatermark = runtime.LogosArenaStat().NoscanUsed
				if i == 2 {
					break SelectOuter
				}
			}
		default:
		}
	}
	return lastWatermark
}

func TestLogosArenaSelectBreakWatermarkBounded(t *testing.T) {
	if !goexperiment.LogosArena {
		t.Skip("skipping test: GOEXPERIMENT=logosarena not enabled")
	}
	if runtime.GOARCH != "amd64" {
		t.Skip("skipping test: arena make lowering is amd64-only")
	}
	inFreshGoroutine(t, func() {
		ch := make(chan int)
		close(ch)
		before := runtime.LogosArenaStat().NoscanUsed

		wm1 := arenaMakeSelectBreakLoop(1, 256, ch)
		used1 := wm1 - before
		if used1 == 0 {
			t.Fatalf("select-break probe allocated nothing in the arena: 1-iter used 0 B, the attack is not live")
		}

		wm10 := arenaMakeSelectBreakLoop(10, 256, ch)
		used10 := wm10 - before

		if used10 > used1*2 {
			t.Errorf("select-break arena watermark grew unboundedly: 1-iter used %d B, 10-iter used %d B (enclosing-select exit restore missing)", used1, used10)
		}
	})
}

const (
	arenaMakeLenPanic = "runtime error: makeslice: len out of range"
	arenaMakeCapPanic = "runtime error: makeslice: cap out of range"

	arenaMakeAllocSlack = 1 << 20
)

type arenaMakeCase struct {
	len, cap int64
	want     string
}

type arenaElem24 [3]int64

//go:noinline
func arenaSeamInt64(n, c int) (length, capacity int, used uintptr) {
	b := make([]int64, n, c)
	return len(b), cap(b), runtime.LogosArenaStat().NoscanUsed
}

//go:noinline
func arenaSeamElem24(n, c int) (int, int) {
	b := make([]arenaElem24, n, c)
	return len(b), cap(b)
}

//go:noinline
func arenaSeamByte(n, c int) (int, int) {
	b := make([]byte, n, c)
	return len(b), cap(b)
}

//go:noinline
func arenaSeamUint32Wide(n, c uint64) (int, int) {
	b := make([]uint32, n, c)
	return len(b), cap(b)
}

//go:noinline
func arenaSeamFill(n, c int) bool {
	b := make([]int64, n, c)
	if len(b) != n || cap(b) != c {
		return false
	}
	b = b[:cap(b)]
	for i := range b {
		if b[i] != 0 {
			return false
		}
		b[i] = int64(i) + 1
	}
	for i := range b {
		if b[i] != int64(i)+1 {
			return false
		}
	}
	return true
}

//go:noinline
func heapSeamInt64(n, c int) { arenaSink = make([]int64, n, c) }

//go:noinline
func heapSeamElem24(n, c int) { arenaSink = make([]arenaElem24, n, c) }

//go:noinline
func heapSeamByte(n, c int) { arenaSink = make([]byte, n, c) }

//go:noinline
func heapSeamUint32Wide(n, c uint64) { arenaSink = make([]uint32, n, c) }

func arenaMakeMustPanic(t *testing.T, what, want string, fn func()) {
	t.Helper()
	var before, after runtime.MemStats
	runtime.ReadMemStats(&before)
	msg, panicked := recoverMsg(fn)
	runtime.ReadMemStats(&after)
	if !panicked {
		t.Fatalf("%s: make returned without panic, want %q", what, want)
	}
	if msg != want {
		t.Errorf("%s: panic %q, want %q", what, msg, want)
	}
	if grew := after.TotalAlloc - before.TotalAlloc; grew > arenaMakeAllocSlack {
		t.Errorf("%s: panicking make allocated %d heap bytes, want at most %d", what, grew, arenaMakeAllocSlack)
	}
}

func arenaMakeOverflowCases(size uintptr) []arenaMakeCase {
	wrap := int64(^uintptr(0)/size + 1)
	minInt := int64(math.MinInt)
	return []arenaMakeCase{
		{0, wrap, arenaMakeCapPanic},
		{0, wrap + 1, arenaMakeCapPanic},
		{1, wrap, arenaMakeCapPanic},
		{1, wrap + 1, arenaMakeCapPanic},
		{wrap, wrap, arenaMakeLenPanic},
		{wrap + 1, wrap + 1, arenaMakeLenPanic},
		{wrap + 1, wrap + 2, arenaMakeLenPanic},
		{wrap, 1, arenaMakeLenPanic},
		{0, minInt, arenaMakeCapPanic},
		{minInt, 0, arenaMakeLenPanic},
		{minInt, minInt, arenaMakeLenPanic},
		{-1, 1, arenaMakeLenPanic},
		{-1, -1, arenaMakeLenPanic},
		{1, -1, arenaMakeCapPanic},
		{2, 1, arenaMakeCapPanic},
	}
}

func arenaMakeRun(t *testing.T, size uintptr, arena, heap func(n, c int)) {
	for _, tc := range arenaMakeOverflowCases(size) {
		t.Run(fmt.Sprintf("len=%d,cap=%d", tc.len, tc.cap), func(t *testing.T) {
			n, c := int(tc.len), int(tc.cap)
			arenaMakeMustPanic(t, "heap control", tc.want, func() { heap(n, c) })
			arenaMakeMustPanic(t, "arena make", tc.want, func() { arena(n, c) })
		})
	}
}

func TestLogosArenaMakeOverflow(t *testing.T) {
	if !goexperiment.LogosArena {
		t.Skip("skipping test: GOEXPERIMENT=logosarena not enabled")
	}
	t.Run("int64", func(t *testing.T) {
		arenaMakeRun(t, 8, func(n, c int) { arenaSeamInt64(n, c) }, heapSeamInt64)
	})
	t.Run("elem24", func(t *testing.T) {
		arenaMakeRun(t, 24, func(n, c int) { arenaSeamElem24(n, c) }, heapSeamElem24)
	})
}

func TestLogosArenaMakeOverflowWide(t *testing.T) {
	if !goexperiment.LogosArena {
		t.Skip("skipping test: GOEXPERIMENT=logosarena not enabled")
	}
	top := uint64(1) << 63
	wrap := uint64(1) << 62
	for _, tc := range [][2]uint64{
		{0, wrap}, {wrap, wrap}, {0, top}, {top, top}, {top, 0}, {1, top + 1}, {wrap + 1, wrap + 1},
	} {
		t.Run(fmt.Sprintf("len=%d,cap=%d", tc[0], tc[1]), func(t *testing.T) {
			n, c := tc[0], tc[1]
			msg, panicked := recoverMsg(func() { heapSeamUint32Wide(n, c) })
			if !panicked {
				t.Fatal("heap control: make returned without panic")
			}
			arenaMakeMustPanic(t, "arena make", msg, func() { arenaSeamUint32Wide(n, c) })
		})
	}
}

func TestLogosArenaMakeOverflowMaxAlloc(t *testing.T) {
	if !goexperiment.LogosArena {
		t.Skip("skipping test: GOEXPERIMENT=logosarena not enabled")
	}
	wrapsToZero := int(^uintptr(0)/8 + 1)
	if _, panicked := recoverMsg(func() { arenaSeamInt64(0, wrapsToZero) }); !panicked {
		t.Fatal("make([]int64, 0, cap) with a wrapping byte size returned without panic; refusing to attempt sizes past maxAlloc")
	}
	const pastInt64 = int(runtime.LogosArenaMaxAlloc/8 + 1)
	for _, tc := range []struct {
		name string
		n, c int
		want string
	}{
		{"cap", 0, pastInt64, arenaMakeCapPanic},
		{"len", pastInt64, pastInt64, arenaMakeLenPanic},
		{"maxint-cap", 0, math.MaxInt, arenaMakeCapPanic},
		{"maxint-len", math.MaxInt, math.MaxInt, arenaMakeLenPanic},
	} {
		t.Run("int64/"+tc.name, func(t *testing.T) {
			arenaMakeMustPanic(t, "heap control", tc.want, func() { heapSeamInt64(tc.n, tc.c) })
			arenaMakeMustPanic(t, "arena make", tc.want, func() { arenaSeamInt64(tc.n, tc.c) })
		})
	}
	if bits.UintSize == 64 {
		maxAlloc := runtime.LogosArenaMaxAlloc
		pastByte := int(maxAlloc) + 1
		t.Run("byte/cap", func(t *testing.T) {
			arenaMakeMustPanic(t, "heap control", arenaMakeCapPanic, func() { heapSeamByte(0, pastByte) })
			arenaMakeMustPanic(t, "arena make", arenaMakeCapPanic, func() { arenaSeamByte(0, pastByte) })
		})
		t.Run("byte/len", func(t *testing.T) {
			arenaMakeMustPanic(t, "heap control", arenaMakeLenPanic, func() { heapSeamByte(pastByte, pastByte) })
			arenaMakeMustPanic(t, "arena make", arenaMakeLenPanic, func() { arenaSeamByte(pastByte, pastByte) })
		})
	}
}

func TestLogosArenaMakeSeamPlacement(t *testing.T) {
	if !goexperiment.LogosArena {
		t.Skip("skipping test: GOEXPERIMENT=logosarena not enabled")
	}
	if runtime.GOARCH != "amd64" {
		t.Skip("skipping test: arena make lowering is amd64-only")
	}
	inFreshGoroutine(t, func() {
		before := runtime.LogosArenaStat().NoscanUsed
		n, c, during := arenaSeamInt64(64, 96)
		after := runtime.LogosArenaStat().NoscanUsed
		if n != 64 || c != 96 {
			t.Errorf("make([]int64, 64, 96) has len %d cap %d", n, c)
		}
		if want := before + 96*8; during < want {
			t.Errorf("watermark during make = %d, want at least %d (probe site is not arena-placed)", during, want)
		}
		if after != before {
			t.Errorf("watermark after frame exit = %d, want %d", after, before)
		}
		for _, tc := range [][2]int{{0, 0}, {1, 1}, {3, 5}, {64, 64}, {513, 1024}, {int(runtime.LogosArenaChunkBytes/8) + 1, int(runtime.LogosArenaChunkBytes/8) + 1}} {
			if !arenaSeamFill(tc[0], tc[1]) {
				t.Errorf("make([]int64, %d, %d) is not zeroed, writable, and cap-sized", tc[0], tc[1])
			}
		}
	})
}

//go:noinline
func arenaOwnIntInScan(done chan struct{}) unsafe.Pointer {
	obj := new(int)
	runtime.AddCleanup(obj, func(c chan struct{}) { close(c) }, done)
	p := runtime.LogosArenaAllocScan((*int)(nil), 64)
	for i := range 64 {
		*(*byte)(unsafe.Pointer(uintptr(p) + uintptr(i))) = 0xAA
	}
	*(**int)(p) = obj
	return p
}

func TestLogosArenaRestoreDropsScanRoots(t *testing.T) {
	inFreshGoroutine(t, func() {
		defer debug.SetGCPercent(debug.SetGCPercent(-1))
		runtime.GC()
		m := runtime.LogosArenaSave()
		done := make(chan struct{})
		p := arenaOwnIntInScan(done)
		runtime.GC()
		runtime.GC()
		select {
		case <-done:
			t.Fatal("cleanup ran while the arena still held the only reference")
		default:
		}
		runtime.LogosArenaRestore(m)
		_ = p
		for range 100 {
			runtime.GC()
			select {
			case <-done:
				return
			default:
			}
		}
		t.Fatal("cleanup did not run within 100 GCs after restore")
	})
}

func TestLogosArenaScanRootsKeepReferentAlive(t *testing.T) {
	inFreshGoroutine(t, func() {
		defer debug.SetGCPercent(debug.SetGCPercent(-1))
		runtime.GC()
		m := runtime.LogosArenaSave()
		obj := new(int)
		*obj = 42
		done := make(chan struct{})
		runtime.AddCleanup(obj, func(c chan struct{}) { close(c) }, done)
		w := weak.Make(obj)
		p := runtime.LogosArenaAllocScan((*int)(nil), 64)
		fillBytes(p, 64, 0)
		*(**int)(p) = obj
		obj = nil
		runtime.GC()
		runtime.GC()
		kept := w.Value()
		if kept == nil {
			t.Fatal("heap referent collected while arena scan region holds the only reference")
		}
		if *kept != 42 {
			t.Fatalf("heap referent corrupted: got %d, want 42", *kept)
		}
		runtime.LogosArenaRestore(m)
		runtime.KeepAlive(p)
		deadline := time.After(30 * time.Second)
		for {
			runtime.GC()
			select {
			case <-done:
				return
			case <-deadline:
				t.Fatal("cleanup did not run within 30s after arena restore")
				return
			}
		}
	})
}
