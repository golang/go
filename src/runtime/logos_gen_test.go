// Copyright 2026 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

//go:build goexperiment.logosgen

package runtime_test

import (
	"bytes"
	"fmt"
	"runtime"
	"runtime/debug"
	"testing"
	"unsafe"
	"weak"
)

// M19-S4 attacks for GOEXPERIMENT=logosgen: a card table, minor cycles
// that skip mature spans, and a full cycle after four minor cycles.

// genHolder is a pointer-bearing object larger than the small-object
// limit, so it owns its span and the span's age is the object's age.
type genHolder struct{ slots [5000]unsafe.Pointer }

var genRoot *genHolder

// genQuiet disables heap-triggered cycles so each test controls every
// cycle and the span ages are exact.
func genQuiet(t *testing.T) {
	t.Helper()
	old := debug.SetGCPercent(-1)
	t.Cleanup(func() { debug.SetGCPercent(old) })
}

// genStoreHidden stores a young object into h.slots[0] through a uintptr
// alias: the store carries no write barrier and marks no card.
//
//go:noinline
func genStoreHidden(h *genHolder) weak.Pointer[[64]byte] {
	y := new([64]byte)
	w := weak.Make(y)
	*(*uintptr)(unsafe.Pointer(&h.slots[0])) = uintptr(unsafe.Pointer(y))
	return w
}

// genStoreBarrier stores a young object into h.slots[0] with an
// ordinary pointer store, so the write barrier marks the card.
//
//go:noinline
func genStoreBarrier(h *genHolder) weak.Pointer[[64]byte] {
	y := new([64]byte)
	w := weak.Make(y)
	h.slots[0] = unsafe.Pointer(y)
	return w
}

func genClearSlot(h *genHolder) {
	*(*uintptr)(unsafe.Pointer(&h.slots[0])) = 0
}

func TestLogosGenActive(t *testing.T) {
	if !runtime.LogosGenActive() {
		t.Fatal("logosgen is built in but the collector is not generational")
	}
}

func TestLogosGenCardTableDensity(t *testing.T) {
	if got := runtime.LogosGenCardBytes(); got != 512 {
		t.Fatalf("card covers %d bytes, want 512", got)
	}
	if got, want := runtime.LogosGenCardTableBytes(), runtime.LogosGenArenaBytes()/512; got != want {
		t.Fatalf("card table is %d bytes per arena, want %d (1 byte per 512 bytes of heap)", got, want)
	}
	h := new(genHolder)
	base := uintptr(unsafe.Pointer(h)) &^ 511
	first := runtime.LogosGenCardAddr(base)
	if first == 0 {
		t.Fatal("heap address has no card")
	}
	if got := runtime.LogosGenCardAddr(base + 511); got != first {
		t.Errorf("addresses 511 bytes apart use cards %#x and %#x, want one card", first, got)
	}
	if got := runtime.LogosGenCardAddr(base + 512); got != first+1 {
		t.Errorf("next 512 bytes use card %#x, want %#x (one byte per 512 bytes)", got, first+1)
	}
	runtime.KeepAlive(h)
}

func TestLogosGenOldToYoungKeepsYoungAlive(t *testing.T) {
	genQuiet(t)
	runtime.GC()
	genRoot = new(genHolder)
	defer func() { genRoot = nil }()
	for range runtime.LogosGenMatureAge() {
		runtime.LogosGenMinorCycle()
	}
	if age := runtime.LogosGenSpanAge(unsafe.Pointer(genRoot)); age < runtime.LogosGenMatureAge() {
		t.Fatalf("old object span age = %d, want mature", age)
	}

	y := new([64]byte)
	w := weak.Make(y)
	if !runtime.LogosGenStoreFlush(&genRoot.slots[0], unsafe.Pointer(y)) {
		t.Fatal("pointer store into a mature span did not mark its card")
	}
	y = nil
	runtime.LogosGenMinorCycle()
	if w.Value() == nil {
		t.Fatal("young object referenced only from an old object was freed by a minor cycle")
	}
	genClearSlot(genRoot)
}

func TestLogosGenMinorTriggerAtFiftyPercent(t *testing.T) {
	const marked = 200 << 20
	for _, tc := range []struct {
		name    string
		growth  uint64
		trigger bool
	}{
		{"49%", 49, false},
		{"50%", 50, true},
		{"51%", 51, true},
	} {
		live := uint64(marked) + uint64(marked)*tc.growth/100
		pacer := uint64(marked) * 3
		if got := live >= runtime.LogosGenHeapTrigger(pacer, marked, 4<<20); got != tc.trigger {
			t.Errorf("growth %s: heap trigger fires = %v, want %v", tc.name, got, tc.trigger)
		}
	}
	if got := runtime.LogosGenHeapTrigger(1<<30, 1<<20, 4<<20); got != 4<<20 {
		t.Errorf("small heap trigger = %d, want the heap minimum %d", got, 4<<20)
	}
	if got := runtime.LogosGenHeapTrigger(150<<20, marked, 4<<20); got != 150<<20 {
		t.Errorf("pacer trigger below 50%% growth = %d, want it kept", got)
	}
}

func TestLogosGenMaturityScansOnlyDirtyCards(t *testing.T) {
	genQuiet(t)
	for _, survived := range []int{1, 2, 3} {
		for _, dirty := range []bool{false, true} {
			t.Run(fmt.Sprintf("survived=%d/dirty=%v", survived, dirty), func(t *testing.T) {
				runtime.GC()
				genRoot = new(genHolder)
				defer func() { genRoot = nil }()
				for range survived {
					runtime.LogosGenMinorCycle()
				}
				if age, want := runtime.LogosGenSpanAge(unsafe.Pointer(genRoot)), min(survived, runtime.LogosGenMatureAge()); age != want {
					t.Fatalf("span age = %d after %d survived cycles, want %d", age, survived, want)
				}
				mature := survived >= runtime.LogosGenMatureAge()

				var w weak.Pointer[[64]byte]
				if dirty {
					w = genStoreBarrier(genRoot)
				} else {
					w = genStoreHidden(genRoot)
				}
				runtime.LogosGenMinorCycle()
				alive := w.Value() != nil
				genClearSlot(genRoot)

				wantAlive := dirty || !mature
				if alive != wantAlive {
					t.Fatalf("mature=%v dirty=%v: young object alive = %v, want %v (a clean mature span must be skipped, a young or dirty one scanned)",
						mature, dirty, alive, wantAlive)
				}
			})
		}
	}
}

func TestLogosGenMatureGarbageWaitsForFullCycle(t *testing.T) {
	genQuiet(t)
	runtime.GC()
	genRoot = new(genHolder)
	w := weak.Make(genRoot)
	for range runtime.LogosGenMatureAge() {
		runtime.LogosGenMinorCycle()
	}
	genRoot = nil
	for i := runtime.LogosGenMatureAge(); i < runtime.LogosGenFullEvery(); i++ {
		runtime.LogosGenMinorCycle()
		if w.Value() == nil {
			t.Fatalf("mature garbage was reclaimed by minor cycle %d", i+1)
		}
	}
	runtime.LogosGenAutoCycle()
	if w.Value() != nil {
		t.Fatalf("the cycle after %d minor cycles did not reclaim unreachable mature garbage", runtime.LogosGenFullEvery())
	}
}

func TestLogosGenFullCycleFollowsFourMinorCycles(t *testing.T) {
	genQuiet(t)
	runtime.GC()
	runtime.LogosGenResetHistory()
	for range 10 {
		runtime.LogosGenAutoCycle()
	}
	if got, want := string(runtime.LogosGenHistory()), "mmmmFmmmmF"; got != want {
		t.Fatalf("cycle kinds = %s, want %s", got, want)
	}
}

var genChurnSink []byte

// genHeapCycleKinds allocates until n heap-triggered cycles have run and
// returns their kinds.
func genHeapCycleKinds(n int) []byte {
	runtime.GC()
	runtime.LogosGenResetHistory()
	for i := 0; runtime.LogosGenCycles() < n && i < 4_000_000; i++ {
		genChurnSink = make([]byte, 1<<17)
	}
	return runtime.LogosGenHistory()
}

func TestLogosGenHeapTriggeredCyclesAreMinorUntilFull(t *testing.T) {
	kinds := genHeapCycleKinds(12)
	if len(kinds) < 12 {
		t.Fatalf("only %d heap-triggered cycles ran", len(kinds))
	}
	run := 0
	for i, k := range kinds {
		switch k {
		case 'm':
			run++
		case 'F':
			run = 0
		default:
			t.Fatalf("unknown cycle kind %q in %s", k, kinds)
		}
		if run > runtime.LogosGenFullEvery() {
			t.Fatalf("cycle %d is minor number %d in a row: history %s", i, run, kinds)
		}
	}
	if !bytes.Contains(kinds, []byte("mmmmF")) {
		t.Fatalf("no full cycle follows four minor cycles in %s", kinds)
	}
}
