// Copyright 2013 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

package runtime_test

import (
	"flag"
	"fmt"
	"internal/race"
	"internal/testenv"
	"os"
	"os/exec"
	"reflect"
	"runtime"
	. "runtime"
	"strings"
	"sync/atomic"
	"testing"
	"time"
	"unsafe"
)

var testMemStatsCount int

func TestMemStats(t *testing.T) {
	testMemStatsCount++

	// Make sure there's at least one forced GC.
	GC()

	// Test that MemStats has sane values.
	st := new(MemStats)
	ReadMemStats(st)

	nz := func(x any) error {
		if x != reflect.Zero(reflect.TypeOf(x)).Interface() {
			return nil
		}
		return fmt.Errorf("zero value")
	}
	le := func(thresh float64) func(any) error {
		return func(x any) error {
			// These sanity tests aren't necessarily valid
			// with high -test.count values, so only run
			// them once.
			if testMemStatsCount > 1 {
				return nil
			}

			if reflect.ValueOf(x).Convert(reflect.TypeOf(thresh)).Float() < thresh {
				return nil
			}
			return fmt.Errorf("insanely high value (overflow?); want <= %v", thresh)
		}
	}
	eq := func(x any) func(any) error {
		return func(y any) error {
			if x == y {
				return nil
			}
			return fmt.Errorf("want %v", x)
		}
	}
	// Of the uint fields, HeapReleased, HeapIdle can be 0.
	// PauseTotalNs can be 0 if timer resolution is poor.
	fields := map[string][]func(any) error{
		"Alloc": {nz, le(1e10)}, "TotalAlloc": {nz, le(1e11)}, "Sys": {nz, le(1e10)},
		"Lookups": {eq(uint64(0))}, "Mallocs": {nz, le(1e10)}, "Frees": {nz, le(1e10)},
		"HeapAlloc": {nz, le(1e10)}, "HeapSys": {nz, le(1e10)}, "HeapIdle": {le(1e10)},
		"HeapInuse": {nz, le(1e10)}, "HeapReleased": {le(1e10)}, "HeapObjects": {nz, le(1e10)},
		"StackInuse": {nz, le(1e10)}, "StackSys": {nz, le(1e10)},
		"MSpanInuse": {nz, le(1e10)}, "MSpanSys": {nz, le(1e10)},
		"MCacheInuse": {nz, le(1e10)}, "MCacheSys": {nz, le(1e10)},
		"BuckHashSys": {nz, le(1e10)}, "GCSys": {nz, le(1e10)}, "OtherSys": {nz, le(1e10)},
		"NextGC": {nz, le(1e10)}, "LastGC": {nz},
		"PauseTotalNs": {le(1e11)}, "PauseNs": nil, "PauseEnd": nil,
		"NumGC": {nz, le(1e9)}, "NumForcedGC": {nz, le(1e9)},
		"GCCPUFraction": {le(0.99)}, "EnableGC": {eq(true)}, "DebugGC": {eq(false)},
		"BySize": nil,
	}

	rst := reflect.ValueOf(st).Elem()
	for i := 0; i < rst.Type().NumField(); i++ {
		name, val := rst.Type().Field(i).Name, rst.Field(i).Interface()
		checks, ok := fields[name]
		if !ok {
			t.Errorf("unknown MemStats field %s", name)
			continue
		}
		for _, check := range checks {
			if err := check(val); err != nil {
				t.Errorf("%s = %v: %s", name, val, err)
			}
		}
	}

	if st.Sys != st.HeapSys+st.StackSys+st.MSpanSys+st.MCacheSys+
		st.BuckHashSys+st.GCSys+st.OtherSys {
		t.Fatalf("Bad sys value: %+v", *st)
	}

	if st.HeapIdle+st.HeapInuse != st.HeapSys {
		t.Fatalf("HeapIdle(%d) + HeapInuse(%d) should be equal to HeapSys(%d), but isn't.", st.HeapIdle, st.HeapInuse, st.HeapSys)
	}

	if lpe := st.PauseEnd[int(st.NumGC+255)%len(st.PauseEnd)]; st.LastGC != lpe {
		t.Fatalf("LastGC(%d) != last PauseEnd(%d)", st.LastGC, lpe)
	}

	var pauseTotal uint64
	for _, pause := range st.PauseNs {
		pauseTotal += pause
	}
	if int(st.NumGC) < len(st.PauseNs) {
		// We have all pauses, so this should be exact.
		if st.PauseTotalNs != pauseTotal {
			t.Fatalf("PauseTotalNs(%d) != sum PauseNs(%d)", st.PauseTotalNs, pauseTotal)
		}
		for i := int(st.NumGC); i < len(st.PauseNs); i++ {
			if st.PauseNs[i] != 0 {
				t.Fatalf("Non-zero PauseNs[%d]: %+v", i, st)
			}
			if st.PauseEnd[i] != 0 {
				t.Fatalf("Non-zero PauseEnd[%d]: %+v", i, st)
			}
		}
	} else {
		if st.PauseTotalNs < pauseTotal {
			t.Fatalf("PauseTotalNs(%d) < sum PauseNs(%d)", st.PauseTotalNs, pauseTotal)
		}
	}

	if st.NumForcedGC > st.NumGC {
		t.Fatalf("NumForcedGC(%d) > NumGC(%d)", st.NumForcedGC, st.NumGC)
	}
}

func TestStringConcatenationAllocs(t *testing.T) {
	n := testing.AllocsPerRun(1e3, func() {
		b := make([]byte, 10)
		for i := 0; i < 10; i++ {
			b[i] = byte(i) + '0'
		}
		s := "foo" + string(b)
		if want := "foo0123456789"; s != want {
			t.Fatalf("want %v, got %v", want, s)
		}
	})
	// Only string concatenation allocates.
	if n != 1 {
		t.Fatalf("want 1 allocation, got %v", n)
	}
}

func TestTinyAlloc(t *testing.T) {
	if runtime.Raceenabled {
		t.Skip("tinyalloc suppressed when running in race mode")
	}
	const N = 16
	var v [N]unsafe.Pointer
	for i := range v {
		v[i] = unsafe.Pointer(new(byte))
	}

	chunks := make(map[uintptr]bool, N)
	for _, p := range v {
		chunks[uintptr(p)&^7] = true
	}

	if len(chunks) == N {
		t.Fatal("no bytes allocated within the same 8-byte chunk")
	}
}

var (
	tinyByteSink   *byte
	tinyUint32Sink *uint32
	tinyObj12Sink  *obj12
)

type obj12 struct {
	a uint64
	b uint32
}

func TestTinyAllocIssue37262(t *testing.T) {
	if runtime.Raceenabled {
		t.Skip("tinyalloc suppressed when running in race mode")
	}
	// Try to cause an alignment access fault
	// by atomically accessing the first 64-bit
	// value of a tiny-allocated object.
	// See issue 37262 for details.

	// GC twice, once to reach a stable heap state
	// and again to make sure we finish the sweep phase.
	runtime.GC()
	runtime.GC()

	// Disable preemption so we stay on one P's tiny allocator and
	// nothing else allocates from it.
	runtime.Acquirem()

	// Make 1-byte allocations until we get a fresh tiny slot.
	aligned := false
	for i := 0; i < 16; i++ {
		tinyByteSink = new(byte)
		if uintptr(unsafe.Pointer(tinyByteSink))&0xf == 0xf {
			aligned = true
			break
		}
	}
	if !aligned {
		runtime.Releasem()
		t.Fatal("unable to get a fresh tiny slot")
	}

	// Create a 4-byte object so that the current
	// tiny slot is partially filled.
	tinyUint32Sink = new(uint32)

	// Create a 12-byte object, which fits into the
	// tiny slot. If it actually gets place there,
	// then the field "a" will be improperly aligned
	// for atomic access on 32-bit architectures.
	// This won't be true if issue 36606 gets resolved.
	tinyObj12Sink = new(obj12)

	// Try to atomically access "x.a".
	atomic.StoreUint64(&tinyObj12Sink.a, 10)

	// Clear the sinks.
	tinyByteSink = nil
	tinyUint32Sink = nil
	tinyObj12Sink = nil

	runtime.Releasem()
}

func TestPageCacheLeak(t *testing.T) {
	defer GOMAXPROCS(GOMAXPROCS(1))
	leaked := PageCachePagesLeaked()
	if leaked != 0 {
		t.Fatalf("found %d leaked pages in page caches", leaked)
	}
}

func TestPhysicalMemoryUtilization(t *testing.T) {
	got := runTestProg(t, "testprog", "GCPhys")
	want := "OK\n"
	if got != want {
		t.Fatalf("expected %q, but got %q", want, got)
	}
}

func TestScavengedBitsCleared(t *testing.T) {
	var mismatches [128]BitsMismatch
	if n, ok := CheckScavengedBitsCleared(mismatches[:]); !ok {
		t.Errorf("uncleared scavenged bits")
		for _, m := range mismatches[:n] {
			t.Logf("\t@ address 0x%x", m.Base)
			t.Logf("\t|  got: %064b", m.Got)
			t.Logf("\t| want: %064b", m.Want)
		}
		t.FailNow()
	}
}

type acLink struct {
	x [1 << 20]byte
}

var arenaCollisionSink []*acLink

func TestArenaCollision(t *testing.T) {
	testenv.MustHaveExec(t)

	// Test that mheap.sysAlloc handles collisions with other
	// memory mappings.
	if os.Getenv("TEST_ARENA_COLLISION") != "1" {
		cmd := testenv.CleanCmdEnv(exec.Command(os.Args[0], "-test.run=TestArenaCollision", "-test.v"))
		cmd.Env = append(cmd.Env, "TEST_ARENA_COLLISION=1")
		out, err := cmd.CombinedOutput()
		if race.Enabled {
			// This test runs the runtime out of hint
			// addresses, so it will start mapping the
			// heap wherever it can. The race detector
			// doesn't support this, so look for the
			// expected failure.
			if want := "too many address space collisions"; !strings.Contains(string(out), want) {
				t.Fatalf("want %q, got:\n%s", want, string(out))
			}
		} else if !strings.Contains(string(out), "PASS\n") || err != nil {
			t.Fatalf("%s\n(exit status %v)", string(out), err)
		}
		return
	}
	disallowed := [][2]uintptr{}
	// Drop all but the next 3 hints. 64-bit has a lot of hints,
	// so it would take a lot of memory to go through all of them.
	KeepNArenaHints(3)
	// Consume these 3 hints and force the runtime to find some
	// fallback hints.
	for i := 0; i < 5; i++ {
		// Reserve memory at the next hint so it can't be used
		// for the heap.
		start, end := MapNextArenaHint()
		disallowed = append(disallowed, [2]uintptr{start, end})
		// Allocate until the runtime tries to use the hint we
		// just mapped over.
		hint := GetNextArenaHint()
		for GetNextArenaHint() == hint {
			ac := new(acLink)
			arenaCollisionSink = append(arenaCollisionSink, ac)
			// The allocation must not have fallen into
			// one of the reserved regions.
			p := uintptr(unsafe.Pointer(ac))
			for _, d := range disallowed {
				if d[0] <= p && p < d[1] {
					t.Fatalf("allocation %#x in reserved region [%#x, %#x)", p, d[0], d[1])
				}
			}
		}
	}
}

var mallocSink uintptr

func BenchmarkMalloc8(b *testing.B) {
	var x uintptr
	for i := 0; i < b.N; i++ {
		p := new(int64)
		x ^= uintptr(unsafe.Pointer(p))
	}
	mallocSink = x
}

func BenchmarkMalloc16(b *testing.B) {
	var x uintptr
	for i := 0; i < b.N; i++ {
		p := new([2]int64)
		x ^= uintptr(unsafe.Pointer(p))
	}
	mallocSink = x
}

func BenchmarkMallocTypeInfo8(b *testing.B) {
	var x uintptr
	for i := 0; i < b.N; i++ {
		p := new(struct {
			p [8 / unsafe.Sizeof(uintptr(0))]*int
		})
		x ^= uintptr(unsafe.Pointer(p))
	}
	mallocSink = x
}

func BenchmarkMallocTypeInfo16(b *testing.B) {
	var x uintptr
	for i := 0; i < b.N; i++ {
		p := new(struct {
			p [16 / unsafe.Sizeof(uintptr(0))]*int
		})
		x ^= uintptr(unsafe.Pointer(p))
	}
	mallocSink = x
}

type LargeStruct struct {
	x [16][]byte
}

func BenchmarkMallocLargeStruct(b *testing.B) {
	var x uintptr
	for i := 0; i < b.N; i++ {
		p := make([]LargeStruct, 2)
		x ^= uintptr(unsafe.Pointer(&p[0]))
	}
	mallocSink = x
}

var n = flag.Int("n", 1000, "number of goroutines")

func BenchmarkGoroutineSelect(b *testing.B) {
	quit := make(chan struct{})
	read := func(ch chan struct{}) {
		for {
			select {
			case _, ok := <-ch:
				if !ok {
					return
				}
			case <-quit:
				return
			}
		}
	}
	benchHelper(b, *n, read)
}

func BenchmarkGoroutineBlocking(b *testing.B) {
	read := func(ch chan struct{}) {
		for {
			if _, ok := <-ch; !ok {
				return
			}
		}
	}
	benchHelper(b, *n, read)
}

func BenchmarkGoroutineForRange(b *testing.B) {
	read := func(ch chan struct{}) {
		for range ch {
		}
	}
	benchHelper(b, *n, read)
}

func benchHelper(b *testing.B, n int, read func(chan struct{})) {
	m := make([]chan struct{}, n)
	for i := range m {
		m[i] = make(chan struct{}, 1)
		go read(m[i])
	}
	b.StopTimer()
	b.ResetTimer()
	GC()

	for i := 0; i < b.N; i++ {
		for _, ch := range m {
			if ch != nil {
				ch <- struct{}{}
			}
		}
		time.Sleep(10 * time.Millisecond)
		b.StartTimer()
		GC()
		b.StopTimer()
	}

	for _, ch := range m {
		close(ch)
	}
	time.Sleep(10 * time.Millisecond)
}

func BenchmarkGoroutineIdle(b *testing.B) {
	quit := make(chan struct{})
	fn := func() {
		<-quit
	}
	for i := 0; i < *n; i++ {
		go fn()
	}

	GC()
	b.ResetTimer()

	for i := 0; i < b.N; i++ {
		GC()
	}

	b.StopTimer()
	close(quit)
	time.Sleep(10 * time.Millisecond)
}
