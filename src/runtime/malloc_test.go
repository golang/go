// Copyright 2013 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

package runtime_test

import (
	"flag"
	"fmt"
	"internal/asan"
	"internal/race"
	"internal/testenv"
	"os"
	"os/exec"
	"reflect"
	"runtime"
	. "runtime"
	"strings"
	"sync"
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
	if asan.Enabled {
		t.Skip("tinyalloc suppressed when running in asan mode due to redzone")
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

type obj12 struct {
	a uint64
	b uint32
}

func TestTinyAllocIssue37262(t *testing.T) {
	if runtime.Raceenabled {
		t.Skip("tinyalloc suppressed when running in race mode")
	}
	if asan.Enabled {
		t.Skip("tinyalloc suppressed when running in asan mode due to redzone")
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
		x := runtime.Escape(new(byte))
		if uintptr(unsafe.Pointer(x))&0xf == 0xf {
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
	runtime.Escape(new(uint32))

	// Create a 12-byte object, which fits into the
	// tiny slot. If it actually gets place there,
	// then the field "a" will be improperly aligned
	// for atomic access on 32-bit architectures.
	// This won't be true if issue 36606 gets resolved.
	tinyObj12 := runtime.Escape(new(obj12))

	// Try to atomically access "x.a".
	atomic.StoreUint64(&tinyObj12.a, 10)

	runtime.Releasem()
}

// TestFreegc does basic testing of explicit frees.
func TestFreegc(t *testing.T) {
	tests := []struct {
		size   string
		f      func(noscan bool) func(*testing.T)
		noscan bool
	}{
		// Types without pointers.
		{"size=16", testFreegc[[16]byte], true}, // smallest we support currently
		{"size=17", testFreegc[[17]byte], true},
		{"size=64", testFreegc[[64]byte], true},
		{"size=500", testFreegc[[500]byte], true},
		{"size=512", testFreegc[[512]byte], true},
		{"size=4096", testFreegc[[4096]byte], true},
		{"size=20000", testFreegc[[20000]byte], true},       // not power of 2 or spc boundary
		{"size=32KiB-8", testFreegc[[1<<15 - 8]byte], true}, // max noscan small object for 64-bit
	}

	// Run the tests twice if not in -short mode or not otherwise saving test time.
	// First while manually calling runtime.GC to slightly increase isolation (perhaps making
	// problems more reproducible).
	for _, tt := range tests {
		runtime.GC()
		t.Run(fmt.Sprintf("gc=yes/ptrs=%v/%s", !tt.noscan, tt.size), tt.f(tt.noscan))
	}
	runtime.GC()

	if testing.Short() || !RuntimeFreegcEnabled || runtime.Raceenabled {
		return
	}

	// Again, but without manually calling runtime.GC in the loop (perhaps less isolation might
	// trigger problems).
	for _, tt := range tests {
		t.Run(fmt.Sprintf("gc=no/ptrs=%v/%s", !tt.noscan, tt.size), tt.f(tt.noscan))
	}
	runtime.GC()
}

func testFreegc[T comparable](noscan bool) func(*testing.T) {
	// We use stressMultiple to influence the duration of the tests.
	// When testing freegc changes, stressMultiple can be increased locally
	// to test longer or in some cases with more goroutines.
	// It can also be helpful to test with GODEBUG=clobberfree=1 and
	// with and without doubleCheckMalloc and doubleCheckReusable enabled.
	stressMultiple := 10
	if testing.Short() || !RuntimeFreegcEnabled || runtime.Raceenabled {
		stressMultiple = 1
	}

	return func(t *testing.T) {
		alloc := func() *T {
			// Force heap alloc, plus some light validation of zeroed memory.
			t.Helper()
			p := Escape(new(T))
			var zero T
			if *p != zero {
				t.Fatalf("allocator returned non-zero memory: %v", *p)
			}
			return p
		}

		free := func(p *T) {
			t.Helper()
			var zero T
			if *p != zero {
				t.Fatalf("found non-zero memory before freegc (tests do not modify memory): %v", *p)
			}
			runtime.Freegc(unsafe.Pointer(p), unsafe.Sizeof(*p), noscan)
		}

		t.Run("basic-free", func(t *testing.T) {
			// Test that freeing a live heap object doesn't crash.
			for range 100 {
				p := alloc()
				free(p)
			}
		})

		t.Run("stack-free", func(t *testing.T) {
			// Test that freeing a stack object doesn't crash.
			for range 100 {
				var x [32]byte
				var y [32]*int
				runtime.Freegc(unsafe.Pointer(&x), unsafe.Sizeof(x), true)  // noscan
				runtime.Freegc(unsafe.Pointer(&y), unsafe.Sizeof(y), false) // !noscan
			}
		})

		// Check our allocations. These tests rely on the
		// current implementation treating a re-used object
		// as not adding to the allocation counts seen
		// by testing.AllocsPerRun. (This is not the desired
		// long-term behavior, but it is the current behavior and
		// makes these tests convenient).

		t.Run("allocs-baseline", func(t *testing.T) {
			// Baseline result without any explicit free.
			allocs := testing.AllocsPerRun(100, func() {
				for range 100 {
					p := alloc()
					_ = p
				}
			})
			if allocs < 100 {
				// TODO(thepudds): we get exactly 100 for almost all the tests, but investigate why
				// ~101 allocs for TestFreegc/ptrs=true/size=32KiB-8.
				t.Fatalf("expected >=100 allocations, got %v", allocs)
			}
		})

		t.Run("allocs-with-free", func(t *testing.T) {
			// Same allocations, but now using explicit free so that
			// no allocs get reported. (Again, not the desired long-term behavior).
			if SizeSpecializedMallocEnabled {
				t.Skip("temporarily skipping alloc tests for GOEXPERIMENT=sizespecializedmalloc")
			}
			if !RuntimeFreegcEnabled {
				t.Skip("skipping alloc tests with runtime.freegc disabled")
			}
			allocs := testing.AllocsPerRun(100, func() {
				for range 100 {
					p := alloc()
					free(p)
				}
			})
			if allocs != 0 {
				t.Fatalf("expected 0 allocations, got %v", allocs)
			}
		})

		t.Run("free-multiple", func(t *testing.T) {
			// Multiple allocations outstanding before explicitly freeing,
			// but still within the limit of our smallest free list size
			// so that no allocs are reported. (Again, not long-term behavior).
			if SizeSpecializedMallocEnabled {
				t.Skip("temporarily skipping alloc tests for GOEXPERIMENT=sizespecializedmalloc")
			}
			if !RuntimeFreegcEnabled {
				t.Skip("skipping alloc tests with runtime.freegc disabled")
			}
			const maxOutstanding = 20
			s := make([]*T, 0, maxOutstanding)
			allocs := testing.AllocsPerRun(100*stressMultiple, func() {
				s = s[:0]
				for range maxOutstanding {
					p := alloc()
					s = append(s, p)
				}
				for _, p := range s {
					free(p)
				}
			})
			if allocs != 0 {
				t.Fatalf("expected 0 allocations, got %v", allocs)
			}
		})

		if runtime.GOARCH == "wasm" {
			// TODO(thepudds): for wasm, double-check if just slow, vs. some test logic problem,
			// vs. something else. It might have been wasm was slowest with tests that spawn
			// many goroutines, which might be expected for wasm. This skip might no longer be
			// needed now that we have tuned test execution time more, or perhaps wasm should just
			// always run in short mode, which might also let us remove this skip.
			t.Skip("skipping remaining freegc tests, was timing out on wasm")
		}

		t.Run("free-many", func(t *testing.T) {
			// Confirm we are graceful if we have more freed elements at once
			// than the max free list size.
			s := make([]*T, 0, 1000)
			iterations := stressMultiple * stressMultiple // currently 1 (-short) or 100
			for range iterations {
				s = s[:0]
				for range 1000 {
					p := alloc()
					s = append(s, p)
				}
				for _, p := range s {
					free(p)
				}
			}
		})

		t.Run("duplicate-check", func(t *testing.T) {
			// A simple duplicate allocation test. We track what should be the set
			// of live pointers in a map across a series of allocs and frees,
			// and fail if a live pointer value is returned by an allocation.
			// TODO: maybe add randomness? allow more live pointers? do across goroutines?
			live := make(map[uintptr]bool)
			for i := range 100 * stressMultiple {
				var s []*T
				// Alloc 10 times, tracking the live pointer values.
				for j := range 10 {
					p := alloc()
					uptr := uintptr(unsafe.Pointer(p))
					if live[uptr] {
						t.Fatalf("found duplicate pointer (0x%x). i: %d j: %d", uptr, i, j)
					}
					live[uptr] = true
					s = append(s, p)
				}
				// Explicitly free those pointers, removing them from the live map.
				for k := range s {
					p := s[k]
					s[k] = nil
					uptr := uintptr(unsafe.Pointer(p))
					free(p)
					delete(live, uptr)
				}
			}
		})

		t.Run("free-other-goroutine", func(t *testing.T) {
			// Use explicit free, but the free happens on a different goroutine than the alloc.
			// This also lightly simulates how the free code sees P migration or flushing
			// the mcache, assuming we have > 1 P. (Not using testing.AllocsPerRun here).
			iterations := 10 * stressMultiple * stressMultiple // currently 10 (-short) or 1000
			for _, capacity := range []int{2} {
				for range iterations {
					ch := make(chan *T, capacity)
					var wg sync.WaitGroup
					for range 2 {
						wg.Add(1)
						go func() {
							defer wg.Done()
							for p := range ch {
								free(p)
							}
						}()
					}
					for range 100 {
						p := alloc()
						ch <- p
					}
					close(ch)
					wg.Wait()
				}
			}
		})

		t.Run("many-goroutines", func(t *testing.T) {
			// Allocate across multiple goroutines, freeing on the same goroutine.
			// TODO: probably remove the duplicate checking here; not that useful.
			counts := []int{1, 2, 4, 8, 10 * stressMultiple}
			for _, goroutines := range counts {
				var wg sync.WaitGroup
				for range goroutines {
					wg.Add(1)
					go func() {
						defer wg.Done()
						live := make(map[uintptr]bool)
						for range 100 * stressMultiple {
							p := alloc()
							uptr := uintptr(unsafe.Pointer(p))
							if live[uptr] {
								panic("TestFreeLive: found duplicate pointer")
							}
							live[uptr] = true
							free(p)
							delete(live, uptr)
						}
					}()
				}
				wg.Wait()
			}
		})

		t.Run("assist-credit", func(t *testing.T) {
			// Allocate and free using the same span class repeatedly while
			// verifying it results in a net zero change in assist credit.
			// This helps double-check our manipulation of the assist credit
			// during mallocgc/freegc, including in cases when there is
			// internal fragmentation when the requested mallocgc size is
			// smaller than the size class.
			//
			// See https://go.dev/cl/717520 for some additional discussion,
			// including how we can deliberately cause the test to fail currently
			// if we purposefully introduce some assist credit bugs.
			if SizeSpecializedMallocEnabled {
				// TODO(thepudds): skip this test at this point in the stack; later CL has
				// integration with sizespecializedmalloc.
				t.Skip("temporarily skip assist credit test for GOEXPERIMENT=sizespecializedmalloc")
			}
			if !RuntimeFreegcEnabled {
				t.Skip("skipping assist credit test with runtime.freegc disabled")
			}

			// Use a background goroutine to continuously run the GC.
			done := make(chan struct{})
			defer close(done)
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

			// If making changes related to this test, consider testing locally with
			// larger counts, like 100K or 1M.
			counts := []int{1, 2, 10, 100 * stressMultiple}
			// Dropping down to GOMAXPROCS=1 might help reduce noise.
			defer GOMAXPROCS(GOMAXPROCS(1))
			size := int64(unsafe.Sizeof(*new(T)))
			for _, count := range counts {
				// Start by forcing a GC to reset this g's assist credit
				// and perhaps help us get a cleaner measurement of GC cycle count.
				runtime.GC()
				for i := range count {
					// We disable preemption to reduce other code's ability to adjust this g's
					// assist credit or otherwise change things while we are measuring.
					Acquirem()

					// We do two allocations per loop, with the second allocation being
					// the one we measure. The first allocation tries to ensure at least one
					// reusable object on the mspan's free list when we do our measured allocation.
					p := alloc()
					free(p)

					// Now do our primary allocation of interest, bracketed by measurements.
					// We measure more than we strictly need (to log details in case of a failure).
					creditStart := AssistCredit()
					blackenStart := GcBlackenEnable()
					p = alloc()
					blackenAfterAlloc := GcBlackenEnable()
					creditAfterAlloc := AssistCredit()
					free(p)
					blackenEnd := GcBlackenEnable()
					creditEnd := AssistCredit()

					Releasem()
					GoschedIfBusy()

					delta := creditEnd - creditStart
					if delta != 0 {
						t.Logf("assist credit non-zero delta: %d", delta)
						t.Logf("\t| size: %d i: %d count: %d", size, i, count)
						t.Logf("\t| credit before: %d credit after: %d", creditStart, creditEnd)
						t.Logf("\t| alloc delta: %d free delta: %d",
							creditAfterAlloc-creditStart, creditEnd-creditAfterAlloc)
						t.Logf("\t| gcBlackenEnable (start / after alloc / end): %v/%v/%v",
							blackenStart, blackenAfterAlloc, blackenEnd)
						t.FailNow()
					}
				}
			}
		})
	}
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
	// Test that mheap.sysAlloc handles collisions with other
	// memory mappings.
	if os.Getenv("TEST_ARENA_COLLISION") != "1" {
		cmd := testenv.CleanCmdEnv(exec.Command(testenv.Executable(t), "-test.run=^TestArenaCollision$", "-test.v"))
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
		start, end, ok := MapNextArenaHint()
		if !ok {
			t.Skipf("failed to reserve memory at next arena hint [%#x, %#x)", start, end)
		}
		t.Logf("reserved [%#x, %#x)", start, end)
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

func BenchmarkMalloc8(b *testing.B) {
	for i := 0; i < b.N; i++ {
		p := new(int64)
		Escape(p)
	}
}

func BenchmarkMalloc16(b *testing.B) {
	for i := 0; i < b.N; i++ {
		p := new([2]int64)
		Escape(p)
	}
}

func BenchmarkMalloc32(b *testing.B) {
	for i := 0; i < b.N; i++ {
		p := new([4]int64)
		Escape(p)
	}
}

func BenchmarkMallocTypeInfo8(b *testing.B) {
	for i := 0; i < b.N; i++ {
		p := new(struct {
			p [8 / unsafe.Sizeof(uintptr(0))]*int
		})
		Escape(p)
	}
}

func BenchmarkMallocTypeInfo16(b *testing.B) {
	for i := 0; i < b.N; i++ {
		p := new(struct {
			p [16 / unsafe.Sizeof(uintptr(0))]*int
		})
		Escape(p)
	}
}

func BenchmarkMallocTypeInfo32(b *testing.B) {
	for i := 0; i < b.N; i++ {
		p := new(struct {
			p [32 / unsafe.Sizeof(uintptr(0))]*int
		})
		Escape(p)
	}
}

type LargeStruct struct {
	x [16][]byte
}

func BenchmarkMallocLargeStruct(b *testing.B) {
	for i := 0; i < b.N; i++ {
		p := make([]LargeStruct, 2)
		Escape(p)
	}
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

func TestMkmalloc(t *testing.T) {
	testenv.MustHaveGoRun(t)
	testenv.MustHaveExternalNetwork(t) // To download the golang.org/x/tools dependency.
	output, err := exec.Command("go", "-C", "_mkmalloc", "test").CombinedOutput()
	t.Logf("test output:\n%s", output)
	if err != nil {
		t.Errorf("_mkmalloc tests failed: %v", err)
	}
}
