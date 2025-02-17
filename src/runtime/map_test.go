// Copyright 2013 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

package runtime_test

import (
	"fmt"
	"internal/goexperiment"
	"internal/testenv"
	"math"
	"os"
	"reflect"
	"runtime"
	"slices"
	"strconv"
	"strings"
	"sync"
	"testing"
	"unsafe"
)

// negative zero is a good test because:
//  1. 0 and -0 are equal, yet have distinct representations.
//  2. 0 is represented as all zeros, -0 isn't.
//
// I'm not sure the language spec actually requires this behavior,
// but it's what the current map implementation does.
func TestNegativeZero(t *testing.T) {
	m := make(map[float64]bool, 0)

	m[+0.0] = true
	m[math.Copysign(0.0, -1.0)] = true // should overwrite +0 entry

	if len(m) != 1 {
		t.Error("length wrong")
	}

	for k := range m {
		if math.Copysign(1.0, k) > 0 {
			t.Error("wrong sign")
		}
	}

	m = make(map[float64]bool, 0)
	m[math.Copysign(0.0, -1.0)] = true
	m[+0.0] = true // should overwrite -0.0 entry

	if len(m) != 1 {
		t.Error("length wrong")
	}

	for k := range m {
		if math.Copysign(1.0, k) < 0 {
			t.Error("wrong sign")
		}
	}
}

func testMapNan(t *testing.T, m map[float64]int) {
	if len(m) != 3 {
		t.Error("length wrong")
	}
	s := 0
	for k, v := range m {
		if k == k {
			t.Error("nan disappeared")
		}
		if (v & (v - 1)) != 0 {
			t.Error("value wrong")
		}
		s |= v
	}
	if s != 7 {
		t.Error("values wrong")
	}
}

// nan is a good test because nan != nan, and nan has
// a randomized hash value.
func TestMapAssignmentNan(t *testing.T) {
	m := make(map[float64]int, 0)
	nan := math.NaN()

	// Test assignment.
	m[nan] = 1
	m[nan] = 2
	m[nan] = 4
	testMapNan(t, m)
}

// nan is a good test because nan != nan, and nan has
// a randomized hash value.
func TestMapOperatorAssignmentNan(t *testing.T) {
	m := make(map[float64]int, 0)
	nan := math.NaN()

	// Test assignment operations.
	m[nan] += 1
	m[nan] += 2
	m[nan] += 4
	testMapNan(t, m)
}

func TestMapOperatorAssignment(t *testing.T) {
	m := make(map[int]int, 0)

	// "m[k] op= x" is rewritten into "m[k] = m[k] op x"
	// differently when op is / or % than when it isn't.
	// Simple test to make sure they all work as expected.
	m[0] = 12345
	m[0] += 67890
	m[0] /= 123
	m[0] %= 456

	const want = (12345 + 67890) / 123 % 456
	if got := m[0]; got != want {
		t.Errorf("got %d, want %d", got, want)
	}
}

var sinkAppend bool

func TestMapAppendAssignment(t *testing.T) {
	m := make(map[int][]int, 0)

	m[0] = nil
	m[0] = append(m[0], 12345)
	m[0] = append(m[0], 67890)
	sinkAppend, m[0] = !sinkAppend, append(m[0], 123, 456)
	a := []int{7, 8, 9, 0}
	m[0] = append(m[0], a...)

	want := []int{12345, 67890, 123, 456, 7, 8, 9, 0}
	if got := m[0]; !slices.Equal(got, want) {
		t.Errorf("got %v, want %v", got, want)
	}
}

// Maps aren't actually copied on assignment.
func TestAlias(t *testing.T) {
	m := make(map[int]int, 0)
	m[0] = 5
	n := m
	n[0] = 6
	if m[0] != 6 {
		t.Error("alias didn't work")
	}
}

func TestGrowWithNaN(t *testing.T) {
	m := make(map[float64]int, 4)
	nan := math.NaN()

	// Use both assignment and assignment operations as they may
	// behave differently.
	m[nan] = 1
	m[nan] = 2
	m[nan] += 4

	cnt := 0
	s := 0
	growflag := true
	for k, v := range m {
		if growflag {
			// force a hashtable resize
			for i := 0; i < 50; i++ {
				m[float64(i)] = i
			}
			for i := 50; i < 100; i++ {
				m[float64(i)] += i
			}
			growflag = false
		}
		if k != k {
			cnt++
			s |= v
		}
	}
	if cnt != 3 {
		t.Error("NaN keys lost during grow")
	}
	if s != 7 {
		t.Error("NaN values lost during grow")
	}
}

type FloatInt struct {
	x float64
	y int
}

func TestGrowWithNegativeZero(t *testing.T) {
	negzero := math.Copysign(0.0, -1.0)
	m := make(map[FloatInt]int, 4)
	m[FloatInt{0.0, 0}] = 1
	m[FloatInt{0.0, 1}] += 2
	m[FloatInt{0.0, 2}] += 4
	m[FloatInt{0.0, 3}] = 8
	growflag := true
	s := 0
	cnt := 0
	negcnt := 0
	// The first iteration should return the +0 key.
	// The subsequent iterations should return the -0 key.
	// I'm not really sure this is required by the spec,
	// but it makes sense.
	// TODO: are we allowed to get the first entry returned again???
	for k, v := range m {
		if v == 0 {
			continue
		} // ignore entries added to grow table
		cnt++
		if math.Copysign(1.0, k.x) < 0 {
			if v&16 == 0 {
				t.Error("key/value not updated together 1")
			}
			negcnt++
			s |= v & 15
		} else {
			if v&16 == 16 {
				t.Error("key/value not updated together 2", k, v)
			}
			s |= v
		}
		if growflag {
			// force a hashtable resize
			for i := 0; i < 100; i++ {
				m[FloatInt{3.0, i}] = 0
			}
			// then change all the entries
			// to negative zero
			m[FloatInt{negzero, 0}] = 1 | 16
			m[FloatInt{negzero, 1}] = 2 | 16
			m[FloatInt{negzero, 2}] = 4 | 16
			m[FloatInt{negzero, 3}] = 8 | 16
			growflag = false
		}
	}
	if s != 15 {
		t.Error("entry missing", s)
	}
	if cnt != 4 {
		t.Error("wrong number of entries returned by iterator", cnt)
	}
	if negcnt != 3 {
		t.Error("update to negzero missed by iteration", negcnt)
	}
}

func TestIterGrowAndDelete(t *testing.T) {
	m := make(map[int]int, 4)
	for i := 0; i < 100; i++ {
		m[i] = i
	}
	growflag := true
	for k := range m {
		if growflag {
			// grow the table
			for i := 100; i < 1000; i++ {
				m[i] = i
			}
			// delete all odd keys
			for i := 1; i < 1000; i += 2 {
				delete(m, i)
			}
			growflag = false
		} else {
			if k&1 == 1 {
				t.Error("odd value returned")
			}
		}
	}
}

// make sure old bucket arrays don't get GCd while
// an iterator is still using them.
func TestIterGrowWithGC(t *testing.T) {
	m := make(map[int]int, 4)
	for i := 0; i < 8; i++ {
		m[i] = i
	}
	for i := 8; i < 16; i++ {
		m[i] += i
	}
	growflag := true
	bitmask := 0
	for k := range m {
		if k < 16 {
			bitmask |= 1 << uint(k)
		}
		if growflag {
			// grow the table
			for i := 100; i < 1000; i++ {
				m[i] = i
			}
			// trigger a gc
			runtime.GC()
			growflag = false
		}
	}
	if bitmask != 1<<16-1 {
		t.Error("missing key", bitmask)
	}
}

func testConcurrentReadsAfterGrowth(t *testing.T, useReflect bool) {
	t.Parallel()
	if runtime.GOMAXPROCS(-1) == 1 {
		defer runtime.GOMAXPROCS(runtime.GOMAXPROCS(16))
	}
	numLoop := 10
	numGrowStep := 250
	numReader := 16
	if testing.Short() {
		numLoop, numGrowStep = 2, 100
	}
	for i := 0; i < numLoop; i++ {
		m := make(map[int]int, 0)
		for gs := 0; gs < numGrowStep; gs++ {
			m[gs] = gs
			var wg sync.WaitGroup
			wg.Add(numReader * 2)
			for nr := 0; nr < numReader; nr++ {
				go func() {
					defer wg.Done()
					for range m {
					}
				}()
				go func() {
					defer wg.Done()
					for key := 0; key < gs; key++ {
						_ = m[key]
					}
				}()
				if useReflect {
					wg.Add(1)
					go func() {
						defer wg.Done()
						mv := reflect.ValueOf(m)
						keys := mv.MapKeys()
						for _, k := range keys {
							mv.MapIndex(k)
						}
					}()
				}
			}
			wg.Wait()
		}
	}
}

func TestConcurrentReadsAfterGrowth(t *testing.T) {
	testConcurrentReadsAfterGrowth(t, false)
}

func TestConcurrentReadsAfterGrowthReflect(t *testing.T) {
	testConcurrentReadsAfterGrowth(t, true)
}

func TestBigItems(t *testing.T) {
	var key [256]string
	for i := 0; i < 256; i++ {
		key[i] = "foo"
	}
	m := make(map[[256]string][256]string, 4)
	for i := 0; i < 100; i++ {
		key[37] = fmt.Sprintf("string%02d", i)
		m[key] = key
	}
	var keys [100]string
	var values [100]string
	i := 0
	for k, v := range m {
		keys[i] = k[37]
		values[i] = v[37]
		i++
	}
	slices.Sort(keys[:])
	slices.Sort(values[:])
	for i := 0; i < 100; i++ {
		if keys[i] != fmt.Sprintf("string%02d", i) {
			t.Errorf("#%d: missing key: %v", i, keys[i])
		}
		if values[i] != fmt.Sprintf("string%02d", i) {
			t.Errorf("#%d: missing value: %v", i, values[i])
		}
	}
}

func TestMapHugeZero(t *testing.T) {
	type T [4000]byte
	m := map[int]T{}
	x := m[0]
	if x != (T{}) {
		t.Errorf("map value not zero")
	}
	y, ok := m[0]
	if ok {
		t.Errorf("map value should be missing")
	}
	if y != (T{}) {
		t.Errorf("map value not zero")
	}
}

type empty struct {
}

func TestEmptyKeyAndValue(t *testing.T) {
	a := make(map[int]empty, 4)
	b := make(map[empty]int, 4)
	c := make(map[empty]empty, 4)
	a[0] = empty{}
	b[empty{}] = 0
	b[empty{}] = 1
	c[empty{}] = empty{}

	if len(a) != 1 {
		t.Errorf("empty value insert problem")
	}
	if len(b) != 1 {
		t.Errorf("empty key insert problem")
	}
	if len(c) != 1 {
		t.Errorf("empty key+value insert problem")
	}
	if b[empty{}] != 1 {
		t.Errorf("empty key returned wrong value")
	}
}

// Tests a map with a single bucket, with same-lengthed short keys
// ("quick keys") as well as long keys.
func TestSingleBucketMapStringKeys_DupLen(t *testing.T) {
	testMapLookups(t, map[string]string{
		"x":                      "x1val",
		"xx":                     "x2val",
		"foo":                    "fooval",
		"bar":                    "barval", // same key length as "foo"
		"xxxx":                   "x4val",
		strings.Repeat("x", 128): "longval1",
		strings.Repeat("y", 128): "longval2",
	})
}

// Tests a map with a single bucket, with all keys having different lengths.
func TestSingleBucketMapStringKeys_NoDupLen(t *testing.T) {
	testMapLookups(t, map[string]string{
		"x":                      "x1val",
		"xx":                     "x2val",
		"foo":                    "fooval",
		"xxxx":                   "x4val",
		"xxxxx":                  "x5val",
		"xxxxxx":                 "x6val",
		strings.Repeat("x", 128): "longval",
	})
}

func testMapLookups(t *testing.T, m map[string]string) {
	for k, v := range m {
		if m[k] != v {
			t.Fatalf("m[%q] = %q; want %q", k, m[k], v)
		}
	}
}

// Tests whether the iterator returns the right elements when
// started in the middle of a grow, when the keys are NaNs.
func TestMapNanGrowIterator(t *testing.T) {
	m := make(map[float64]int)
	nan := math.NaN()
	const nBuckets = 16
	// To fill nBuckets buckets takes LOAD * nBuckets keys.
	nKeys := int(nBuckets * runtime.HashLoad)

	// Get map to full point with nan keys.
	for i := 0; i < nKeys; i++ {
		m[nan] = i
	}
	// Trigger grow
	m[1.0] = 1
	delete(m, 1.0)

	// Run iterator
	found := make(map[int]struct{})
	for _, v := range m {
		if v != -1 {
			if _, repeat := found[v]; repeat {
				t.Fatalf("repeat of value %d", v)
			}
			found[v] = struct{}{}
		}
		if len(found) == nKeys/2 {
			// Halfway through iteration, finish grow.
			for i := 0; i < nBuckets; i++ {
				delete(m, 1.0)
			}
		}
	}
	if len(found) != nKeys {
		t.Fatalf("missing value")
	}
}

// Issue 8410
func TestMapSparseIterOrder(t *testing.T) {
	// Run several rounds to increase the probability
	// of failure. One is not enough.
NextRound:
	for round := 0; round < 10; round++ {
		m := make(map[int]bool)
		// Add 1000 items, remove 980.
		for i := 0; i < 1000; i++ {
			m[i] = true
		}
		for i := 20; i < 1000; i++ {
			delete(m, i)
		}

		var first []int
		for i := range m {
			first = append(first, i)
		}

		// 800 chances to get a different iteration order.
		// See bug 8736 for why we need so many tries.
		for n := 0; n < 800; n++ {
			idx := 0
			for i := range m {
				if i != first[idx] {
					// iteration order changed.
					continue NextRound
				}
				idx++
			}
		}
		t.Fatalf("constant iteration order on round %d: %v", round, first)
	}
}

// Map iteration must not return duplicate entries.
func TestMapIterDuplicate(t *testing.T) {
	// Run several rounds to increase the probability
	// of failure. One is not enough.
	for range 1000 {
		m := make(map[int]bool)
		// Add 1000 items, remove 980.
		for i := 0; i < 1000; i++ {
			m[i] = true
		}
		for i := 20; i < 1000; i++ {
			delete(m, i)
		}

		var want []int
		for i := 0; i < 20; i++ {
			want = append(want, i)
		}

		var got []int
		for i := range m {
			got = append(got, i)
		}

		slices.Sort(got)

		if !reflect.DeepEqual(got, want) {
			t.Errorf("iteration got %v want %v\n", got, want)
		}
	}
}

func TestMapStringBytesLookup(t *testing.T) {
	// Use large string keys to avoid small-allocation coalescing,
	// which can cause AllocsPerRun to report lower counts than it should.
	m := map[string]int{
		"1000000000000000000000000000000000000000000000000": 1,
		"2000000000000000000000000000000000000000000000000": 2,
	}
	buf := []byte("1000000000000000000000000000000000000000000000000")
	if x := m[string(buf)]; x != 1 {
		t.Errorf(`m[string([]byte("1"))] = %d, want 1`, x)
	}
	buf[0] = '2'
	if x := m[string(buf)]; x != 2 {
		t.Errorf(`m[string([]byte("2"))] = %d, want 2`, x)
	}

	var x int
	n := testing.AllocsPerRun(100, func() {
		x += m[string(buf)]
	})
	if n != 0 {
		t.Errorf("AllocsPerRun for m[string(buf)] = %v, want 0", n)
	}

	x = 0
	n = testing.AllocsPerRun(100, func() {
		y, ok := m[string(buf)]
		if !ok {
			panic("!ok")
		}
		x += y
	})
	if n != 0 {
		t.Errorf("AllocsPerRun for x,ok = m[string(buf)] = %v, want 0", n)
	}
}

func TestMapLargeKeyNoPointer(t *testing.T) {
	const (
		I = 1000
		N = 64
	)
	type T [N]int
	m := make(map[T]int)
	for i := 0; i < I; i++ {
		var v T
		for j := 0; j < N; j++ {
			v[j] = i + j
		}
		m[v] = i
	}
	runtime.GC()
	for i := 0; i < I; i++ {
		var v T
		for j := 0; j < N; j++ {
			v[j] = i + j
		}
		if m[v] != i {
			t.Fatalf("corrupted map: want %+v, got %+v", i, m[v])
		}
	}
}

func TestMapLargeValNoPointer(t *testing.T) {
	const (
		I = 1000
		N = 64
	)
	type T [N]int
	m := make(map[int]T)
	for i := 0; i < I; i++ {
		var v T
		for j := 0; j < N; j++ {
			v[j] = i + j
		}
		m[i] = v
	}
	runtime.GC()
	for i := 0; i < I; i++ {
		var v T
		for j := 0; j < N; j++ {
			v[j] = i + j
		}
		v1 := m[i]
		for j := 0; j < N; j++ {
			if v1[j] != v[j] {
				t.Fatalf("corrupted map: want %+v, got %+v", v, v1)
			}
		}
	}
}

// Test that making a map with a large or invalid hint
// doesn't panic. (Issue 19926).
func TestIgnoreBogusMapHint(t *testing.T) {
	for _, hint := range []int64{-1, 1 << 62} {
		_ = make(map[int]int, hint)
	}
}

var testNonEscapingMapVariable int = 8

func TestNonEscapingMap(t *testing.T) {
	n := testing.AllocsPerRun(1000, func() {
		m := map[int]int{}
		m[0] = 0
	})
	if n != 0 {
		t.Errorf("mapliteral: want 0 allocs, got %v", n)
	}
	n = testing.AllocsPerRun(1000, func() {
		m := make(map[int]int)
		m[0] = 0
	})
	if n != 0 {
		t.Errorf("no hint: want 0 allocs, got %v", n)
	}
	n = testing.AllocsPerRun(1000, func() {
		m := make(map[int]int, 8)
		m[0] = 0
	})
	if n != 0 {
		t.Errorf("with small hint: want 0 allocs, got %v", n)
	}
	n = testing.AllocsPerRun(1000, func() {
		m := make(map[int]int, testNonEscapingMapVariable)
		m[0] = 0
	})
	if n != 0 {
		t.Errorf("with variable hint: want 0 allocs, got %v", n)
	}

}

func TestDeferDeleteSlow(t *testing.T) {
	ks := []complex128{0, 1, 2, 3}

	m := make(map[any]int)
	for i, k := range ks {
		m[k] = i
	}
	if len(m) != len(ks) {
		t.Errorf("want %d elements, got %d", len(ks), len(m))
	}

	func() {
		for _, k := range ks {
			defer delete(m, k)
		}
	}()
	if len(m) != 0 {
		t.Errorf("want 0 elements, got %d", len(m))
	}
}

// TestIncrementAfterDeleteValueInt and other test Issue 25936.
// Value types int, int32, int64 are affected. Value type string
// works as expected.
func TestIncrementAfterDeleteValueInt(t *testing.T) {
	const key1 = 12
	const key2 = 13

	m := make(map[int]int)
	m[key1] = 99
	delete(m, key1)
	m[key2]++
	if n2 := m[key2]; n2 != 1 {
		t.Errorf("incremented 0 to %d", n2)
	}
}

func TestIncrementAfterDeleteValueInt32(t *testing.T) {
	const key1 = 12
	const key2 = 13

	m := make(map[int]int32)
	m[key1] = 99
	delete(m, key1)
	m[key2]++
	if n2 := m[key2]; n2 != 1 {
		t.Errorf("incremented 0 to %d", n2)
	}
}

func TestIncrementAfterDeleteValueInt64(t *testing.T) {
	const key1 = 12
	const key2 = 13

	m := make(map[int]int64)
	m[key1] = 99
	delete(m, key1)
	m[key2]++
	if n2 := m[key2]; n2 != 1 {
		t.Errorf("incremented 0 to %d", n2)
	}
}

func TestIncrementAfterDeleteKeyStringValueInt(t *testing.T) {
	const key1 = ""
	const key2 = "x"

	m := make(map[string]int)
	m[key1] = 99
	delete(m, key1)
	m[key2] += 1
	if n2 := m[key2]; n2 != 1 {
		t.Errorf("incremented 0 to %d", n2)
	}
}

func TestIncrementAfterDeleteKeyValueString(t *testing.T) {
	const key1 = ""
	const key2 = "x"

	m := make(map[string]string)
	m[key1] = "99"
	delete(m, key1)
	m[key2] += "1"
	if n2 := m[key2]; n2 != "1" {
		t.Errorf("appended '1' to empty (nil) string, got %s", n2)
	}
}

// TestIncrementAfterBulkClearKeyStringValueInt tests that map bulk
// deletion (mapclear) still works as expected. Note that it was not
// affected by Issue 25936.
func TestIncrementAfterBulkClearKeyStringValueInt(t *testing.T) {
	const key1 = ""
	const key2 = "x"

	m := make(map[string]int)
	m[key1] = 99
	for k := range m {
		delete(m, k)
	}
	m[key2]++
	if n2 := m[key2]; n2 != 1 {
		t.Errorf("incremented 0 to %d", n2)
	}
}

func TestMapTombstones(t *testing.T) {
	m := map[int]int{}
	const N = 10000
	// Fill a map.
	for i := 0; i < N; i++ {
		m[i] = i
	}
	runtime.MapTombstoneCheck(m)
	// Delete half of the entries.
	for i := 0; i < N; i += 2 {
		delete(m, i)
	}
	runtime.MapTombstoneCheck(m)
	// Add new entries to fill in holes.
	for i := N; i < 3*N/2; i++ {
		m[i] = i
	}
	runtime.MapTombstoneCheck(m)
	// Delete everything.
	for i := 0; i < 3*N/2; i++ {
		delete(m, i)
	}
	runtime.MapTombstoneCheck(m)
}

type canString int

func (c canString) String() string {
	return fmt.Sprintf("%d", int(c))
}

func TestMapInterfaceKey(t *testing.T) {
	// Test all the special cases in runtime.typehash.
	type GrabBag struct {
		f32  float32
		f64  float64
		c64  complex64
		c128 complex128
		s    string
		i0   any
		i1   interface {
			String() string
		}
		a [4]string
	}

	m := map[any]bool{}
	// Put a bunch of data in m, so that a bad hash is likely to
	// lead to a bad bucket, which will lead to a missed lookup.
	for i := 0; i < 1000; i++ {
		m[i] = true
	}
	m[GrabBag{f32: 1.0}] = true
	if !m[GrabBag{f32: 1.0}] {
		panic("f32 not found")
	}
	m[GrabBag{f64: 1.0}] = true
	if !m[GrabBag{f64: 1.0}] {
		panic("f64 not found")
	}
	m[GrabBag{c64: 1.0i}] = true
	if !m[GrabBag{c64: 1.0i}] {
		panic("c64 not found")
	}
	m[GrabBag{c128: 1.0i}] = true
	if !m[GrabBag{c128: 1.0i}] {
		panic("c128 not found")
	}
	m[GrabBag{s: "foo"}] = true
	if !m[GrabBag{s: "foo"}] {
		panic("string not found")
	}
	m[GrabBag{i0: "foo"}] = true
	if !m[GrabBag{i0: "foo"}] {
		panic("interface{} not found")
	}
	m[GrabBag{i1: canString(5)}] = true
	if !m[GrabBag{i1: canString(5)}] {
		panic("interface{String() string} not found")
	}
	m[GrabBag{a: [4]string{"foo", "bar", "baz", "bop"}}] = true
	if !m[GrabBag{a: [4]string{"foo", "bar", "baz", "bop"}}] {
		panic("array not found")
	}
}

type panicStructKey struct {
	sli []int
}

func (p panicStructKey) String() string {
	return "panic"
}

type structKey struct {
}

func (structKey) String() string {
	return "structKey"
}

func TestEmptyMapWithInterfaceKey(t *testing.T) {
	var (
		b    bool
		i    int
		i8   int8
		i16  int16
		i32  int32
		i64  int64
		ui   uint
		ui8  uint8
		ui16 uint16
		ui32 uint32
		ui64 uint64
		uipt uintptr
		f32  float32
		f64  float64
		c64  complex64
		c128 complex128
		a    [4]string
		s    string
		p    *int
		up   unsafe.Pointer
		ch   chan int
		i0   any
		i1   interface {
			String() string
		}
		structKey structKey
		i0Panic   any = []int{}
		i1Panic   interface {
			String() string
		} = panicStructKey{}
		panicStructKey = panicStructKey{}
		sli            []int
		me             = map[any]struct{}{}
		mi             = map[interface {
			String() string
		}]struct{}{}
	)
	mustNotPanic := func(f func()) {
		f()
	}
	mustPanic := func(f func()) {
		defer func() {
			r := recover()
			if r == nil {
				t.Errorf("didn't panic")
			}
		}()
		f()
	}
	mustNotPanic(func() {
		_ = me[b]
	})
	mustNotPanic(func() {
		_ = me[i]
	})
	mustNotPanic(func() {
		_ = me[i8]
	})
	mustNotPanic(func() {
		_ = me[i16]
	})
	mustNotPanic(func() {
		_ = me[i32]
	})
	mustNotPanic(func() {
		_ = me[i64]
	})
	mustNotPanic(func() {
		_ = me[ui]
	})
	mustNotPanic(func() {
		_ = me[ui8]
	})
	mustNotPanic(func() {
		_ = me[ui16]
	})
	mustNotPanic(func() {
		_ = me[ui32]
	})
	mustNotPanic(func() {
		_ = me[ui64]
	})
	mustNotPanic(func() {
		_ = me[uipt]
	})
	mustNotPanic(func() {
		_ = me[f32]
	})
	mustNotPanic(func() {
		_ = me[f64]
	})
	mustNotPanic(func() {
		_ = me[c64]
	})
	mustNotPanic(func() {
		_ = me[c128]
	})
	mustNotPanic(func() {
		_ = me[a]
	})
	mustNotPanic(func() {
		_ = me[s]
	})
	mustNotPanic(func() {
		_ = me[p]
	})
	mustNotPanic(func() {
		_ = me[up]
	})
	mustNotPanic(func() {
		_ = me[ch]
	})
	mustNotPanic(func() {
		_ = me[i0]
	})
	mustNotPanic(func() {
		_ = me[i1]
	})
	mustNotPanic(func() {
		_ = me[structKey]
	})
	mustPanic(func() {
		_ = me[i0Panic]
	})
	mustPanic(func() {
		_ = me[i1Panic]
	})
	mustPanic(func() {
		_ = me[panicStructKey]
	})
	mustPanic(func() {
		_ = me[sli]
	})
	mustPanic(func() {
		_ = me[me]
	})

	mustNotPanic(func() {
		_ = mi[structKey]
	})
	mustPanic(func() {
		_ = mi[panicStructKey]
	})
}

func TestMapKeys(t *testing.T) {
	if goexperiment.SwissMap {
		t.Skip("mapkeys not implemented for swissmaps")
	}

	type key struct {
		s   string
		pad [128]byte // sizeof(key) > abi.MapMaxKeyBytes
	}
	m := map[key]int{{s: "a"}: 1, {s: "b"}: 2}
	keys := make([]key, 0, len(m))
	runtime.MapKeys(m, unsafe.Pointer(&keys))
	for _, k := range keys {
		if len(k.s) != 1 {
			t.Errorf("len(k.s) == %d, want 1", len(k.s))
		}
	}
}

func TestMapValues(t *testing.T) {
	if goexperiment.SwissMap {
		t.Skip("mapvalues not implemented for swissmaps")
	}

	type val struct {
		s   string
		pad [128]byte // sizeof(val) > abi.MapMaxElemBytes
	}
	m := map[int]val{1: {s: "a"}, 2: {s: "b"}}
	vals := make([]val, 0, len(m))
	runtime.MapValues(m, unsafe.Pointer(&vals))
	for _, v := range vals {
		if len(v.s) != 1 {
			t.Errorf("len(v.s) == %d, want 1", len(v.s))
		}
	}
}

func computeHash() uintptr {
	var v struct{}
	return runtime.MemHash(unsafe.Pointer(&v), 0, unsafe.Sizeof(v))
}

func subprocessHash(t *testing.T, env string) uintptr {
	t.Helper()

	cmd := testenv.CleanCmdEnv(testenv.Command(t, os.Args[0], "-test.run=^TestMemHashGlobalSeed$"))
	cmd.Env = append(cmd.Env, "GO_TEST_SUBPROCESS_HASH=1")
	if env != "" {
		cmd.Env = append(cmd.Env, env)
	}

	out, err := cmd.Output()
	if err != nil {
		t.Fatalf("cmd.Output got err %v want nil", err)
	}

	s := strings.TrimSpace(string(out))
	h, err := strconv.ParseUint(s, 10, 64)
	if err != nil {
		t.Fatalf("Parse output %q got err %v want nil", s, err)
	}
	return uintptr(h)
}

// memhash has unique per-process seeds, so hashes should differ across
// processes.
//
// Regression test for https://go.dev/issue/66885.
func TestMemHashGlobalSeed(t *testing.T) {
	if os.Getenv("GO_TEST_SUBPROCESS_HASH") != "" {
		fmt.Println(computeHash())
		os.Exit(0)
		return
	}

	testenv.MustHaveExec(t)

	// aeshash and memhashFallback use separate per-process seeds, so test
	// both.
	t.Run("aes", func(t *testing.T) {
		if !*runtime.UseAeshash {
			t.Skip("No AES")
		}

		h1 := subprocessHash(t, "")
		t.Logf("%d", h1)
		h2 := subprocessHash(t, "")
		t.Logf("%d", h2)
		h3 := subprocessHash(t, "")
		t.Logf("%d", h3)

		if h1 == h2 && h2 == h3 {
			t.Errorf("got duplicate hash %d want unique", h1)
		}
	})

	t.Run("noaes", func(t *testing.T) {
		env := ""
		if *runtime.UseAeshash {
			env = "GODEBUG=cpu.aes=off"
		}

		h1 := subprocessHash(t, env)
		t.Logf("%d", h1)
		h2 := subprocessHash(t, env)
		t.Logf("%d", h2)
		h3 := subprocessHash(t, env)
		t.Logf("%d", h3)

		if h1 == h2 && h2 == h3 {
			t.Errorf("got duplicate hash %d want unique", h1)
		}
	})
}

func TestMapIterDeleteReplace(t *testing.T) {
	inc := 1
	if testing.Short() {
		inc = 100
	}
	for i := 0; i < 10000; i += inc {
		t.Run(fmt.Sprint(i), func(t *testing.T) {
			m := make(map[int]bool)
			for j := range i {
				m[j] = false
			}

			// Delete and replace all entries.
			for k := range m {
				delete(m, k)
				m[k] = true
			}

			for k, v := range m {
				if !v {
					t.Errorf("m[%d] got false want true", k)
				}
			}
		})
	}
}
