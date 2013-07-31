// Copyright 2013 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

package runtime_test

import (
	"fmt"
	"math"
	"reflect"
	"runtime"
	"sort"
	"strings"
	"sync"
	"testing"
)

// negative zero is a good test because:
//  1) 0 and -0 are equal, yet have distinct representations.
//  2) 0 is represented as all zeros, -0 isn't.
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

// nan is a good test because nan != nan, and nan has
// a randomized hash value.
func TestNan(t *testing.T) {
	m := make(map[float64]int, 0)
	nan := math.NaN()
	m[nan] = 1
	m[nan] = 2
	m[nan] = 4
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
	m[nan] = 1
	m[nan] = 2
	m[nan] = 4
	cnt := 0
	s := 0
	growflag := true
	for k, v := range m {
		if growflag {
			// force a hashtable resize
			for i := 0; i < 100; i++ {
				m[float64(i)] = i
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
	m[FloatInt{0.0, 1}] = 2
	m[FloatInt{0.0, 2}] = 4
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
	for i := 0; i < 16; i++ {
		m[i] = i
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
	if runtime.GOMAXPROCS(-1) == 1 {
		defer runtime.GOMAXPROCS(runtime.GOMAXPROCS(16))
	}
	numLoop := 10
	numGrowStep := 250
	numReader := 16
	if testing.Short() {
		numLoop, numGrowStep = 2, 500
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
					for _ = range m {
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
	sort.Strings(keys[:])
	sort.Strings(values[:])
	for i := 0; i < 100; i++ {
		if keys[i] != fmt.Sprintf("string%02d", i) {
			t.Errorf("#%d: missing key: %v", i, keys[i])
		}
		if values[i] != fmt.Sprintf("string%02d", i) {
			t.Errorf("#%d: missing value: %v", i, values[i])
		}
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
	if b[empty{}] != 1 {
		t.Errorf("empty key returned wrong value")
	}
}

// Tests a map with a single bucket, with same-lengthed short keys
// ("quick keys") as well as long keys.
func TestSingleBucketMapStringKeys_DupLen(t *testing.T) {
	testMapLookups(t, map[string]string{
		"x":    "x1val",
		"xx":   "x2val",
		"foo":  "fooval",
		"bar":  "barval", // same key length as "foo"
		"xxxx": "x4val",
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
