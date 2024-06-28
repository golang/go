// Copyright 2013 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

package runtime_test

import (
	"encoding/binary"
	"fmt"
	"math/rand"
	"runtime"
	"slices"
	"strconv"
	"strings"
	"testing"
	"unsafe"
)

const size = 10

func BenchmarkHashStringSpeed(b *testing.B) {
	strings := make([]string, size)
	for i := 0; i < size; i++ {
		strings[i] = fmt.Sprintf("string#%d", i)
	}
	sum := 0
	m := make(map[string]int, size)
	for i := 0; i < size; i++ {
		m[strings[i]] = 0
	}
	idx := 0
	b.ResetTimer()
	for i := 0; i < b.N; i++ {
		sum += m[strings[idx]]
		idx++
		if idx == size {
			idx = 0
		}
	}
}

type chunk [17]byte

func BenchmarkHashBytesSpeed(b *testing.B) {
	// a bunch of chunks, each with a different alignment mod 16
	var chunks [size]chunk
	// initialize each to a different value
	for i := 0; i < size; i++ {
		chunks[i][0] = byte(i)
	}
	// put into a map
	m := make(map[chunk]int, size)
	for i, c := range chunks {
		m[c] = i
	}
	idx := 0
	b.ResetTimer()
	for i := 0; i < b.N; i++ {
		if m[chunks[idx]] != idx {
			b.Error("bad map entry for chunk")
		}
		idx++
		if idx == size {
			idx = 0
		}
	}
}

func BenchmarkHashInt32Speed(b *testing.B) {
	ints := make([]int32, size)
	for i := 0; i < size; i++ {
		ints[i] = int32(i)
	}
	sum := 0
	m := make(map[int32]int, size)
	for i := 0; i < size; i++ {
		m[ints[i]] = 0
	}
	idx := 0
	b.ResetTimer()
	for i := 0; i < b.N; i++ {
		sum += m[ints[idx]]
		idx++
		if idx == size {
			idx = 0
		}
	}
}

func BenchmarkHashInt64Speed(b *testing.B) {
	ints := make([]int64, size)
	for i := 0; i < size; i++ {
		ints[i] = int64(i)
	}
	sum := 0
	m := make(map[int64]int, size)
	for i := 0; i < size; i++ {
		m[ints[i]] = 0
	}
	idx := 0
	b.ResetTimer()
	for i := 0; i < b.N; i++ {
		sum += m[ints[idx]]
		idx++
		if idx == size {
			idx = 0
		}
	}
}
func BenchmarkHashStringArraySpeed(b *testing.B) {
	stringpairs := make([][2]string, size)
	for i := 0; i < size; i++ {
		for j := 0; j < 2; j++ {
			stringpairs[i][j] = fmt.Sprintf("string#%d/%d", i, j)
		}
	}
	sum := 0
	m := make(map[[2]string]int, size)
	for i := 0; i < size; i++ {
		m[stringpairs[i]] = 0
	}
	idx := 0
	b.ResetTimer()
	for i := 0; i < b.N; i++ {
		sum += m[stringpairs[idx]]
		idx++
		if idx == size {
			idx = 0
		}
	}
}

func BenchmarkMegMap(b *testing.B) {
	m := make(map[string]bool)
	for suffix := 'A'; suffix <= 'G'; suffix++ {
		m[strings.Repeat("X", 1<<20-1)+fmt.Sprint(suffix)] = true
	}
	key := strings.Repeat("X", 1<<20-1) + "k"
	b.ResetTimer()
	for i := 0; i < b.N; i++ {
		_, _ = m[key]
	}
}

func BenchmarkMegOneMap(b *testing.B) {
	m := make(map[string]bool)
	m[strings.Repeat("X", 1<<20)] = true
	key := strings.Repeat("Y", 1<<20)
	b.ResetTimer()
	for i := 0; i < b.N; i++ {
		_, _ = m[key]
	}
}

func BenchmarkMegEqMap(b *testing.B) {
	m := make(map[string]bool)
	key1 := strings.Repeat("X", 1<<20)
	key2 := strings.Repeat("X", 1<<20) // equal but different instance
	m[key1] = true
	b.ResetTimer()
	for i := 0; i < b.N; i++ {
		_, _ = m[key2]
	}
}

func BenchmarkMegEmptyMap(b *testing.B) {
	m := make(map[string]bool)
	key := strings.Repeat("X", 1<<20)
	b.ResetTimer()
	for i := 0; i < b.N; i++ {
		_, _ = m[key]
	}
}

func BenchmarkMegEmptyMapWithInterfaceKey(b *testing.B) {
	m := make(map[any]bool)
	key := strings.Repeat("X", 1<<20)
	b.ResetTimer()
	for i := 0; i < b.N; i++ {
		_, _ = m[key]
	}
}

func BenchmarkSmallStrMap(b *testing.B) {
	m := make(map[string]bool)
	for suffix := 'A'; suffix <= 'G'; suffix++ {
		m[fmt.Sprint(suffix)] = true
	}
	key := "k"
	b.ResetTimer()
	for i := 0; i < b.N; i++ {
		_, _ = m[key]
	}
}

func BenchmarkMapStringKeysEight_16(b *testing.B) { benchmarkMapStringKeysEight(b, 16) }
func BenchmarkMapStringKeysEight_32(b *testing.B) { benchmarkMapStringKeysEight(b, 32) }
func BenchmarkMapStringKeysEight_64(b *testing.B) { benchmarkMapStringKeysEight(b, 64) }
func BenchmarkMapStringKeysEight_1M(b *testing.B) { benchmarkMapStringKeysEight(b, 1<<20) }

func benchmarkMapStringKeysEight(b *testing.B, keySize int) {
	m := make(map[string]bool)
	for i := 0; i < 8; i++ {
		m[strings.Repeat("K", i+1)] = true
	}
	key := strings.Repeat("K", keySize)
	b.ResetTimer()
	for i := 0; i < b.N; i++ {
		_ = m[key]
	}
}

func BenchmarkMapFirst(b *testing.B) {
	for n := 1; n <= 16; n++ {
		b.Run(fmt.Sprintf("%d", n), func(b *testing.B) {
			m := make(map[int]bool)
			for i := 0; i < n; i++ {
				m[i] = true
			}
			b.ResetTimer()
			for i := 0; i < b.N; i++ {
				_ = m[0]
			}
		})
	}
}
func BenchmarkMapMid(b *testing.B) {
	for n := 1; n <= 16; n++ {
		b.Run(fmt.Sprintf("%d", n), func(b *testing.B) {
			m := make(map[int]bool)
			for i := 0; i < n; i++ {
				m[i] = true
			}
			b.ResetTimer()
			for i := 0; i < b.N; i++ {
				_ = m[n>>1]
			}
		})
	}
}
func BenchmarkMapLast(b *testing.B) {
	for n := 1; n <= 16; n++ {
		b.Run(fmt.Sprintf("%d", n), func(b *testing.B) {
			m := make(map[int]bool)
			for i := 0; i < n; i++ {
				m[i] = true
			}
			b.ResetTimer()
			for i := 0; i < b.N; i++ {
				_ = m[n-1]
			}
		})
	}
}

func BenchmarkMapCycle(b *testing.B) {
	// Arrange map entries to be a permutation, so that
	// we hit all entries, and one lookup is data dependent
	// on the previous lookup.
	const N = 3127
	p := rand.New(rand.NewSource(1)).Perm(N)
	m := map[int]int{}
	for i := 0; i < N; i++ {
		m[i] = p[i]
	}
	b.ResetTimer()
	j := 0
	for i := 0; i < b.N; i++ {
		j = m[j]
	}
	sink = uint64(j)
}

// Accessing the same keys in a row.
func benchmarkRepeatedLookup(b *testing.B, lookupKeySize int) {
	m := make(map[string]bool)
	// At least bigger than a single bucket:
	for i := 0; i < 64; i++ {
		m[fmt.Sprintf("some key %d", i)] = true
	}
	base := strings.Repeat("x", lookupKeySize-1)
	key1 := base + "1"
	key2 := base + "2"
	b.ResetTimer()
	for i := 0; i < b.N/4; i++ {
		_ = m[key1]
		_ = m[key1]
		_ = m[key2]
		_ = m[key2]
	}
}

func BenchmarkRepeatedLookupStrMapKey32(b *testing.B) { benchmarkRepeatedLookup(b, 32) }
func BenchmarkRepeatedLookupStrMapKey1M(b *testing.B) { benchmarkRepeatedLookup(b, 1<<20) }

func BenchmarkMakeMap(b *testing.B) {
	b.Run("[Byte]Byte", func(b *testing.B) {
		var m map[byte]byte
		for i := 0; i < b.N; i++ {
			m = make(map[byte]byte, 10)
		}
		hugeSink = m
	})
	b.Run("[Int]Int", func(b *testing.B) {
		var m map[int]int
		for i := 0; i < b.N; i++ {
			m = make(map[int]int, 10)
		}
		hugeSink = m
	})
}

func BenchmarkNewEmptyMap(b *testing.B) {
	b.ReportAllocs()
	for i := 0; i < b.N; i++ {
		_ = make(map[int]int)
	}
}

func BenchmarkNewSmallMap(b *testing.B) {
	b.ReportAllocs()
	for i := 0; i < b.N; i++ {
		m := make(map[int]int)
		m[0] = 0
		m[1] = 1
	}
}

func BenchmarkSameLengthMap(b *testing.B) {
	// long strings, same length, differ in first few
	// and last few bytes.
	m := make(map[string]bool)
	s1 := "foo" + strings.Repeat("-", 100) + "bar"
	s2 := "goo" + strings.Repeat("-", 100) + "ber"
	m[s1] = true
	m[s2] = true
	b.ResetTimer()
	for i := 0; i < b.N; i++ {
		_ = m[s1]
	}
}

func BenchmarkSmallKeyMap(b *testing.B) {
	m := make(map[int16]bool)
	m[5] = true
	for i := 0; i < b.N; i++ {
		_ = m[5]
	}
}

func BenchmarkMapPopulate(b *testing.B) {
	for size := 1; size < 1000000; size *= 10 {
		b.Run(strconv.Itoa(size), func(b *testing.B) {
			b.ReportAllocs()
			for i := 0; i < b.N; i++ {
				m := make(map[int]bool)
				for j := 0; j < size; j++ {
					m[j] = true
				}
			}
		})
	}
}

type ComplexAlgKey struct {
	a, b, c int64
	_       int
	d       int32
	_       int
	e       string
	_       int
	f, g, h int64
}

func BenchmarkComplexAlgMap(b *testing.B) {
	m := make(map[ComplexAlgKey]bool)
	var k ComplexAlgKey
	m[k] = true
	for i := 0; i < b.N; i++ {
		_ = m[k]
	}
}

func BenchmarkGoMapClear(b *testing.B) {
	b.Run("Reflexive", func(b *testing.B) {
		for size := 1; size < 100000; size *= 10 {
			b.Run(strconv.Itoa(size), func(b *testing.B) {
				m := make(map[int]int, size)
				for i := 0; i < b.N; i++ {
					m[0] = size // Add one element so len(m) != 0 avoiding fast paths.
					clear(m)
				}
			})
		}
	})
	b.Run("NonReflexive", func(b *testing.B) {
		for size := 1; size < 100000; size *= 10 {
			b.Run(strconv.Itoa(size), func(b *testing.B) {
				m := make(map[float64]int, size)
				for i := 0; i < b.N; i++ {
					m[1.0] = size // Add one element so len(m) != 0 avoiding fast paths.
					clear(m)
				}
			})
		}
	})
}

func BenchmarkMapStringConversion(b *testing.B) {
	for _, length := range []int{32, 64} {
		b.Run(strconv.Itoa(length), func(b *testing.B) {
			bytes := make([]byte, length)
			b.Run("simple", func(b *testing.B) {
				b.ReportAllocs()
				m := make(map[string]int)
				m[string(bytes)] = 0
				for i := 0; i < b.N; i++ {
					_ = m[string(bytes)]
				}
			})
			b.Run("struct", func(b *testing.B) {
				b.ReportAllocs()
				type stringstruct struct{ s string }
				m := make(map[stringstruct]int)
				m[stringstruct{string(bytes)}] = 0
				for i := 0; i < b.N; i++ {
					_ = m[stringstruct{string(bytes)}]
				}
			})
			b.Run("array", func(b *testing.B) {
				b.ReportAllocs()
				type stringarray [1]string
				m := make(map[stringarray]int)
				m[stringarray{string(bytes)}] = 0
				for i := 0; i < b.N; i++ {
					_ = m[stringarray{string(bytes)}]
				}
			})
		})
	}
}

var BoolSink bool

func BenchmarkMapInterfaceString(b *testing.B) {
	m := map[any]bool{}

	for i := 0; i < 100; i++ {
		m[fmt.Sprintf("%d", i)] = true
	}

	key := (any)("A")
	b.ResetTimer()
	for i := 0; i < b.N; i++ {
		BoolSink = m[key]
	}
}
func BenchmarkMapInterfacePtr(b *testing.B) {
	m := map[any]bool{}

	for i := 0; i < 100; i++ {
		i := i
		m[&i] = true
	}

	key := new(int)
	b.ResetTimer()
	for i := 0; i < b.N; i++ {
		BoolSink = m[key]
	}
}

var (
	hintLessThan8    = 7
	hintGreaterThan8 = 32
)

func BenchmarkNewEmptyMapHintLessThan8(b *testing.B) {
	b.ReportAllocs()
	for i := 0; i < b.N; i++ {
		_ = make(map[int]int, hintLessThan8)
	}
}

func BenchmarkNewEmptyMapHintGreaterThan8(b *testing.B) {
	b.ReportAllocs()
	for i := 0; i < b.N; i++ {
		_ = make(map[int]int, hintGreaterThan8)
	}
}

func benchSizes(f func(b *testing.B, n int)) func(*testing.B) {
	var cases = []int{
		0,
		6,
		12,
		18,
		24,
		30,
		64,
		128,
		256,
		512,
		1024,
		2048,
		4096,
		8192,
		1 << 16,
		1 << 18,
		1 << 20,
		1 << 22,
	}

	return func(b *testing.B) {
		for _, n := range cases {
			b.Run("len="+strconv.Itoa(n), func(b *testing.B) {
				f(b, n)
			})
		}
	}
}

// A 16 byte type.
type smallType [16]byte

// A 512 byte type.
type mediumType [1 << 9]byte

// A 4KiB type.
type bigType [1 << 12]byte

type mapBenchmarkKeyType interface {
	int32 | int64 | string | smallType | mediumType | bigType | *int32
}

type mapBenchmarkElemType interface {
	mapBenchmarkKeyType | []int32
}

func genIntValues[T int | int32 | int64](start, end int) []T {
	vals := make([]T, 0, end-start)
	for i := start; i < end; i++ {
		vals = append(vals, T(i))
	}
	return vals
}

func genStringValues(start, end int) []string {
	vals := make([]string, 0, end-start)
	for i := start; i < end; i++ {
		vals = append(vals, strconv.Itoa(i))
	}
	return vals
}

func genSmallValues(start, end int) []smallType {
	vals := make([]smallType, 0, end-start)
	for i := start; i < end; i++ {
		var v smallType
		binary.NativeEndian.PutUint64(v[:], uint64(i))
		vals = append(vals, v)
	}
	return vals
}

func genMediumValues(start, end int) []mediumType {
	vals := make([]mediumType, 0, end-start)
	for i := start; i < end; i++ {
		var v mediumType
		binary.NativeEndian.PutUint64(v[:], uint64(i))
		vals = append(vals, v)
	}
	return vals
}

func genBigValues(start, end int) []bigType {
	vals := make([]bigType, 0, end-start)
	for i := start; i < end; i++ {
		var v bigType
		binary.NativeEndian.PutUint64(v[:], uint64(i))
		vals = append(vals, v)
	}
	return vals
}

func genPtrValues[T any](start, end int) []*T {
	// Start and end don't mean much. Each pointer by definition has a
	// unique identity.
	vals := make([]*T, 0, end-start)
	for i := start; i < end; i++ {
		v := new(T)
		vals = append(vals, v)
	}
	return vals
}

func genIntSliceValues[T int | int32 | int64](start, end int) [][]T {
	vals := make([][]T, 0, end-start)
	for i := start; i < end; i++ {
		vals = append(vals, []T{T(i)})
	}
	return vals
}

func genValues[T mapBenchmarkElemType](start, end int) []T {
	var t T
	switch any(t).(type) {
	case int32:
		return any(genIntValues[int32](start, end)).([]T)
	case int64:
		return any(genIntValues[int64](start, end)).([]T)
	case string:
		return any(genStringValues(start, end)).([]T)
	case smallType:
		return any(genSmallValues(start, end)).([]T)
	case mediumType:
		return any(genMediumValues(start, end)).([]T)
	case bigType:
		return any(genBigValues(start, end)).([]T)
	case *int32:
		return any(genPtrValues[int32](start, end)).([]T)
	case []int32:
		return any(genIntSliceValues[int32](start, end)).([]T)
	default:
		panic("unreachable")
	}
}

// Avoid inlining to force a heap allocation.
//
//go:noinline
func newSink[T mapBenchmarkElemType]() *T {
	return new(T)
}

// Return a new maps filled with keys and elems. Both slices must be the same length.
func fillMap[K mapBenchmarkKeyType, E mapBenchmarkElemType](keys []K, elems []E) map[K]E {
	m := make(map[K]E, len(keys))
	for i := range keys {
		m[keys[i]] = elems[i]
	}
	return m
}

func iterCount(b *testing.B, n int) int {
	// Divide b.N by n so that the ns/op reports time per element,
	// not time per full map iteration. This makes benchmarks of
	// different map sizes more comparable.
	//
	// If size is zero we still need to do iterations.
	if n == 0 {
		return b.N
	}
	return b.N / n
}

func checkAllocSize[K, E any](b *testing.B, n int) {
	var k K
	size := uint64(n) * uint64(unsafe.Sizeof(k))
	var e E
	size += uint64(n) * uint64(unsafe.Sizeof(e))

	if size >= 1<<30 {
		b.Skipf("Total key+elem size %d exceeds 1GiB", size)
	}
}

func benchmarkMapIter[K mapBenchmarkKeyType, E mapBenchmarkElemType](b *testing.B, n int) {
	checkAllocSize[K, E](b, n)
	k := genValues[K](0, n)
	e := genValues[E](0, n)
	m := fillMap(k, e)
	iterations := iterCount(b, n)
	sinkK := newSink[K]()
	sinkE := newSink[E]()
	b.ResetTimer()

	for i := 0; i < iterations; i++ {
		for k, e := range m {
			*sinkK = k
			*sinkE = e
		}
	}
}

func BenchmarkMapIter(b *testing.B) {
	b.Run("Key=int32/Elem=int32", benchSizes(benchmarkMapIter[int32, int32]))
	b.Run("Key=int64/Elem=int64", benchSizes(benchmarkMapIter[int64, int64]))
	b.Run("Key=string/Elem=string", benchSizes(benchmarkMapIter[string, string]))
	b.Run("Key=smallType/Elem=int32", benchSizes(benchmarkMapIter[smallType, int32]))
	b.Run("Key=mediumType/Elem=int32", benchSizes(benchmarkMapIter[mediumType, int32]))
	b.Run("Key=bigType/Elem=int32", benchSizes(benchmarkMapIter[bigType, int32]))
	b.Run("Key=bigType/Elem=bigType", benchSizes(benchmarkMapIter[bigType, bigType]))
	b.Run("Key=int32/Elem=bigType", benchSizes(benchmarkMapIter[int32, bigType]))
	b.Run("Key=*int32/Elem=int32", benchSizes(benchmarkMapIter[*int32, int32]))
	b.Run("Key=int32/Elem=*int32", benchSizes(benchmarkMapIter[int32, *int32]))
}

func benchmarkMapAccessHit[K mapBenchmarkKeyType, E mapBenchmarkElemType](b *testing.B, n int) {
	if n == 0 {
		b.Skip("can't access empty map")
	}
	checkAllocSize[K, E](b, n)
	k := genValues[K](0, n)
	e := genValues[E](0, n)
	m := fillMap(k, e)
	sink := newSink[E]()
	b.ResetTimer()

	for i := 0; i < b.N; i++ {
		*sink = m[k[i%n]]
	}
}

func BenchmarkMapAccessHit(b *testing.B) {
	b.Run("Key=int32/Elem=int32", benchSizes(benchmarkMapAccessHit[int32, int32]))
	b.Run("Key=int64/Elem=int64", benchSizes(benchmarkMapAccessHit[int64, int64]))
	b.Run("Key=string/Elem=string", benchSizes(benchmarkMapAccessHit[string, string]))
	b.Run("Key=smallType/Elem=int32", benchSizes(benchmarkMapAccessHit[smallType, int32]))
	b.Run("Key=mediumType/Elem=int32", benchSizes(benchmarkMapAccessHit[mediumType, int32]))
	b.Run("Key=bigType/Elem=int32", benchSizes(benchmarkMapAccessHit[bigType, int32]))
	b.Run("Key=bigType/Elem=bigType", benchSizes(benchmarkMapAccessHit[bigType, bigType]))
	b.Run("Key=int32/Elem=bigType", benchSizes(benchmarkMapAccessHit[int32, bigType]))
	b.Run("Key=*int32/Elem=int32", benchSizes(benchmarkMapAccessHit[*int32, int32]))
	b.Run("Key=int32/Elem=*int32", benchSizes(benchmarkMapAccessHit[int32, *int32]))
}

var sinkOK bool

func benchmarkMapAccessMiss[K mapBenchmarkKeyType, E mapBenchmarkElemType](b *testing.B, n int) {
	checkAllocSize[K, E](b, n)
	k := genValues[K](0, n)
	e := genValues[E](0, n)
	m := fillMap(k, e)
	if n == 0 { // Create a lookup values for empty maps.
		n = 1
	}
	w := genValues[K](n, 2*n)
	b.ResetTimer()

	var ok bool
	for i := 0; i < b.N; i++ {
		_, ok = m[w[i%n]]
	}

	sinkOK = ok
}

func BenchmarkMapAccessMiss(b *testing.B) {
	b.Run("Key=int32/Elem=int32", benchSizes(benchmarkMapAccessMiss[int32, int32]))
	b.Run("Key=int64/Elem=int64", benchSizes(benchmarkMapAccessMiss[int64, int64]))
	b.Run("Key=string/Elem=string", benchSizes(benchmarkMapAccessMiss[string, string]))
	b.Run("Key=smallType/Elem=int32", benchSizes(benchmarkMapAccessMiss[smallType, int32]))
	b.Run("Key=mediumType/Elem=int32", benchSizes(benchmarkMapAccessMiss[mediumType, int32]))
	b.Run("Key=bigType/Elem=int32", benchSizes(benchmarkMapAccessMiss[bigType, int32]))
	b.Run("Key=bigType/Elem=bigType", benchSizes(benchmarkMapAccessMiss[bigType, bigType]))
	b.Run("Key=int32/Elem=bigType", benchSizes(benchmarkMapAccessMiss[int32, bigType]))
	b.Run("Key=*int32/Elem=int32", benchSizes(benchmarkMapAccessMiss[*int32, int32]))
	b.Run("Key=int32/Elem=*int32", benchSizes(benchmarkMapAccessMiss[int32, *int32]))
}

// Assign to a key that already exists.
func benchmarkMapAssignExists[K mapBenchmarkKeyType, E mapBenchmarkElemType](b *testing.B, n int) {
	if n == 0 {
		b.Skip("can't assign to existing keys in empty map")
	}
	checkAllocSize[K, E](b, n)
	k := genValues[K](0, n)
	e := genValues[E](0, n)
	m := fillMap(k, e)
	b.ResetTimer()

	for i := 0; i < b.N; i++ {
		m[k[i%n]] = e[i%n]
	}
}

func BenchmarkMapAssignExists(b *testing.B) {
	b.Run("Key=int32/Elem=int32", benchSizes(benchmarkMapAssignExists[int32, int32]))
	b.Run("Key=int64/Elem=int64", benchSizes(benchmarkMapAssignExists[int64, int64]))
	b.Run("Key=string/Elem=string", benchSizes(benchmarkMapAssignExists[string, string]))
	b.Run("Key=smallType/Elem=int32", benchSizes(benchmarkMapAssignExists[smallType, int32]))
	b.Run("Key=mediumType/Elem=int32", benchSizes(benchmarkMapAssignExists[mediumType, int32]))
	b.Run("Key=bigType/Elem=int32", benchSizes(benchmarkMapAssignExists[bigType, int32]))
	b.Run("Key=bigType/Elem=bigType", benchSizes(benchmarkMapAssignExists[bigType, bigType]))
	b.Run("Key=int32/Elem=bigType", benchSizes(benchmarkMapAssignExists[int32, bigType]))
	b.Run("Key=*int32/Elem=int32", benchSizes(benchmarkMapAssignExists[*int32, int32]))
	b.Run("Key=int32/Elem=*int32", benchSizes(benchmarkMapAssignExists[int32, *int32]))
}

// Fill a map of size n with no hint. Time is per-key. A new map is created
// every n assignments.
//
// TODO(prattmic): Results don't make much sense if b.N < n.
// TODO(prattmic): Measure distribution of assign time to reveal the grow
// latency.
func benchmarkMapAssignFillNoHint[K mapBenchmarkKeyType, E mapBenchmarkElemType](b *testing.B, n int) {
	if n == 0 {
		b.Skip("can't create empty map via assignment")
	}
	checkAllocSize[K, E](b, n)
	k := genValues[K](0, n)
	e := genValues[E](0, n)
	b.ResetTimer()

	var m map[K]E
	for i := 0; i < b.N; i++ {
		if i%n == 0 {
			m = make(map[K]E)
		}
		m[k[i%n]] = e[i%n]
	}
}

func BenchmarkMapAssignFillNoHint(b *testing.B) {
	b.Run("Key=int32/Elem=int32", benchSizes(benchmarkMapAssignFillNoHint[int32, int32]))
	b.Run("Key=int64/Elem=int64", benchSizes(benchmarkMapAssignFillNoHint[int64, int64]))
	b.Run("Key=string/Elem=string", benchSizes(benchmarkMapAssignFillNoHint[string, string]))
	b.Run("Key=smallType/Elem=int32", benchSizes(benchmarkMapAssignFillNoHint[smallType, int32]))
	b.Run("Key=mediumType/Elem=int32", benchSizes(benchmarkMapAssignFillNoHint[mediumType, int32]))
	b.Run("Key=bigType/Elem=int32", benchSizes(benchmarkMapAssignFillNoHint[bigType, int32]))
	b.Run("Key=bigType/Elem=bigType", benchSizes(benchmarkMapAssignFillNoHint[bigType, bigType]))
	b.Run("Key=int32/Elem=bigType", benchSizes(benchmarkMapAssignFillNoHint[int32, bigType]))
	b.Run("Key=*int32/Elem=int32", benchSizes(benchmarkMapAssignFillNoHint[*int32, int32]))
	b.Run("Key=int32/Elem=*int32", benchSizes(benchmarkMapAssignFillNoHint[int32, *int32]))
}

// Identical to benchmarkMapAssignFillNoHint, but additionally measures the
// latency of each mapassign to report tail latency due to map grow.
func benchmarkMapAssignGrowLatency[K mapBenchmarkKeyType, E mapBenchmarkElemType](b *testing.B, n int) {
	if n == 0 {
		b.Skip("can't create empty map via assignment")
	}
	checkAllocSize[K, E](b, n)
	k := genValues[K](0, n)
	e := genValues[E](0, n)

	// Store the run time of each mapassign. Keeping the full data rather
	// than a histogram provides higher precision. b.N tends to be <10M, so
	// the memory requirement isn't too bad.
	sample := make([]int64, b.N)

	b.ResetTimer()

	var m map[K]E
	for i := 0; i < b.N; i++ {
		if i%n == 0 {
			m = make(map[K]E)
		}
		start := runtime.Nanotime()
		m[k[i%n]] = e[i%n]
		end := runtime.Nanotime()
		sample[i] = end - start
	}

	b.StopTimer()

	slices.Sort(sample)
	// TODO(prattmic): Grow is so rare that even p99.99 often doesn't
	// display a grow case. Switch to a more direct measure of grow cases
	// only?
	b.ReportMetric(float64(sample[int(float64(len(sample))*0.5)]), "p50-ns/op")
	b.ReportMetric(float64(sample[int(float64(len(sample))*0.99)]), "p99-ns/op")
	b.ReportMetric(float64(sample[int(float64(len(sample))*0.999)]), "p99.9-ns/op")
	b.ReportMetric(float64(sample[int(float64(len(sample))*0.9999)]), "p99.99-ns/op")
	b.ReportMetric(float64(sample[len(sample)-1]), "p100-ns/op")
}

func BenchmarkMapAssignGrowLatency(b *testing.B) {
	b.Run("Key=int32/Elem=int32", benchSizes(benchmarkMapAssignGrowLatency[int32, int32]))
	b.Run("Key=int64/Elem=int64", benchSizes(benchmarkMapAssignGrowLatency[int64, int64]))
	b.Run("Key=string/Elem=string", benchSizes(benchmarkMapAssignGrowLatency[string, string]))
	b.Run("Key=smallType/Elem=int32", benchSizes(benchmarkMapAssignGrowLatency[smallType, int32]))
	b.Run("Key=mediumType/Elem=int32", benchSizes(benchmarkMapAssignGrowLatency[mediumType, int32]))
	b.Run("Key=bigType/Elem=int32", benchSizes(benchmarkMapAssignGrowLatency[bigType, int32]))
	b.Run("Key=bigType/Elem=bigType", benchSizes(benchmarkMapAssignGrowLatency[bigType, bigType]))
	b.Run("Key=int32/Elem=bigType", benchSizes(benchmarkMapAssignGrowLatency[int32, bigType]))
	b.Run("Key=*int32/Elem=int32", benchSizes(benchmarkMapAssignGrowLatency[*int32, int32]))
	b.Run("Key=int32/Elem=*int32", benchSizes(benchmarkMapAssignGrowLatency[int32, *int32]))
}

// Fill a map of size n with size hint. Time is per-key. A new map is created
// every n assignments.
//
// TODO(prattmic): Results don't make much sense if b.N < n.
func benchmarkMapAssignFillHint[K mapBenchmarkKeyType, E mapBenchmarkElemType](b *testing.B, n int) {
	if n == 0 {
		b.Skip("can't create empty map via assignment")
	}
	checkAllocSize[K, E](b, n)
	k := genValues[K](0, n)
	e := genValues[E](0, n)
	b.ResetTimer()

	var m map[K]E
	for i := 0; i < b.N; i++ {
		if i%n == 0 {
			m = make(map[K]E, n)
		}
		m[k[i%n]] = e[i%n]
	}
}

func BenchmarkMapAssignFillHint(b *testing.B) {
	b.Run("Key=int32/Elem=int32", benchSizes(benchmarkMapAssignFillHint[int32, int32]))
	b.Run("Key=int64/Elem=int64", benchSizes(benchmarkMapAssignFillHint[int64, int64]))
	b.Run("Key=string/Elem=string", benchSizes(benchmarkMapAssignFillHint[string, string]))
	b.Run("Key=smallType/Elem=int32", benchSizes(benchmarkMapAssignFillHint[smallType, int32]))
	b.Run("Key=mediumType/Elem=int32", benchSizes(benchmarkMapAssignFillHint[mediumType, int32]))
	b.Run("Key=bigType/Elem=int32", benchSizes(benchmarkMapAssignFillHint[bigType, int32]))
	b.Run("Key=bigType/Elem=bigType", benchSizes(benchmarkMapAssignFillHint[bigType, bigType]))
	b.Run("Key=int32/Elem=bigType", benchSizes(benchmarkMapAssignFillHint[int32, bigType]))
	b.Run("Key=*int32/Elem=int32", benchSizes(benchmarkMapAssignFillHint[*int32, int32]))
	b.Run("Key=int32/Elem=*int32", benchSizes(benchmarkMapAssignFillHint[int32, *int32]))
}

// Fill a map of size n, reusing the same map. Time is per-key. The map is
// cleared every n assignments.
//
// TODO(prattmic): Results don't make much sense if b.N < n.
func benchmarkMapAssignFillClear[K mapBenchmarkKeyType, E mapBenchmarkElemType](b *testing.B, n int) {
	if n == 0 {
		b.Skip("can't create empty map via assignment")
	}
	checkAllocSize[K, E](b, n)
	k := genValues[K](0, n)
	e := genValues[E](0, n)
	m := fillMap(k, e)
	b.ResetTimer()

	for i := 0; i < b.N; i++ {
		if i%n == 0 {
			clear(m)
		}
		m[k[i%n]] = e[i%n]
	}
}

func BenchmarkMapAssignFillClear(b *testing.B) {
	b.Run("Key=int32/Elem=int32", benchSizes(benchmarkMapAssignFillClear[int32, int32]))
	b.Run("Key=int64/Elem=int64", benchSizes(benchmarkMapAssignFillClear[int64, int64]))
	b.Run("Key=string/Elem=string", benchSizes(benchmarkMapAssignFillClear[string, string]))
	b.Run("Key=smallType/Elem=int32", benchSizes(benchmarkMapAssignFillClear[smallType, int32]))
	b.Run("Key=mediumType/Elem=int32", benchSizes(benchmarkMapAssignFillClear[mediumType, int32]))
	b.Run("Key=bigType/Elem=int32", benchSizes(benchmarkMapAssignFillClear[bigType, int32]))
	b.Run("Key=bigType/Elem=bigType", benchSizes(benchmarkMapAssignFillClear[bigType, bigType]))
	b.Run("Key=int32/Elem=bigType", benchSizes(benchmarkMapAssignFillClear[int32, bigType]))
	b.Run("Key=*int32/Elem=int32", benchSizes(benchmarkMapAssignFillClear[*int32, int32]))
	b.Run("Key=int32/Elem=*int32", benchSizes(benchmarkMapAssignFillClear[int32, *int32]))
}

// Modify values using +=.
func benchmarkMapAssignAddition[K mapBenchmarkKeyType, E int32 | int64 | string](b *testing.B, n int) {
	if n == 0 {
		b.Skip("can't modify empty map via assignment")
	}
	checkAllocSize[K, E](b, n)
	k := genValues[K](0, n)
	e := genValues[E](0, n)
	m := fillMap(k, e)
	b.ResetTimer()

	for i := 0; i < b.N; i++ {
		m[k[i%n]] += e[i%n]
	}
}

func BenchmarkMapAssignAddition(b *testing.B) {
	b.Run("Key=int32/Elem=int32", benchSizes(benchmarkMapAssignAddition[int32, int32]))
	b.Run("Key=int64/Elem=int64", benchSizes(benchmarkMapAssignAddition[int64, int64]))
	b.Run("Key=string/Elem=string", benchSizes(benchmarkMapAssignAddition[string, string]))
	b.Run("Key=smallType/Elem=int32", benchSizes(benchmarkMapAssignAddition[smallType, int32]))
	b.Run("Key=mediumType/Elem=int32", benchSizes(benchmarkMapAssignAddition[mediumType, int32]))
	b.Run("Key=bigType/Elem=int32", benchSizes(benchmarkMapAssignAddition[bigType, int32]))
}

// Modify values append.
func benchmarkMapAssignAppend[K mapBenchmarkKeyType](b *testing.B, n int) {
	if n == 0 {
		b.Skip("can't modify empty map via append")
	}
	checkAllocSize[K, []int32](b, n)
	k := genValues[K](0, n)
	e := genValues[[]int32](0, n)
	m := fillMap(k, e)
	b.ResetTimer()

	for i := 0; i < b.N; i++ {
		m[k[i%n]] = append(m[k[i%n]], e[i%n][0])
	}
}

func BenchmarkMapAssignAppend(b *testing.B) {
	b.Run("Key=int32/Elem=[]int32", benchSizes(benchmarkMapAssignAppend[int32]))
	b.Run("Key=int64/Elem=[]int32", benchSizes(benchmarkMapAssignAppend[int64]))
	b.Run("Key=string/Elem=[]int32", benchSizes(benchmarkMapAssignAppend[string]))
}

func benchmarkMapDelete[K mapBenchmarkKeyType, E mapBenchmarkElemType](b *testing.B, n int) {
	if n == 0 {
		b.Skip("can't delete from empty map")
	}
	checkAllocSize[K, E](b, n)
	k := genValues[K](0, n)
	e := genValues[E](0, n)
	m := fillMap(k, e)
	b.ResetTimer()

	for i := 0; i < b.N; i++ {
		if len(m) == 0 {
			b.StopTimer()
			for j := range k {
				m[k[j]] = e[j]
			}
			b.StartTimer()
		}
		delete(m, k[i%n])
	}
}

func BenchmarkMapDelete(b *testing.B) {
	b.Run("Key=int32/Elem=int32", benchSizes(benchmarkMapDelete[int32, int32]))
	b.Run("Key=int64/Elem=int64", benchSizes(benchmarkMapDelete[int64, int64]))
	b.Run("Key=string/Elem=string", benchSizes(benchmarkMapDelete[string, string]))
	b.Run("Key=smallType/Elem=int32", benchSizes(benchmarkMapDelete[smallType, int32]))
	b.Run("Key=mediumType/Elem=int32", benchSizes(benchmarkMapDelete[mediumType, int32]))
	b.Run("Key=bigType/Elem=int32", benchSizes(benchmarkMapDelete[bigType, int32]))
	b.Run("Key=bigType/Elem=bigType", benchSizes(benchmarkMapDelete[bigType, bigType]))
	b.Run("Key=int32/Elem=bigType", benchSizes(benchmarkMapDelete[int32, bigType]))
	b.Run("Key=*int32/Elem=int32", benchSizes(benchmarkMapDelete[*int32, int32]))
	b.Run("Key=int32/Elem=*int32", benchSizes(benchmarkMapDelete[int32, *int32]))
}

// Use iterator to pop an element. We want this to be fast, see
// https://go.dev/issue/8412.
func benchmarkMapPop[K mapBenchmarkKeyType, E mapBenchmarkElemType](b *testing.B, n int) {
	if n == 0 {
		b.Skip("can't delete from empty map")
	}
	checkAllocSize[K, E](b, n)
	k := genValues[K](0, n)
	e := genValues[E](0, n)
	m := fillMap(k, e)
	b.ResetTimer()

	for i := 0; i < b.N; i++ {
		if len(m) == 0 {
			// We'd like to StopTimer while refilling the map, but
			// it is way too expensive and thus makes the benchmark
			// take a long time. See https://go.dev/issue/20875.
			for j := range k {
				m[k[j]] = e[j]
			}
		}
		for key := range m {
			delete(m, key)
			break
		}
	}
}

func BenchmarkMapPop(b *testing.B) {
	b.Run("Key=int32/Elem=int32", benchSizes(benchmarkMapPop[int32, int32]))
	b.Run("Key=int64/Elem=int64", benchSizes(benchmarkMapPop[int64, int64]))
	b.Run("Key=string/Elem=string", benchSizes(benchmarkMapPop[string, string]))
	b.Run("Key=smallType/Elem=int32", benchSizes(benchmarkMapPop[smallType, int32]))
	b.Run("Key=mediumType/Elem=int32", benchSizes(benchmarkMapPop[mediumType, int32]))
	b.Run("Key=bigType/Elem=int32", benchSizes(benchmarkMapPop[bigType, int32]))
	b.Run("Key=bigType/Elem=bigType", benchSizes(benchmarkMapPop[bigType, bigType]))
	b.Run("Key=int32/Elem=bigType", benchSizes(benchmarkMapPop[int32, bigType]))
	b.Run("Key=*int32/Elem=int32", benchSizes(benchmarkMapPop[*int32, int32]))
	b.Run("Key=int32/Elem=*int32", benchSizes(benchmarkMapPop[int32, *int32]))
}
