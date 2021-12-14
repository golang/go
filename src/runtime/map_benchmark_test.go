// Copyright 2013 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

package runtime_test

import (
	"fmt"
	"math/rand"
	"strconv"
	"strings"
	"testing"
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

func BenchmarkIntMap(b *testing.B) {
	m := make(map[int]bool)
	for i := 0; i < 8; i++ {
		m[i] = true
	}
	b.ResetTimer()
	for i := 0; i < b.N; i++ {
		_, _ = m[7]
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

func BenchmarkMapIter(b *testing.B) {
	m := make(map[int]bool)
	for i := 0; i < 8; i++ {
		m[i] = true
	}
	b.ResetTimer()
	for i := 0; i < b.N; i++ {
		for range m {
		}
	}
}

func BenchmarkMapIterEmpty(b *testing.B) {
	m := make(map[int]bool)
	b.ResetTimer()
	for i := 0; i < b.N; i++ {
		for range m {
		}
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

type BigKey [3]int64

func BenchmarkBigKeyMap(b *testing.B) {
	m := make(map[BigKey]bool)
	k := BigKey{3, 4, 5}
	m[k] = true
	for i := 0; i < b.N; i++ {
		_ = m[k]
	}
}

type BigVal [3]int64

func BenchmarkBigValMap(b *testing.B) {
	m := make(map[BigKey]BigVal)
	k := BigKey{3, 4, 5}
	m[k] = BigVal{6, 7, 8}
	for i := 0; i < b.N; i++ {
		_ = m[k]
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
					for k := range m {
						delete(m, k)
					}
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
					for k := range m {
						delete(m, k)
					}
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
	m := map[interface{}]bool{}

	for i := 0; i < 100; i++ {
		m[fmt.Sprintf("%d", i)] = true
	}

	key := (interface{})("A")
	b.ResetTimer()
	for i := 0; i < b.N; i++ {
		BoolSink = m[key]
	}
}
func BenchmarkMapInterfacePtr(b *testing.B) {
	m := map[interface{}]bool{}

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
