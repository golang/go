// Copyright 2009 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

package sort

import (
	"fmt"
	"rand"
	"strconv"
	"testing"
)


var ints = [...]int{74, 59, 238, -784, 9845, 959, 905, 0, 0, 42, 7586, -5467984, 7586}
var float64s = [...]float64{74.3, 59.0, 238.2, -784.0, 2.3, 9845.768, -959.7485, 905, 7.8, 7.8}
var strings = [...]string{"", "Hello", "foo", "bar", "foo", "f00", "%*&^*&^&", "***"}

func TestSortIntArray(t *testing.T) {
	data := ints
	a := IntArray(data[0:])
	Sort(a)
	if !IsSorted(a) {
		t.Errorf("sorted %v", ints)
		t.Errorf("   got %v", data)
	}
}

func TestSortFloat64Array(t *testing.T) {
	data := float64s
	a := Float64Array(data[0:])
	Sort(a)
	if !IsSorted(a) {
		t.Errorf("sorted %v", float64s)
		t.Errorf("   got %v", data)
	}
}

func TestSortStringArray(t *testing.T) {
	data := strings
	a := StringArray(data[0:])
	Sort(a)
	if !IsSorted(a) {
		t.Errorf("sorted %v", strings)
		t.Errorf("   got %v", data)
	}
}

func TestSortInts(t *testing.T) {
	data := ints
	SortInts(data[0:])
	if !IntsAreSorted(data[0:]) {
		t.Errorf("sorted %v", ints)
		t.Errorf("   got %v", data)
	}
}

func TestSortFloat64s(t *testing.T) {
	data := float64s
	SortFloat64s(data[0:])
	if !Float64sAreSorted(data[0:]) {
		t.Errorf("sorted %v", float64s)
		t.Errorf("   got %v", data)
	}
}

func TestSortStrings(t *testing.T) {
	data := strings
	SortStrings(data[0:])
	if !StringsAreSorted(data[0:]) {
		t.Errorf("sorted %v", strings)
		t.Errorf("   got %v", data)
	}
}

func TestSortLarge_Random(t *testing.T) {
	data := make([]int, 1000000)
	for i := 0; i < len(data); i++ {
		data[i] = rand.Intn(100)
	}
	if IntsAreSorted(data) {
		t.Fatalf("terrible rand.rand")
	}
	SortInts(data)
	if !IntsAreSorted(data) {
		t.Errorf("sort didn't sort - 1M ints")
	}
}

func BenchmarkSortString1K(b *testing.B) {
	b.StopTimer()
	for i := 0; i < b.N; i++ {
		data := make([]string, 1<<10)
		for i := 0; i < len(data); i++ {
			data[i] = strconv.Itoa(i ^ 0x2cc)
		}
		b.StartTimer()
		SortStrings(data)
		b.StopTimer()
	}
}

func BenchmarkSortInt1K(b *testing.B) {
	b.StopTimer()
	for i := 0; i < b.N; i++ {
		data := make([]int, 1<<10)
		for i := 0; i < len(data); i++ {
			data[i] = i ^ 0x2cc
		}
		b.StartTimer()
		SortInts(data)
		b.StopTimer()
	}
}

func BenchmarkSortInt64K(b *testing.B) {
	b.StopTimer()
	for i := 0; i < b.N; i++ {
		data := make([]int, 1<<16)
		for i := 0; i < len(data); i++ {
			data[i] = i ^ 0xcccc
		}
		b.StartTimer()
		SortInts(data)
		b.StopTimer()
	}
}

const (
	_Sawtooth = iota
	_Rand
	_Stagger
	_Plateau
	_Shuffle
	_NDist
)

const (
	_Copy = iota
	_Reverse
	_ReverseFirstHalf
	_ReverseSecondHalf
	_Sorted
	_Dither
	_NMode
)

type testingData struct {
	desc    string
	t       *testing.T
	data    []int
	maxswap int // number of swaps allowed
	nswap   int
}

func (d *testingData) Len() int           { return len(d.data) }
func (d *testingData) Less(i, j int) bool { return d.data[i] < d.data[j] }
func (d *testingData) Swap(i, j int) {
	if d.nswap >= d.maxswap {
		d.t.Errorf("%s: used %d swaps sorting array of %d", d.desc, d.nswap, len(d.data))
		d.t.FailNow()
	}
	d.nswap++
	d.data[i], d.data[j] = d.data[j], d.data[i]
}

func lg(n int) int {
	i := 0
	for 1<<uint(i) < n {
		i++
	}
	return i
}

func TestBentleyMcIlroy(t *testing.T) {
	sizes := []int{100, 1023, 1024, 1025}
	dists := []string{"sawtooth", "rand", "stagger", "plateau", "shuffle"}
	modes := []string{"copy", "reverse", "reverse1", "reverse2", "sort", "dither"}
	var tmp1, tmp2 [1025]int
	for ni := 0; ni < len(sizes); ni++ {
		n := sizes[ni]
		for m := 1; m < 2*n; m *= 2 {
			for dist := 0; dist < _NDist; dist++ {
				j := 0
				k := 1
				data := tmp1[0:n]
				for i := 0; i < n; i++ {
					switch dist {
					case _Sawtooth:
						data[i] = i % m
					case _Rand:
						data[i] = rand.Intn(m)
					case _Stagger:
						data[i] = (i*m + i) % n
					case _Plateau:
						data[i] = min(i, m)
					case _Shuffle:
						if rand.Intn(m) != 0 {
							j += 2
							data[i] = j
						} else {
							k += 2
							data[i] = k
						}
					}
				}

				mdata := tmp2[0:n]
				for mode := 0; mode < _NMode; mode++ {
					switch mode {
					case _Copy:
						for i := 0; i < n; i++ {
							mdata[i] = data[i]
						}
					case _Reverse:
						for i := 0; i < n; i++ {
							mdata[i] = data[n-i-1]
						}
					case _ReverseFirstHalf:
						for i := 0; i < n/2; i++ {
							mdata[i] = data[n/2-i-1]
						}
						for i := n / 2; i < n; i++ {
							mdata[i] = data[i]
						}
					case _ReverseSecondHalf:
						for i := 0; i < n/2; i++ {
							mdata[i] = data[i]
						}
						for i := n / 2; i < n; i++ {
							mdata[i] = data[n-(i-n/2)-1]
						}
					case _Sorted:
						for i := 0; i < n; i++ {
							mdata[i] = data[i]
						}
						// SortInts is known to be correct
						// because mode Sort runs after mode _Copy.
						SortInts(mdata)
					case _Dither:
						for i := 0; i < n; i++ {
							mdata[i] = data[i] + i%5
						}
					}

					desc := fmt.Sprintf("n=%d m=%d dist=%s mode=%s", n, m, dists[dist], modes[mode])
					d := &testingData{desc, t, mdata[0:n], n * lg(n) * 12 / 10, 0}
					Sort(d)

					// If we were testing C qsort, we'd have to make a copy
					// of the array and sort it ourselves and then compare
					// x against it, to ensure that qsort was only permuting
					// the data, not (for example) overwriting it with zeros.
					//
					// In go, we don't have to be so paranoid: since the only
					// mutating method Sort can call is TestingData.swap,
					// it suffices here just to check that the final array is sorted.
					if !IntsAreSorted(mdata) {
						t.Errorf("%s: ints not sorted", desc)
						t.Errorf("\t%v", mdata)
						t.FailNow()
					}
				}
			}
		}
	}
}
