// Copyright 2023 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

package sort_test

import (
	"math/rand/v2"
	"slices"
	. "sort"
	"strconv"
	stringspkg "strings"
	"testing"
)

// Benchmarks comparing sorting from the slices package with functions from
// the sort package (avoiding functions that are just forwarding to the slices
// package).

func makeRandomInts(n int) []int {
	r := rand.New(rand.NewPCG(42, 0))
	ints := make([]int, n)
	for i := 0; i < n; i++ {
		ints[i] = r.IntN(n)
	}
	return ints
}

func makeSortedInts(n int) []int {
	ints := make([]int, n)
	for i := 0; i < n; i++ {
		ints[i] = i
	}
	return ints
}

func makeReversedInts(n int) []int {
	ints := make([]int, n)
	for i := 0; i < n; i++ {
		ints[i] = n - i
	}
	return ints
}

func makeSortedStrings(n int) []string {
	x := make([]string, n)
	for i := 0; i < n; i++ {
		x[i] = strconv.Itoa(i)
	}
	Strings(x)
	return x
}

const N = 100_000

func BenchmarkSortInts(b *testing.B) {
	for i := 0; i < b.N; i++ {
		b.StopTimer()
		ints := makeRandomInts(N)
		b.StartTimer()
		Sort(IntSlice(ints))
	}
}

func BenchmarkSlicesSortInts(b *testing.B) {
	for i := 0; i < b.N; i++ {
		b.StopTimer()
		ints := makeRandomInts(N)
		b.StartTimer()
		slices.Sort(ints)
	}
}

func BenchmarkSortIsSorted(b *testing.B) {
	for i := 0; i < b.N; i++ {
		b.StopTimer()
		ints := makeSortedInts(N)
		b.StartTimer()
		IsSorted(IntSlice(ints))
	}
}

func BenchmarkSlicesIsSorted(b *testing.B) {
	for i := 0; i < b.N; i++ {
		b.StopTimer()
		ints := makeSortedInts(N)
		b.StartTimer()
		slices.IsSorted(ints)
	}
}

// makeRandomStrings generates n random strings with alphabetic runes of
// varying lengths.
func makeRandomStrings(n int) []string {
	r := rand.New(rand.NewPCG(42, 0))
	var letters = []rune("abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ")
	ss := make([]string, n)
	for i := 0; i < n; i++ {
		var sb stringspkg.Builder
		slen := 2 + r.IntN(50)
		for j := 0; j < slen; j++ {
			sb.WriteRune(letters[r.IntN(len(letters))])
		}
		ss[i] = sb.String()
	}
	return ss
}

func BenchmarkSortStrings(b *testing.B) {
	for i := 0; i < b.N; i++ {
		b.StopTimer()
		ss := makeRandomStrings(N)
		b.StartTimer()
		Sort(StringSlice(ss))
	}
}

func BenchmarkSlicesSortStrings(b *testing.B) {
	for i := 0; i < b.N; i++ {
		b.StopTimer()
		ss := makeRandomStrings(N)
		b.StartTimer()
		slices.Sort(ss)
	}
}

func BenchmarkSortStrings_Sorted(b *testing.B) {
	ss := makeSortedStrings(N)
	b.ResetTimer()

	for i := 0; i < b.N; i++ {
		Sort(StringSlice(ss))
	}
}

func BenchmarkSlicesSortStrings_Sorted(b *testing.B) {
	ss := makeSortedStrings(N)
	b.ResetTimer()

	for i := 0; i < b.N; i++ {
		slices.Sort(ss)
	}
}

// These benchmarks compare sorting a slice of structs with sort.Sort vs.
// slices.SortFunc.
type myStruct struct {
	a, b, c, d string
	n          int
}

type myStructs []*myStruct

func (s myStructs) Len() int           { return len(s) }
func (s myStructs) Less(i, j int) bool { return s[i].n < s[j].n }
func (s myStructs) Swap(i, j int)      { s[i], s[j] = s[j], s[i] }

func makeRandomStructs(n int) myStructs {
	r := rand.New(rand.NewPCG(42, 0))
	structs := make([]*myStruct, n)
	for i := 0; i < n; i++ {
		structs[i] = &myStruct{n: r.IntN(n)}
	}
	return structs
}

func TestStructSorts(t *testing.T) {
	ss := makeRandomStructs(200)
	ss2 := make([]*myStruct, len(ss))
	for i := range ss {
		ss2[i] = &myStruct{n: ss[i].n}
	}

	Sort(ss)
	slices.SortFunc(ss2, func { a, b -> a.n - b.n })

	for i := range ss {
		if *ss[i] != *ss2[i] {
			t.Fatalf("ints2 mismatch at %d; %v != %v", i, *ss[i], *ss2[i])
		}
	}
}

func BenchmarkSortStructs(b *testing.B) {
	for i := 0; i < b.N; i++ {
		b.StopTimer()
		ss := makeRandomStructs(N)
		b.StartTimer()
		Sort(ss)
	}
}

func BenchmarkSortFuncStructs(b *testing.B) {
	cmpFunc := func(a, b *myStruct) int { return a.n - b.n }
	for i := 0; i < b.N; i++ {
		b.StopTimer()
		ss := makeRandomStructs(N)
		b.StartTimer()
		slices.SortFunc(ss, cmpFunc)
	}
}
