// Copyright 2021 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

package slices_test

import (
	"cmp"
	"internal/race"
	"internal/testenv"
	"maps"
	"math"
	. "slices"
	"strings"
	"testing"
)

var equalIntTests = []struct {
	s1, s2 []int
	want   bool
}{
	{
		[]int{1},
		nil,
		false,
	},
	{
		[]int{},
		nil,
		true,
	},
	{
		[]int{1, 2, 3},
		[]int{1, 2, 3},
		true,
	},
	{
		[]int{1, 2, 3},
		[]int{1, 2, 3, 4},
		false,
	},
}

var equalFloatTests = []struct {
	s1, s2       []float64
	wantEqual    bool
	wantEqualNaN bool
}{
	{
		[]float64{1, 2},
		[]float64{1, 2},
		true,
		true,
	},
	{
		[]float64{1, 2, math.NaN()},
		[]float64{1, 2, math.NaN()},
		false,
		true,
	},
}

func TestEqual(t *testing.T) {
	for _, test := range equalIntTests {
		if got := Equal(test.s1, test.s2); got != test.want {
			t.Errorf("Equal(%v, %v) = %t, want %t", test.s1, test.s2, got, test.want)
		}
	}
	for _, test := range equalFloatTests {
		if got := Equal(test.s1, test.s2); got != test.wantEqual {
			t.Errorf("Equal(%v, %v) = %t, want %t", test.s1, test.s2, got, test.wantEqual)
		}
	}
}

// equal is simply ==.
func equal[T comparable](v1, v2 T) bool {
	return v1 == v2
}

// equalNaN is like == except that all NaNs are equal.
func equalNaN[T comparable](v1, v2 T) bool {
	isNaN := func(f T) bool { return f != f }
	return v1 == v2 || (isNaN(v1) && isNaN(v2))
}

// offByOne returns true if integers v1 and v2 differ by 1.
func offByOne(v1, v2 int) bool {
	return v1 == v2+1 || v1 == v2-1
}

func TestEqualFunc(t *testing.T) {
	for _, test := range equalIntTests {
		if got := EqualFunc(test.s1, test.s2, equal[int]); got != test.want {
			t.Errorf("EqualFunc(%v, %v, equal[int]) = %t, want %t", test.s1, test.s2, got, test.want)
		}
	}
	for _, test := range equalFloatTests {
		if got := EqualFunc(test.s1, test.s2, equal[float64]); got != test.wantEqual {
			t.Errorf("Equal(%v, %v, equal[float64]) = %t, want %t", test.s1, test.s2, got, test.wantEqual)
		}
		if got := EqualFunc(test.s1, test.s2, equalNaN[float64]); got != test.wantEqualNaN {
			t.Errorf("Equal(%v, %v, equalNaN[float64]) = %t, want %t", test.s1, test.s2, got, test.wantEqualNaN)
		}
	}

	s1 := []int{1, 2, 3}
	s2 := []int{2, 3, 4}
	if EqualFunc(s1, s1, offByOne) {
		t.Errorf("EqualFunc(%v, %v, offByOne) = true, want false", s1, s1)
	}
	if !EqualFunc(s1, s2, offByOne) {
		t.Errorf("EqualFunc(%v, %v, offByOne) = false, want true", s1, s2)
	}

	s3 := []string{"a", "b", "c"}
	s4 := []string{"A", "B", "C"}
	if !EqualFunc(s3, s4, strings.EqualFold) {
		t.Errorf("EqualFunc(%v, %v, strings.EqualFold) = false, want true", s3, s4)
	}

	cmpIntString := func(v1 int, v2 string) bool {
		return string(rune(v1)-1+'a') == v2
	}
	if !EqualFunc(s1, s3, cmpIntString) {
		t.Errorf("EqualFunc(%v, %v, cmpIntString) = false, want true", s1, s3)
	}
}

func BenchmarkEqualFunc_Large(b *testing.B) {
	type Large [4 * 1024]byte

	xs := make([]Large, 1024)
	ys := make([]Large, 1024)
	for i := 0; i < b.N; i++ {
		_ = EqualFunc(xs, ys, func(x, y Large) bool { return x == y })
	}
}

var compareIntTests = []struct {
	s1, s2 []int
	want   int
}{
	{
		[]int{1},
		[]int{1},
		0,
	},
	{
		[]int{1},
		[]int{},
		1,
	},
	{
		[]int{},
		[]int{1},
		-1,
	},
	{
		[]int{},
		[]int{},
		0,
	},
	{
		[]int{1, 2, 3},
		[]int{1, 2, 3},
		0,
	},
	{
		[]int{1, 2, 3},
		[]int{1, 2, 3, 4},
		-1,
	},
	{
		[]int{1, 2, 3, 4},
		[]int{1, 2, 3},
		+1,
	},
	{
		[]int{1, 2, 3},
		[]int{1, 4, 3},
		-1,
	},
	{
		[]int{1, 4, 3},
		[]int{1, 2, 3},
		+1,
	},
	{
		[]int{1, 4, 3},
		[]int{1, 2, 3, 8, 9},
		+1,
	},
}

var compareFloatTests = []struct {
	s1, s2 []float64
	want   int
}{
	{
		[]float64{},
		[]float64{},
		0,
	},
	{
		[]float64{1},
		[]float64{1},
		0,
	},
	{
		[]float64{math.NaN()},
		[]float64{math.NaN()},
		0,
	},
	{
		[]float64{1, 2, math.NaN()},
		[]float64{1, 2, math.NaN()},
		0,
	},
	{
		[]float64{1, math.NaN(), 3},
		[]float64{1, math.NaN(), 4},
		-1,
	},
	{
		[]float64{1, math.NaN(), 3},
		[]float64{1, 2, 4},
		-1,
	},
	{
		[]float64{1, math.NaN(), 3},
		[]float64{1, 2, math.NaN()},
		-1,
	},
	{
		[]float64{1, 2, 3},
		[]float64{1, 2, math.NaN()},
		+1,
	},
	{
		[]float64{1, 2, 3},
		[]float64{1, math.NaN(), 3},
		+1,
	},
	{
		[]float64{1, math.NaN(), 3, 4},
		[]float64{1, 2, math.NaN()},
		-1,
	},
}

func TestCompare(t *testing.T) {
	intWant := func(want bool) string {
		if want {
			return "0"
		}
		return "!= 0"
	}
	for _, test := range equalIntTests {
		if got := Compare(test.s1, test.s2); (got == 0) != test.want {
			t.Errorf("Compare(%v, %v) = %d, want %s", test.s1, test.s2, got, intWant(test.want))
		}
	}
	for _, test := range equalFloatTests {
		if got := Compare(test.s1, test.s2); (got == 0) != test.wantEqualNaN {
			t.Errorf("Compare(%v, %v) = %d, want %s", test.s1, test.s2, got, intWant(test.wantEqualNaN))
		}
	}

	for _, test := range compareIntTests {
		if got := Compare(test.s1, test.s2); got != test.want {
			t.Errorf("Compare(%v, %v) = %d, want %d", test.s1, test.s2, got, test.want)
		}
	}
	for _, test := range compareFloatTests {
		if got := Compare(test.s1, test.s2); got != test.want {
			t.Errorf("Compare(%v, %v) = %d, want %d", test.s1, test.s2, got, test.want)
		}
	}
}

func equalToCmp[T comparable](eq func(T, T) bool) func(T, T) int {
	return func(v1, v2 T) int {
		if eq(v1, v2) {
			return 0
		}
		return 1
	}
}

func TestCompareFunc(t *testing.T) {
	intWant := func(want bool) string {
		if want {
			return "0"
		}
		return "!= 0"
	}
	for _, test := range equalIntTests {
		if got := CompareFunc(test.s1, test.s2, equalToCmp(equal[int])); (got == 0) != test.want {
			t.Errorf("CompareFunc(%v, %v, equalToCmp(equal[int])) = %d, want %s", test.s1, test.s2, got, intWant(test.want))
		}
	}
	for _, test := range equalFloatTests {
		if got := CompareFunc(test.s1, test.s2, equalToCmp(equal[float64])); (got == 0) != test.wantEqual {
			t.Errorf("CompareFunc(%v, %v, equalToCmp(equal[float64])) = %d, want %s", test.s1, test.s2, got, intWant(test.wantEqual))
		}
	}

	for _, test := range compareIntTests {
		if got := CompareFunc(test.s1, test.s2, cmp.Compare[int]); got != test.want {
			t.Errorf("CompareFunc(%v, %v, cmp[int]) = %d, want %d", test.s1, test.s2, got, test.want)
		}
	}
	for _, test := range compareFloatTests {
		if got := CompareFunc(test.s1, test.s2, cmp.Compare[float64]); got != test.want {
			t.Errorf("CompareFunc(%v, %v, cmp[float64]) = %d, want %d", test.s1, test.s2, got, test.want)
		}
	}

	s1 := []int{1, 2, 3}
	s2 := []int{2, 3, 4}
	if got := CompareFunc(s1, s2, equalToCmp(offByOne)); got != 0 {
		t.Errorf("CompareFunc(%v, %v, offByOne) = %d, want 0", s1, s2, got)
	}

	s3 := []string{"a", "b", "c"}
	s4 := []string{"A", "B", "C"}
	if got := CompareFunc(s3, s4, strings.Compare); got != 1 {
		t.Errorf("CompareFunc(%v, %v, strings.Compare) = %d, want 1", s3, s4, got)
	}

	compareLower := func(v1, v2 string) int {
		return strings.Compare(strings.ToLower(v1), strings.ToLower(v2))
	}
	if got := CompareFunc(s3, s4, compareLower); got != 0 {
		t.Errorf("CompareFunc(%v, %v, compareLower) = %d, want 0", s3, s4, got)
	}

	cmpIntString := func(v1 int, v2 string) int {
		return strings.Compare(string(rune(v1)-1+'a'), v2)
	}
	if got := CompareFunc(s1, s3, cmpIntString); got != 0 {
		t.Errorf("CompareFunc(%v, %v, cmpIntString) = %d, want 0", s1, s3, got)
	}
}

var indexTests = []struct {
	s    []int
	v    int
	want int
}{
	{
		nil,
		0,
		-1,
	},
	{
		[]int{},
		0,
		-1,
	},
	{
		[]int{1, 2, 3},
		2,
		1,
	},
	{
		[]int{1, 2, 2, 3},
		2,
		1,
	},
	{
		[]int{1, 2, 3, 2},
		2,
		1,
	},
}

func TestIndex(t *testing.T) {
	for _, test := range indexTests {
		if got := Index(test.s, test.v); got != test.want {
			t.Errorf("Index(%v, %v) = %d, want %d", test.s, test.v, got, test.want)
		}
	}
}

func equalToIndex[T any](f func(T, T) bool, v1 T) func(T) bool {
	return func(v2 T) bool {
		return f(v1, v2)
	}
}

func BenchmarkIndex_Large(b *testing.B) {
	type Large [4 * 1024]byte

	ss := make([]Large, 1024)
	for i := 0; i < b.N; i++ {
		_ = Index(ss, Large{1})
	}
}

func TestIndexFunc(t *testing.T) {
	for _, test := range indexTests {
		if got := IndexFunc(test.s, equalToIndex(equal[int], test.v)); got != test.want {
			t.Errorf("IndexFunc(%v, equalToIndex(equal[int], %v)) = %d, want %d", test.s, test.v, got, test.want)
		}
	}

	s1 := []string{"hi", "HI"}
	if got := IndexFunc(s1, equalToIndex(equal[string], "HI")); got != 1 {
		t.Errorf("IndexFunc(%v, equalToIndex(equal[string], %q)) = %d, want %d", s1, "HI", got, 1)
	}
	if got := IndexFunc(s1, equalToIndex(strings.EqualFold, "HI")); got != 0 {
		t.Errorf("IndexFunc(%v, equalToIndex(strings.EqualFold, %q)) = %d, want %d", s1, "HI", got, 0)
	}
}

func BenchmarkIndexFunc_Large(b *testing.B) {
	type Large [4 * 1024]byte

	ss := make([]Large, 1024)
	for i := 0; i < b.N; i++ {
		_ = IndexFunc(ss, func(e Large) bool {
			return e == Large{1}
		})
	}
}

func TestContains(t *testing.T) {
	for _, test := range indexTests {
		if got := Contains(test.s, test.v); got != (test.want != -1) {
			t.Errorf("Contains(%v, %v) = %t, want %t", test.s, test.v, got, test.want != -1)
		}
	}
}

func TestContainsFunc(t *testing.T) {
	for _, test := range indexTests {
		if got := ContainsFunc(test.s, equalToIndex(equal[int], test.v)); got != (test.want != -1) {
			t.Errorf("ContainsFunc(%v, equalToIndex(equal[int], %v)) = %t, want %t", test.s, test.v, got, test.want != -1)
		}
	}

	s1 := []string{"hi", "HI"}
	if got := ContainsFunc(s1, equalToIndex(equal[string], "HI")); got != true {
		t.Errorf("ContainsFunc(%v, equalToContains(equal[string], %q)) = %t, want %t", s1, "HI", got, true)
	}
	if got := ContainsFunc(s1, equalToIndex(equal[string], "hI")); got != false {
		t.Errorf("ContainsFunc(%v, equalToContains(strings.EqualFold, %q)) = %t, want %t", s1, "hI", got, false)
	}
	if got := ContainsFunc(s1, equalToIndex(strings.EqualFold, "hI")); got != true {
		t.Errorf("ContainsFunc(%v, equalToContains(strings.EqualFold, %q)) = %t, want %t", s1, "hI", got, true)
	}
}

var insertTests = []struct {
	s    []int
	i    int
	add  []int
	want []int
}{
	{
		[]int{1, 2, 3},
		0,
		[]int{4},
		[]int{4, 1, 2, 3},
	},
	{
		[]int{1, 2, 3},
		1,
		[]int{4},
		[]int{1, 4, 2, 3},
	},
	{
		[]int{1, 2, 3},
		3,
		[]int{4},
		[]int{1, 2, 3, 4},
	},
	{
		[]int{1, 2, 3},
		2,
		[]int{4, 5},
		[]int{1, 2, 4, 5, 3},
	},
}

func TestInsert(t *testing.T) {
	s := []int{1, 2, 3}
	if got := Insert(s, 0); !Equal(got, s) {
		t.Errorf("Insert(%v, 0) = %v, want %v", s, got, s)
	}
	for _, test := range insertTests {
		copy := Clone(test.s)
		if got := Insert(copy, test.i, test.add...); !Equal(got, test.want) {
			t.Errorf("Insert(%v, %d, %v...) = %v, want %v", test.s, test.i, test.add, got, test.want)
		}
	}

	if !testenv.OptimizationOff() && !race.Enabled {
		// Allocations should be amortized.
		const count = 50
		n := testing.AllocsPerRun(10, func() {
			s := []int{1, 2, 3}
			for i := 0; i < count; i++ {
				s = Insert(s, 0, 1)
			}
		})
		if n > count/2 {
			t.Errorf("too many allocations inserting %d elements: got %v, want less than %d", count, n, count/2)
		}
	}
}

func TestInsertOverlap(t *testing.T) {
	const N = 10
	a := make([]int, N)
	want := make([]int, 2*N)
	for n := 0; n <= N; n++ { // length
		for i := 0; i <= n; i++ { // insertion point
			for x := 0; x <= N; x++ { // start of inserted data
				for y := x; y <= N; y++ { // end of inserted data
					for k := 0; k < N; k++ {
						a[k] = k
					}
					want = want[:0]
					want = append(want, a[:i]...)
					want = append(want, a[x:y]...)
					want = append(want, a[i:n]...)
					got := Insert(a[:n], i, a[x:y]...)
					if !Equal(got, want) {
						t.Errorf("Insert with overlap failed n=%d i=%d x=%d y=%d, got %v want %v", n, i, x, y, got, want)
					}
				}
			}
		}
	}
}

func TestInsertPanics(t *testing.T) {
	a := [3]int{}
	b := [1]int{}
	for _, test := range []struct {
		name string
		s    []int
		i    int
		v    []int
	}{
		// There are no values.
		{"with negative index", a[:1:1], -1, nil},
		{"with out-of-bounds index and > cap", a[:1:1], 2, nil},
		{"with out-of-bounds index and = cap", a[:1:2], 2, nil},
		{"with out-of-bounds index and < cap", a[:1:3], 2, nil},

		// There are values.
		{"with negative index", a[:1:1], -1, b[:]},
		{"with out-of-bounds index and > cap", a[:1:1], 2, b[:]},
		{"with out-of-bounds index and = cap", a[:1:2], 2, b[:]},
		{"with out-of-bounds index and < cap", a[:1:3], 2, b[:]},
	} {
		if !panics(func() { _ = Insert(test.s, test.i, test.v...) }) {
			t.Errorf("Insert %s: got no panic, want panic", test.name)
		}
	}
}

var deleteTests = []struct {
	s    []int
	i, j int
	want []int
}{
	{
		[]int{1, 2, 3},
		0,
		0,
		[]int{1, 2, 3},
	},
	{
		[]int{1, 2, 3},
		0,
		1,
		[]int{2, 3},
	},
	{
		[]int{1, 2, 3},
		3,
		3,
		[]int{1, 2, 3},
	},
	{
		[]int{1, 2, 3},
		0,
		2,
		[]int{3},
	},
	{
		[]int{1, 2, 3},
		0,
		3,
		[]int{},
	},
}

func TestDelete(t *testing.T) {
	for _, test := range deleteTests {
		copy := Clone(test.s)
		if got := Delete(copy, test.i, test.j); !Equal(got, test.want) {
			t.Errorf("Delete(%v, %d, %d) = %v, want %v", test.s, test.i, test.j, got, test.want)
		}
	}
}

var deleteFuncTests = []struct {
	s    []int
	fn   func(int) bool
	want []int
}{
	{
		nil,
		func(int) bool { return true },
		nil,
	},
	{
		[]int{1, 2, 3},
		func(int) bool { return true },
		nil,
	},
	{
		[]int{1, 2, 3},
		func(int) bool { return false },
		[]int{1, 2, 3},
	},
	{
		[]int{1, 2, 3},
		func(i int) bool { return i > 2 },
		[]int{1, 2},
	},
	{
		[]int{1, 2, 3},
		func(i int) bool { return i < 2 },
		[]int{2, 3},
	},
	{
		[]int{10, 2, 30},
		func(i int) bool { return i >= 10 },
		[]int{2},
	},
}

func TestDeleteFunc(t *testing.T) {
	for i, test := range deleteFuncTests {
		copy := Clone(test.s)
		if got := DeleteFunc(copy, test.fn); !Equal(got, test.want) {
			t.Errorf("DeleteFunc case %d: got %v, want %v", i, got, test.want)
		}
	}
}

func panics(f func()) (b bool) {
	defer func() {
		if x := recover(); x != nil {
			b = true
		}
	}()
	f()
	return false
}

func TestDeletePanics(t *testing.T) {
	s := []int{0, 1, 2, 3, 4}
	s = s[0:2]
	_ = s[0:4] // this is a valid slice of s

	for _, test := range []struct {
		name string
		s    []int
		i, j int
	}{
		{"with negative first index", []int{42}, -2, 1},
		{"with negative second index", []int{42}, 1, -1},
		{"with out-of-bounds first index", []int{42}, 2, 3},
		{"with out-of-bounds second index", []int{42}, 0, 2},
		{"with out-of-bounds both indexes", []int{42}, 2, 2},
		{"with invalid i>j", []int{42}, 1, 0},
		{"s[i:j] is valid and j > len(s)", s, 0, 4},
		{"s[i:j] is valid and i == j > len(s)", s, 3, 3},
	} {
		if !panics(func() { _ = Delete(test.s, test.i, test.j) }) {
			t.Errorf("Delete %s: got no panic, want panic", test.name)
		}
	}
}

func TestDeleteClearTail(t *testing.T) {
	mem := []*int{new(int), new(int), new(int), new(int), new(int), new(int)}
	s := mem[0:5] // there is 1 element beyond len(s), within cap(s)

	s = Delete(s, 2, 4)

	if mem[3] != nil || mem[4] != nil {
		// Check that potential memory leak is avoided
		t.Errorf("Delete: want nil discarded elements, got %v, %v", mem[3], mem[4])
	}
	if mem[5] == nil {
		t.Errorf("Delete: want unchanged elements beyond original len, got nil")
	}
}

func TestDeleteFuncClearTail(t *testing.T) {
	mem := []*int{new(int), new(int), new(int), new(int), new(int), new(int)}
	*mem[2], *mem[3] = 42, 42
	s := mem[0:5] // there is 1 element beyond len(s), within cap(s)

	s = DeleteFunc(s, func(i *int) bool {
		return i != nil && *i == 42
	})

	if mem[3] != nil || mem[4] != nil {
		// Check that potential memory leak is avoided
		t.Errorf("DeleteFunc: want nil discarded elements, got %v, %v", mem[3], mem[4])
	}
	if mem[5] == nil {
		t.Errorf("DeleteFunc: want unchanged elements beyond original len, got nil")
	}
}

func TestClone(t *testing.T) {
	s1 := []int{1, 2, 3}
	s2 := Clone(s1)
	if !Equal(s1, s2) {
		t.Errorf("Clone(%v) = %v, want %v", s1, s2, s1)
	}
	s1[0] = 4
	want := []int{1, 2, 3}
	if !Equal(s2, want) {
		t.Errorf("Clone(%v) changed unexpectedly to %v", want, s2)
	}
	if got := Clone([]int(nil)); got != nil {
		t.Errorf("Clone(nil) = %#v, want nil", got)
	}
	if got := Clone(s1[:0]); got == nil || len(got) != 0 {
		t.Errorf("Clone(%v) = %#v, want %#v", s1[:0], got, s1[:0])
	}
}

var compactTests = []struct {
	name string
	s    []int
	want []int
}{
	{
		"nil",
		nil,
		nil,
	},
	{
		"one",
		[]int{1},
		[]int{1},
	},
	{
		"sorted",
		[]int{1, 2, 3},
		[]int{1, 2, 3},
	},
	{
		"2 items",
		[]int{1, 1, 2},
		[]int{1, 2},
	},
	{
		"unsorted",
		[]int{1, 2, 1},
		[]int{1, 2, 1},
	},
	{
		"many",
		[]int{1, 2, 2, 3, 3, 4},
		[]int{1, 2, 3, 4},
	},
}

func TestCompact(t *testing.T) {
	for _, test := range compactTests {
		copy := Clone(test.s)
		if got := Compact(copy); !Equal(got, test.want) {
			t.Errorf("Compact(%v) = %v, want %v", test.s, got, test.want)
		}
	}
}

func BenchmarkCompact(b *testing.B) {
	for _, c := range compactTests {
		b.Run(c.name, func(b *testing.B) {
			ss := make([]int, 0, 64)
			for k := 0; k < b.N; k++ {
				ss = ss[:0]
				ss = append(ss, c.s...)
				_ = Compact(ss)
			}
		})
	}
}

func BenchmarkCompact_Large(b *testing.B) {
	type Large [16]int
	const N = 1024

	b.Run("all_dup", func(b *testing.B) {
		ss := make([]Large, N)
		b.ResetTimer()
		for i := 0; i < b.N; i++ {
			_ = Compact(ss)
		}
	})
	b.Run("no_dup", func(b *testing.B) {
		ss := make([]Large, N)
		for i := range ss {
			ss[i][0] = i
		}
		b.ResetTimer()
		for i := 0; i < b.N; i++ {
			_ = Compact(ss)
		}
	})
}

func TestCompactFunc(t *testing.T) {
	for _, test := range compactTests {
		copy := Clone(test.s)
		if got := CompactFunc(copy, equal[int]); !Equal(got, test.want) {
			t.Errorf("CompactFunc(%v, equal[int]) = %v, want %v", test.s, got, test.want)
		}
	}

	s1 := []string{"a", "a", "A", "B", "b"}
	copy := Clone(s1)
	want := []string{"a", "B"}
	if got := CompactFunc(copy, strings.EqualFold); !Equal(got, want) {
		t.Errorf("CompactFunc(%v, strings.EqualFold) = %v, want %v", s1, got, want)
	}
}

func TestCompactClearTail(t *testing.T) {
	one, two, three, four := 1, 2, 3, 4
	mem := []*int{&one, &one, &two, &two, &three, &four}
	s := mem[0:5] // there is 1 element beyond len(s), within cap(s)
	copy := Clone(s)

	s = Compact(s)

	if want := []*int{&one, &two, &three}; !Equal(s, want) {
		t.Errorf("Compact(%v) = %v, want %v", copy, s, want)
	}

	if mem[3] != nil || mem[4] != nil {
		// Check that potential memory leak is avoided
		t.Errorf("Compact: want nil discarded elements, got %v, %v", mem[3], mem[4])
	}
	if mem[5] != &four {
		t.Errorf("Compact: want unchanged element beyond original len, got %v", mem[5])
	}
}

func TestCompactFuncClearTail(t *testing.T) {
	a, b, c, d, e, f := 1, 1, 2, 2, 3, 4
	mem := []*int{&a, &b, &c, &d, &e, &f}
	s := mem[0:5] // there is 1 element beyond len(s), within cap(s)
	copy := Clone(s)

	s = CompactFunc(s, func(x, y *int) bool {
		if x == nil || y == nil {
			return x == y
		}
		return *x == *y
	})

	if want := []*int{&a, &c, &e}; !Equal(s, want) {
		t.Errorf("CompactFunc(%v) = %v, want %v", copy, s, want)
	}

	if mem[3] != nil || mem[4] != nil {
		// Check that potential memory leak is avoided
		t.Errorf("CompactFunc: want nil discarded elements, got %v, %v", mem[3], mem[4])
	}
	if mem[5] != &f {
		t.Errorf("CompactFunc: want unchanged elements beyond original len, got %v", mem[5])
	}
}

func BenchmarkCompactFunc(b *testing.B) {
	for _, c := range compactTests {
		b.Run(c.name, func(b *testing.B) {
			ss := make([]int, 0, 64)
			for k := 0; k < b.N; k++ {
				ss = ss[:0]
				ss = append(ss, c.s...)
				_ = CompactFunc(ss, func(a, b int) bool { return a == b })
			}
		})
	}
}

func BenchmarkCompactFunc_Large(b *testing.B) {
	type Element = int
	const N = 1024 * 1024

	b.Run("all_dup", func(b *testing.B) {
		ss := make([]Element, N)
		b.ResetTimer()
		for i := 0; i < b.N; i++ {
			_ = CompactFunc(ss, func(a, b Element) bool { return a == b })
		}
	})
	b.Run("no_dup", func(b *testing.B) {
		ss := make([]Element, N)
		for i := range ss {
			ss[i] = i
		}
		b.ResetTimer()
		for i := 0; i < b.N; i++ {
			_ = CompactFunc(ss, func(a, b Element) bool { return a == b })
		}
	})
}

func TestGrow(t *testing.T) {
	s1 := []int{1, 2, 3}

	copy := Clone(s1)
	s2 := Grow(copy, 1000)
	if !Equal(s1, s2) {
		t.Errorf("Grow(%v) = %v, want %v", s1, s2, s1)
	}
	if cap(s2) < 1000+len(s1) {
		t.Errorf("after Grow(%v) cap = %d, want >= %d", s1, cap(s2), 1000+len(s1))
	}

	// Test mutation of elements between length and capacity.
	copy = Clone(s1)
	s3 := Grow(copy[:1], 2)[:3]
	if !Equal(s1, s3) {
		t.Errorf("Grow should not mutate elements between length and capacity")
	}
	s3 = Grow(copy[:1], 1000)[:3]
	if !Equal(s1, s3) {
		t.Errorf("Grow should not mutate elements between length and capacity")
	}

	// Test number of allocations.
	if n := testing.AllocsPerRun(100, func() { _ = Grow(s2, cap(s2)-len(s2)) }); n != 0 {
		t.Errorf("Grow should not allocate when given sufficient capacity; allocated %v times", n)
	}
	if n := testing.AllocsPerRun(100, func() { _ = Grow(s2, cap(s2)-len(s2)+1) }); n != 1 {
		errorf := t.Errorf
		if race.Enabled || testenv.OptimizationOff() {
			errorf = t.Logf // this allocates multiple times in race detector mode
		}
		errorf("Grow should allocate once when given insufficient capacity; allocated %v times", n)
	}

	// Test for negative growth sizes.
	var gotPanic bool
	func() {
		defer func() { gotPanic = recover() != nil }()
		_ = Grow(s1, -1)
	}()
	if !gotPanic {
		t.Errorf("Grow(-1) did not panic; expected a panic")
	}
}

func TestClip(t *testing.T) {
	s1 := []int{1, 2, 3, 4, 5, 6}[:3]
	orig := Clone(s1)
	if len(s1) != 3 {
		t.Errorf("len(%v) = %d, want 3", s1, len(s1))
	}
	if cap(s1) < 6 {
		t.Errorf("cap(%v[:3]) = %d, want >= 6", orig, cap(s1))
	}
	s2 := Clip(s1)
	if !Equal(s1, s2) {
		t.Errorf("Clip(%v) = %v, want %v", s1, s2, s1)
	}
	if cap(s2) != 3 {
		t.Errorf("cap(Clip(%v)) = %d, want 3", orig, cap(s2))
	}
}

func TestReverse(t *testing.T) {
	even := []int{3, 1, 4, 1, 5, 9} // len = 6
	Reverse(even)
	if want := []int{9, 5, 1, 4, 1, 3}; !Equal(even, want) {
		t.Errorf("Reverse(even) = %v, want %v", even, want)
	}

	odd := []int{3, 1, 4, 1, 5, 9, 2} // len = 7
	Reverse(odd)
	if want := []int{2, 9, 5, 1, 4, 1, 3}; !Equal(odd, want) {
		t.Errorf("Reverse(odd) = %v, want %v", odd, want)
	}

	words := strings.Fields("one two three")
	Reverse(words)
	if want := strings.Fields("three two one"); !Equal(words, want) {
		t.Errorf("Reverse(words) = %v, want %v", words, want)
	}

	singleton := []string{"one"}
	Reverse(singleton)
	if want := []string{"one"}; !Equal(singleton, want) {
		t.Errorf("Reverse(singeleton) = %v, want %v", singleton, want)
	}

	Reverse[[]string](nil)
}

// naiveReplace is a baseline implementation to the Replace function.
func naiveReplace[S ~[]E, E any](s S, i, j int, v ...E) S {
	s = Delete(s, i, j)
	s = Insert(s, i, v...)
	return s
}

func TestReplace(t *testing.T) {
	for _, test := range []struct {
		s, v []int
		i, j int
	}{
		{}, // all zero value
		{
			s: []int{1, 2, 3, 4},
			v: []int{5},
			i: 1,
			j: 2,
		},
		{
			s: []int{1, 2, 3, 4},
			v: []int{5, 6, 7, 8},
			i: 1,
			j: 2,
		},
		{
			s: func() []int {
				s := make([]int, 3, 20)
				s[0] = 0
				s[1] = 1
				s[2] = 2
				return s
			}(),
			v: []int{3, 4, 5, 6, 7},
			i: 0,
			j: 1,
		},
	} {
		ss, vv := Clone(test.s), Clone(test.v)
		want := naiveReplace(ss, test.i, test.j, vv...)
		got := Replace(test.s, test.i, test.j, test.v...)
		if !Equal(got, want) {
			t.Errorf("Replace(%v, %v, %v, %v) = %v, want %v", test.s, test.i, test.j, test.v, got, want)
		}
	}
}

func TestReplacePanics(t *testing.T) {
	s := []int{0, 1, 2, 3, 4}
	s = s[0:2]
	_ = s[0:4] // this is a valid slice of s

	for _, test := range []struct {
		name string
		s, v []int
		i, j int
	}{
		{"indexes out of order", []int{1, 2}, []int{3}, 2, 1},
		{"large index", []int{1, 2}, []int{3}, 1, 10},
		{"negative index", []int{1, 2}, []int{3}, -1, 2},
		{"s[i:j] is valid and j > len(s)", s, nil, 0, 4},
	} {
		ss, vv := Clone(test.s), Clone(test.v)
		if !panics(func() { _ = Replace(ss, test.i, test.j, vv...) }) {
			t.Errorf("Replace %s: should have panicked", test.name)
		}
	}
}

func TestReplaceGrow(t *testing.T) {
	// When Replace needs to allocate a new slice, we want the original slice
	// to not be changed.
	a, b, c, d, e, f := 1, 2, 3, 4, 5, 6
	mem := []*int{&a, &b, &c, &d, &e, &f}
	memcopy := Clone(mem)
	s := mem[0:5] // there is 1 element beyond len(s), within cap(s)
	copy := Clone(s)
	original := s

	// The new elements don't fit within cap(s), so Replace will allocate.
	z := 99
	s = Replace(s, 1, 3, &z, &z, &z, &z)

	if want := []*int{&a, &z, &z, &z, &z, &d, &e}; !Equal(s, want) {
		t.Errorf("Replace(%v, 1, 3, %v, %v, %v, %v) = %v, want %v", copy, &z, &z, &z, &z, s, want)
	}

	if !Equal(original, copy) {
		t.Errorf("original slice has changed, got %v, want %v", original, copy)
	}

	if !Equal(mem, memcopy) {
		// Changing the original tail s[len(s):cap(s)] is unwanted
		t.Errorf("original backing memory has changed, got %v, want %v", mem, memcopy)
	}
}

func TestReplaceClearTail(t *testing.T) {
	a, b, c, d, e, f := 1, 2, 3, 4, 5, 6
	mem := []*int{&a, &b, &c, &d, &e, &f}
	s := mem[0:5] // there is 1 element beyond len(s), within cap(s)
	copy := Clone(s)

	y, z := 8, 9
	s = Replace(s, 1, 4, &y, &z)

	if want := []*int{&a, &y, &z, &e}; !Equal(s, want) {
		t.Errorf("Replace(%v) = %v, want %v", copy, s, want)
	}

	if mem[4] != nil {
		// Check that potential memory leak is avoided
		t.Errorf("Replace: want nil discarded element, got %v", mem[4])
	}
	if mem[5] != &f {
		t.Errorf("Replace: want unchanged elements beyond original len, got %v", mem[5])
	}
}

func TestReplaceOverlap(t *testing.T) {
	const N = 10
	a := make([]int, N)
	want := make([]int, 2*N)
	for n := 0; n <= N; n++ { // length
		for i := 0; i <= n; i++ { // insertion point 1
			for j := i; j <= n; j++ { // insertion point 2
				for x := 0; x <= N; x++ { // start of inserted data
					for y := x; y <= N; y++ { // end of inserted data
						for k := 0; k < N; k++ {
							a[k] = k
						}
						want = want[:0]
						want = append(want, a[:i]...)
						want = append(want, a[x:y]...)
						want = append(want, a[j:n]...)
						got := Replace(a[:n], i, j, a[x:y]...)
						if !Equal(got, want) {
							t.Errorf("Insert with overlap failed n=%d i=%d j=%d x=%d y=%d, got %v want %v", n, i, j, x, y, got, want)
						}
					}
				}
			}
		}
	}
}

func TestReplaceEndClearTail(t *testing.T) {
	s := []int{11, 22, 33}
	v := []int{99}
	// case when j == len(s)
	i, j := 1, 3
	s = Replace(s, i, j, v...)

	x := s[:3][2]
	if want := 0; x != want {
		t.Errorf("TestReplaceEndClearTail: obsolete element is %d, want %d", x, want)
	}
}

func BenchmarkReplace(b *testing.B) {
	cases := []struct {
		name string
		s, v func() []int
		i, j int
	}{
		{
			name: "fast",
			s: func() []int {
				return make([]int, 100)
			},
			v: func() []int {
				return make([]int, 20)
			},
			i: 10,
			j: 40,
		},
		{
			name: "slow",
			s: func() []int {
				return make([]int, 100)
			},
			v: func() []int {
				return make([]int, 20)
			},
			i: 0,
			j: 2,
		},
	}

	for _, c := range cases {
		b.Run("naive-"+c.name, func(b *testing.B) {
			for k := 0; k < b.N; k++ {
				s := c.s()
				v := c.v()
				_ = naiveReplace(s, c.i, c.j, v...)
			}
		})
		b.Run("optimized-"+c.name, func(b *testing.B) {
			for k := 0; k < b.N; k++ {
				s := c.s()
				v := c.v()
				_ = Replace(s, c.i, c.j, v...)
			}
		})
	}

}

func TestInsertGrowthRate(t *testing.T) {
	b := make([]byte, 1)
	maxCap := cap(b)
	nGrow := 0
	const N = 1e6
	for i := 0; i < N; i++ {
		b = Insert(b, len(b)-1, 0)
		if cap(b) > maxCap {
			maxCap = cap(b)
			nGrow++
		}
	}
	want := int(math.Log(N) / math.Log(1.25)) // 1.25 == growth rate for large slices
	if nGrow > want {
		t.Errorf("too many grows. got:%d want:%d", nGrow, want)
	}
}

func TestReplaceGrowthRate(t *testing.T) {
	b := make([]byte, 2)
	maxCap := cap(b)
	nGrow := 0
	const N = 1e6
	for i := 0; i < N; i++ {
		b = Replace(b, len(b)-2, len(b)-1, 0, 0)
		if cap(b) > maxCap {
			maxCap = cap(b)
			nGrow++
		}
	}
	want := int(math.Log(N) / math.Log(1.25)) // 1.25 == growth rate for large slices
	if nGrow > want {
		t.Errorf("too many grows. got:%d want:%d", nGrow, want)
	}
}

func apply[T any](v T, f func(T)) {
	f(v)
}

// Test type inference with a named slice type.
func TestInference(t *testing.T) {
	s1 := []int{1, 2, 3}
	apply(s1, Reverse)
	if want := []int{3, 2, 1}; !Equal(s1, want) {
		t.Errorf("Reverse(%v) = %v, want %v", []int{1, 2, 3}, s1, want)
	}

	type S []int
	s2 := S{4, 5, 6}
	apply(s2, Reverse)
	if want := (S{6, 5, 4}); !Equal(s2, want) {
		t.Errorf("Reverse(%v) = %v, want %v", S{4, 5, 6}, s2, want)
	}
}

func TestConcat(t *testing.T) {
	cases := []struct {
		s    [][]int
		want []int
	}{
		{
			s:    [][]int{nil},
			want: nil,
		},
		{
			s:    [][]int{{1}},
			want: []int{1},
		},
		{
			s:    [][]int{{1}, {2}},
			want: []int{1, 2},
		},
		{
			s:    [][]int{{1}, nil, {2}},
			want: []int{1, 2},
		},
	}
	for _, tc := range cases {
		got := Concat(tc.s...)
		if !Equal(tc.want, got) {
			t.Errorf("Concat(%v) = %v, want %v", tc.s, got, tc.want)
		}
		var sink []int
		allocs := testing.AllocsPerRun(5, func() {
			sink = Concat(tc.s...)
		})
		_ = sink
		if allocs > 1 {
			errorf := t.Errorf
			if testenv.OptimizationOff() || race.Enabled {
				errorf = t.Logf
			}
			errorf("Concat(%v) allocated %v times; want 1", tc.s, allocs)
		}
	}
}

func TestConcat_too_large(t *testing.T) {
	// Use zero length element to minimize memory in testing
	type void struct{}
	cases := []struct {
		lengths     []int
		shouldPanic bool
	}{
		{
			lengths:     []int{0, 0},
			shouldPanic: false,
		},
		{
			lengths:     []int{math.MaxInt, 0},
			shouldPanic: false,
		},
		{
			lengths:     []int{0, math.MaxInt},
			shouldPanic: false,
		},
		{
			lengths:     []int{math.MaxInt - 1, 1},
			shouldPanic: false,
		},
		{
			lengths:     []int{math.MaxInt - 1, 1, 1},
			shouldPanic: true,
		},
		{
			lengths:     []int{math.MaxInt, 1},
			shouldPanic: true,
		},
		{
			lengths:     []int{math.MaxInt, math.MaxInt},
			shouldPanic: true,
		},
	}
	for _, tc := range cases {
		var r any
		ss := make([][]void, 0, len(tc.lengths))
		for _, l := range tc.lengths {
			s := make([]void, l)
			ss = append(ss, s)
		}
		func() {
			defer func() {
				r = recover()
			}()
			_ = Concat(ss...)
		}()
		if didPanic := r != nil; didPanic != tc.shouldPanic {
			t.Errorf("slices.Concat(lens(%v)) got panic == %v",
				tc.lengths, didPanic)
		}
	}
}

func TestRepeat(t *testing.T) {
	// normal cases
	for _, tc := range []struct {
		x     []int
		count int
		want  []int
	}{
		{x: []int(nil), count: 0, want: []int{}},
		{x: []int(nil), count: 1, want: []int{}},
		{x: []int(nil), count: math.MaxInt, want: []int{}},
		{x: []int{}, count: 0, want: []int{}},
		{x: []int{}, count: 1, want: []int{}},
		{x: []int{}, count: math.MaxInt, want: []int{}},
		{x: []int{0}, count: 0, want: []int{}},
		{x: []int{0}, count: 1, want: []int{0}},
		{x: []int{0}, count: 2, want: []int{0, 0}},
		{x: []int{0}, count: 3, want: []int{0, 0, 0}},
		{x: []int{0}, count: 4, want: []int{0, 0, 0, 0}},
		{x: []int{0, 1}, count: 0, want: []int{}},
		{x: []int{0, 1}, count: 1, want: []int{0, 1}},
		{x: []int{0, 1}, count: 2, want: []int{0, 1, 0, 1}},
		{x: []int{0, 1}, count: 3, want: []int{0, 1, 0, 1, 0, 1}},
		{x: []int{0, 1}, count: 4, want: []int{0, 1, 0, 1, 0, 1, 0, 1}},
		{x: []int{0, 1, 2}, count: 0, want: []int{}},
		{x: []int{0, 1, 2}, count: 1, want: []int{0, 1, 2}},
		{x: []int{0, 1, 2}, count: 2, want: []int{0, 1, 2, 0, 1, 2}},
		{x: []int{0, 1, 2}, count: 3, want: []int{0, 1, 2, 0, 1, 2, 0, 1, 2}},
		{x: []int{0, 1, 2}, count: 4, want: []int{0, 1, 2, 0, 1, 2, 0, 1, 2, 0, 1, 2}},
	} {
		if got := Repeat(tc.x, tc.count); got == nil || cap(got) != cap(tc.want) || !Equal(got, tc.want) {
			t.Errorf("Repeat(%v, %v): got: %v, want: %v, (got == nil): %v, cap(got): %v, cap(want): %v",
				tc.x, tc.count, got, tc.want, got == nil, cap(got), cap(tc.want))
		}
	}

	// big slices
	for _, tc := range []struct {
		x     []struct{}
		count int
		want  []struct{}
	}{
		{x: make([]struct{}, math.MaxInt/1-0), count: 1, want: make([]struct{}, 1*(math.MaxInt/1-0))},
		{x: make([]struct{}, math.MaxInt/2-1), count: 2, want: make([]struct{}, 2*(math.MaxInt/2-1))},
		{x: make([]struct{}, math.MaxInt/3-2), count: 3, want: make([]struct{}, 3*(math.MaxInt/3-2))},
		{x: make([]struct{}, math.MaxInt/4-3), count: 4, want: make([]struct{}, 4*(math.MaxInt/4-3))},
		{x: make([]struct{}, math.MaxInt/5-4), count: 5, want: make([]struct{}, 5*(math.MaxInt/5-4))},
		{x: make([]struct{}, math.MaxInt/6-5), count: 6, want: make([]struct{}, 6*(math.MaxInt/6-5))},
		{x: make([]struct{}, math.MaxInt/7-6), count: 7, want: make([]struct{}, 7*(math.MaxInt/7-6))},
		{x: make([]struct{}, math.MaxInt/8-7), count: 8, want: make([]struct{}, 8*(math.MaxInt/8-7))},
		{x: make([]struct{}, math.MaxInt/9-8), count: 9, want: make([]struct{}, 9*(math.MaxInt/9-8))},
	} {
		if got := Repeat(tc.x, tc.count); got == nil || len(got) != len(tc.want) || cap(got) != cap(tc.want) {
			t.Errorf("Repeat(make([]struct{}, %v), %v): (got == nil): %v, len(got): %v, len(want): %v, cap(got): %v, cap(want): %v",
				len(tc.x), tc.count, got == nil, len(got), len(tc.want), cap(got), cap(tc.want))
		}
	}
}

func TestRepeatPanics(t *testing.T) {
	for _, test := range []struct {
		name  string
		x     []struct{}
		count int
	}{
		{name: "cannot be negative", x: make([]struct{}, 0), count: -1},
		{name: "the result of (len(x) * count) overflows, hi > 0", x: make([]struct{}, 3), count: math.MaxInt},
		{name: "the result of (len(x) * count) overflows, lo > maxInt", x: make([]struct{}, 2), count: 1 + math.MaxInt/2},
	} {
		if !panics(func() { _ = Repeat(test.x, test.count) }) {
			t.Errorf("Repeat %s: got no panic, want panic", test.name)
		}
	}
}

var cardinalityTests = []struct {
	s    []int
	v    int
	want int
}{
	{
		nil,
		0,
		0,
	},
	{
		[]int{},
		0,
		0,
	},
	{
		[]int{1, 2, 3},
		2,
		1,
	},
	{
		[]int{1, 2, 2, 3},
		2,
		2,
	},
	{
		[]int{1, 2, 3, 2},
		2,
		2,
	},
}

func TestCardinality(t *testing.T) {
	for _, test := range cardinalityTests {
		if got := Cardinality(test.s, test.v); got != test.want {
			t.Errorf("Cardinality(%v, %v) = %d, want %d", test.s, test.v, got, test.want)
		}
	}
}

func BenchmarkCardinality_Large(b *testing.B) {
	type Large [4 * 1024]byte

	ss := make([]Large, 1024)
	for i := 0; i < b.N; i++ {
		_ = Cardinality(ss, Large{1})
	}
}

func TestCardinalityFunc(t *testing.T) {
	for _, test := range cardinalityTests {
		if got := CardinalityFunc(test.s, equalToIndex(equal[int], test.v)); got != test.want {
			t.Errorf("CardinalityFunc(%v, equalToIndex(equal[int], %v)) = %d, want %d", test.s, test.v, got, test.want)
		}
	}
}

var cardinalityMapTests = []struct {
	s    []int
	want map[int]int
}{
	{
		nil,
		map[int]int{},
	},
	{
		[]int{},
		map[int]int{},
	},
	{
		[]int{1, 2, 3},
		map[int]int{1: 1, 2: 1, 3: 1},
	},
	{
		[]int{1, 2, 2, 3},
		map[int]int{1: 1, 2: 2, 3: 1},
	},
	{
		[]int{1, 2, 3, 2, 3, 3},
		map[int]int{1: 1, 2: 2, 3: 3},
	},
}

func TestCardinalityMap(t *testing.T) {
	for _, test := range cardinalityMapTests {
		if got := CardinalityMap(test.s); maps.Equal(got, test.want) {
			t.Errorf("CardinalityMap(%v) = %v, want %v", test.s, got, test.want)
		}
	}
}

func BenchmarkCardinalityMap_Large(b *testing.B) {
	type Large [4 * 1024]byte

	ss := make([]Large, 1024)
	for i := 0; i < b.N; i++ {
		_ = CardinalityMap(ss)
	}
}
