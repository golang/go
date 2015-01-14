// Copyright 2014 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

package intsets_test

import (
	"fmt"
	"log"
	"math/rand"
	"sort"
	"strings"
	"testing"

	"golang.org/x/tools/container/intsets"
)

func TestBasics(t *testing.T) {
	var s intsets.Sparse
	if len := s.Len(); len != 0 {
		t.Errorf("Len({}): got %d, want 0", len)
	}
	if s := s.String(); s != "{}" {
		t.Errorf("String({}): got %q, want \"{}\"", s)
	}
	if s.Has(3) {
		t.Errorf("Has(3): got true, want false")
	}
	if err := s.Check(); err != nil {
		t.Error(err)
	}

	if !s.Insert(3) {
		t.Errorf("Insert(3): got false, want true")
	}
	if max := s.Max(); max != 3 {
		t.Errorf("Max: got %d, want 3", max)
	}

	if !s.Insert(435) {
		t.Errorf("Insert(435): got false, want true")
	}
	if s := s.String(); s != "{3 435}" {
		t.Errorf("String({3 435}): got %q, want \"{3 435}\"", s)
	}
	if max := s.Max(); max != 435 {
		t.Errorf("Max: got %d, want 435", max)
	}
	if len := s.Len(); len != 2 {
		t.Errorf("Len: got %d, want 2", len)
	}

	if !s.Remove(435) {
		t.Errorf("Remove(435): got false, want true")
	}
	if s := s.String(); s != "{3}" {
		t.Errorf("String({3}): got %q, want \"{3}\"", s)
	}
}

// Insert, Len, IsEmpty, Hash, Clear, AppendTo.
func TestMoreBasics(t *testing.T) {
	set := new(intsets.Sparse)
	set.Insert(456)
	set.Insert(123)
	set.Insert(789)
	if set.Len() != 3 {
		t.Errorf("%s.Len: got %d, want 3", set, set.Len())
	}
	if set.IsEmpty() {
		t.Errorf("%s.IsEmpty: got true", set)
	}
	if !set.Has(123) {
		t.Errorf("%s.Has(123): got false", set)
	}
	if set.Has(1234) {
		t.Errorf("%s.Has(1234): got true", set)
	}
	got := set.AppendTo([]int{-1})
	if want := []int{-1, 123, 456, 789}; fmt.Sprint(got) != fmt.Sprint(want) {
		t.Errorf("%s.AppendTo: got %v, want %v", set, got, want)
	}

	set.Clear()

	if set.Len() != 0 {
		t.Errorf("Clear: got %d, want 0", set.Len())
	}
	if !set.IsEmpty() {
		t.Errorf("IsEmpty: got false")
	}
	if set.Has(123) {
		t.Errorf("%s.Has: got false", set)
	}
}

func TestTakeMin(t *testing.T) {
	var set intsets.Sparse
	set.Insert(456)
	set.Insert(123)
	set.Insert(789)
	set.Insert(-123)
	var got int
	for i, want := range []int{-123, 123, 456, 789} {
		if !set.TakeMin(&got) || got != want {
			t.Errorf("TakeMin #%d: got %d, want %d", i, got, want)
		}
	}
	if set.TakeMin(&got) {
		t.Errorf("%s.TakeMin returned true", &set)
	}
	if err := set.Check(); err != nil {
		t.Fatalf("check: %s: %#v", err, &set)
	}
}

func TestMinAndMax(t *testing.T) {
	values := []int{0, 456, 123, 789, -123} // elt 0 => empty set
	wantMax := []int{intsets.MinInt, 456, 456, 789, 789}
	wantMin := []int{intsets.MaxInt, 456, 123, 123, -123}

	var set intsets.Sparse
	for i, x := range values {
		if i != 0 {
			set.Insert(x)
		}
		if got, want := set.Min(), wantMin[i]; got != want {
			t.Errorf("Min #%d: got %d, want %d", i, got, want)
		}
		if got, want := set.Max(), wantMax[i]; got != want {
			t.Errorf("Max #%d: got %d, want %d", i, got, want)
		}
	}

	set.Insert(intsets.MinInt)
	if got, want := set.Min(), intsets.MinInt; got != want {
		t.Errorf("Min: got %d, want %d", got, want)
	}

	set.Insert(intsets.MaxInt)
	if got, want := set.Max(), intsets.MaxInt; got != want {
		t.Errorf("Max: got %d, want %d", got, want)
	}
}

func TestEquals(t *testing.T) {
	var setX intsets.Sparse
	setX.Insert(456)
	setX.Insert(123)
	setX.Insert(789)

	if !setX.Equals(&setX) {
		t.Errorf("Equals(%s, %s): got false", &setX, &setX)
	}

	var setY intsets.Sparse
	setY.Insert(789)
	setY.Insert(456)
	setY.Insert(123)

	if !setX.Equals(&setY) {
		t.Errorf("Equals(%s, %s): got false", &setX, &setY)
	}

	setY.Insert(1)
	if setX.Equals(&setY) {
		t.Errorf("Equals(%s, %s): got true", &setX, &setY)
	}

	var empty intsets.Sparse
	if setX.Equals(&empty) {
		t.Errorf("Equals(%s, %s): got true", &setX, &empty)
	}

	// Edge case: some block (with offset=0) appears in X but not Y.
	setY.Remove(123)
	if setX.Equals(&setY) {
		t.Errorf("Equals(%s, %s): got true", &setX, &setY)
	}
}

// A pset is a parallel implementation of a set using both an intsets.Sparse
// and a built-in hash map.
type pset struct {
	hash map[int]bool
	bits intsets.Sparse
}

func makePset() *pset {
	return &pset{hash: make(map[int]bool)}
}

func (set *pset) add(n int) {
	prev := len(set.hash)
	set.hash[n] = true
	grewA := len(set.hash) > prev

	grewB := set.bits.Insert(n)

	if grewA != grewB {
		panic(fmt.Sprintf("add(%d): grewA=%t grewB=%t", n, grewA, grewB))
	}
}

func (set *pset) remove(n int) {
	prev := len(set.hash)
	delete(set.hash, n)
	shrankA := len(set.hash) < prev

	shrankB := set.bits.Remove(n)

	if shrankA != shrankB {
		panic(fmt.Sprintf("remove(%d): shrankA=%t shrankB=%t", n, shrankA, shrankB))
	}
}

func (set *pset) check(t *testing.T, msg string) {
	var eltsA []int
	for elt := range set.hash {
		eltsA = append(eltsA, int(elt))
	}
	sort.Ints(eltsA)

	eltsB := set.bits.AppendTo(nil)

	if a, b := fmt.Sprint(eltsA), fmt.Sprint(eltsB); a != b {
		t.Errorf("check(%s): hash=%s bits=%s (%s)", msg, a, b, &set.bits)
	}

	if err := set.bits.Check(); err != nil {
		t.Fatalf("Check(%s): %s: %#v", msg, err, &set.bits)
	}
}

// randomPset returns a parallel set of random size and elements.
func randomPset(prng *rand.Rand, maxSize int) *pset {
	set := makePset()
	size := int(prng.Int()) % maxSize
	for i := 0; i < size; i++ {
		// TODO(adonovan): benchmark how performance varies
		// with this sparsity parameter.
		n := int(prng.Int()) % 10000
		set.add(n)
	}
	return set
}

// TestRandomMutations performs the same random adds/removes on two
// set implementations and ensures that they compute the same result.
func TestRandomMutations(t *testing.T) {
	const debug = false

	set := makePset()
	prng := rand.New(rand.NewSource(0))
	for i := 0; i < 10000; i++ {
		n := int(prng.Int())%2000 - 1000
		if i%2 == 0 {
			if debug {
				log.Printf("add %d", n)
			}
			set.add(n)
		} else {
			if debug {
				log.Printf("remove %d", n)
			}
			set.remove(n)
		}
		if debug {
			set.check(t, "post mutation")
		}
	}
	set.check(t, "final")
	if debug {
		log.Print(&set.bits)
	}
}

// TestSetOperations exercises classic set operations: ∩ , ∪, \.
func TestSetOperations(t *testing.T) {
	prng := rand.New(rand.NewSource(0))

	// Use random sets of sizes from 0 to about 1000.
	// For each operator, we test variations such as
	// Z.op(X, Y), Z.op(X, Z) and Z.op(Z, Y) to exercise
	// the degenerate cases of each method implementation.
	for i := uint(0); i < 12; i++ {
		X := randomPset(prng, 1<<i)
		Y := randomPset(prng, 1<<i)

		// TODO(adonovan): minimise dependencies between stanzas below.

		// Copy(X)
		C := makePset()
		C.bits.Copy(&Y.bits) // no effect on result
		C.bits.Copy(&X.bits)
		C.hash = X.hash
		C.check(t, "C.Copy(X)")
		C.bits.Copy(&C.bits)
		C.check(t, "C.Copy(C)")

		// U.Union(X, Y)
		U := makePset()
		U.bits.Union(&X.bits, &Y.bits)
		for n := range X.hash {
			U.hash[n] = true
		}
		for n := range Y.hash {
			U.hash[n] = true
		}
		U.check(t, "U.Union(X, Y)")

		// U.Union(X, X)
		U.bits.Union(&X.bits, &X.bits)
		U.hash = X.hash
		U.check(t, "U.Union(X, X)")

		// U.Union(U, Y)
		U = makePset()
		U.bits.Copy(&X.bits)
		U.bits.Union(&U.bits, &Y.bits)
		for n := range X.hash {
			U.hash[n] = true
		}
		for n := range Y.hash {
			U.hash[n] = true
		}
		U.check(t, "U.Union(U, Y)")

		// U.Union(X, U)
		U.bits.Copy(&Y.bits)
		U.bits.Union(&X.bits, &U.bits)
		U.check(t, "U.Union(X, U)")

		// U.UnionWith(U)
		U.bits.UnionWith(&U.bits)
		U.check(t, "U.UnionWith(U)")

		// I.Intersection(X, Y)
		I := makePset()
		I.bits.Intersection(&X.bits, &Y.bits)
		for n := range X.hash {
			if Y.hash[n] {
				I.hash[n] = true
			}
		}
		I.check(t, "I.Intersection(X, Y)")

		// I.Intersection(X, X)
		I.bits.Intersection(&X.bits, &X.bits)
		I.hash = X.hash
		I.check(t, "I.Intersection(X, X)")

		// I.Intersection(I, X)
		I.bits.Intersection(&I.bits, &X.bits)
		I.check(t, "I.Intersection(I, X)")

		// I.Intersection(X, I)
		I.bits.Intersection(&X.bits, &I.bits)
		I.check(t, "I.Intersection(X, I)")

		// I.Intersection(I, I)
		I.bits.Intersection(&I.bits, &I.bits)
		I.check(t, "I.Intersection(I, I)")

		// D.Difference(X, Y)
		D := makePset()
		D.bits.Difference(&X.bits, &Y.bits)
		for n := range X.hash {
			if !Y.hash[n] {
				D.hash[n] = true
			}
		}
		D.check(t, "D.Difference(X, Y)")

		// D.Difference(D, Y)
		D.bits.Copy(&X.bits)
		D.bits.Difference(&D.bits, &Y.bits)
		D.check(t, "D.Difference(D, Y)")

		// D.Difference(Y, D)
		D.bits.Copy(&X.bits)
		D.bits.Difference(&Y.bits, &D.bits)
		D.hash = make(map[int]bool)
		for n := range Y.hash {
			if !X.hash[n] {
				D.hash[n] = true
			}
		}
		D.check(t, "D.Difference(Y, D)")

		// D.Difference(X, X)
		D.bits.Difference(&X.bits, &X.bits)
		D.hash = nil
		D.check(t, "D.Difference(X, X)")

		// D.DifferenceWith(D)
		D.bits.Copy(&X.bits)
		D.bits.DifferenceWith(&D.bits)
		D.check(t, "D.DifferenceWith(D)")

		// SD.SymmetricDifference(X, Y)
		SD := makePset()
		SD.bits.SymmetricDifference(&X.bits, &Y.bits)
		for n := range X.hash {
			if !Y.hash[n] {
				SD.hash[n] = true
			}
		}
		for n := range Y.hash {
			if !X.hash[n] {
				SD.hash[n] = true
			}
		}
		SD.check(t, "SD.SymmetricDifference(X, Y)")

		// X.SymmetricDifferenceWith(Y)
		SD.bits.Copy(&X.bits)
		SD.bits.SymmetricDifferenceWith(&Y.bits)
		SD.check(t, "X.SymmetricDifference(Y)")

		// Y.SymmetricDifferenceWith(X)
		SD.bits.Copy(&Y.bits)
		SD.bits.SymmetricDifferenceWith(&X.bits)
		SD.check(t, "Y.SymmetricDifference(X)")

		// SD.SymmetricDifference(X, X)
		SD.bits.SymmetricDifference(&X.bits, &X.bits)
		SD.hash = nil
		SD.check(t, "SD.SymmetricDifference(X, X)")

		// SD.SymmetricDifference(X, Copy(X))
		X2 := makePset()
		X2.bits.Copy(&X.bits)
		SD.bits.SymmetricDifference(&X.bits, &X2.bits)
		SD.check(t, "SD.SymmetricDifference(X, Copy(X))")

		// Copy(X).SymmetricDifferenceWith(X)
		SD.bits.Copy(&X.bits)
		SD.bits.SymmetricDifferenceWith(&X.bits)
		SD.check(t, "Copy(X).SymmetricDifferenceWith(X)")
	}
}

func TestIntersectionWith(t *testing.T) {
	// Edge cases: the pairs (1,1), (1000,2000), (8000,4000)
	// exercise the <, >, == cases in IntersectionWith that the
	// TestSetOperations data is too dense to cover.
	var X, Y intsets.Sparse
	X.Insert(1)
	X.Insert(1000)
	X.Insert(8000)
	Y.Insert(1)
	Y.Insert(2000)
	Y.Insert(4000)
	X.IntersectionWith(&Y)
	if got, want := X.String(), "{1}"; got != want {
		t.Errorf("IntersectionWith: got %s, want %s", got, want)
	}
}

func TestIntersects(t *testing.T) {
	prng := rand.New(rand.NewSource(0))

	for i := uint(0); i < 12; i++ {
		X, Y := randomPset(prng, 1<<i), randomPset(prng, 1<<i)
		x, y := &X.bits, &Y.bits

		// test the slow way
		var z intsets.Sparse
		z.Copy(x)
		z.IntersectionWith(y)

		if got, want := x.Intersects(y), !z.IsEmpty(); got != want {
			t.Errorf("Intersects: got %v, want %v", got, want)
		}

		// make it false
		a := x.AppendTo(nil)
		for _, v := range a {
			y.Remove(v)
		}

		if got, want := x.Intersects(y), false; got != want {
			t.Errorf("Intersects: got %v, want %v", got, want)
		}

		// make it true
		if x.IsEmpty() {
			continue
		}
		i := prng.Intn(len(a))
		y.Insert(a[i])

		if got, want := x.Intersects(y), true; got != want {
			t.Errorf("Intersects: got %v, want %v", got, want)
		}
	}
}

func TestSubsetOf(t *testing.T) {
	prng := rand.New(rand.NewSource(0))

	for i := uint(0); i < 12; i++ {
		X, Y := randomPset(prng, 1<<i), randomPset(prng, 1<<i)
		x, y := &X.bits, &Y.bits

		// test the slow way
		var z intsets.Sparse
		z.Copy(x)
		z.DifferenceWith(y)

		if got, want := x.SubsetOf(y), z.IsEmpty(); got != want {
			t.Errorf("SubsetOf: got %v, want %v", got, want)
		}

		// make it true
		y.UnionWith(x)

		if got, want := x.SubsetOf(y), true; got != want {
			t.Errorf("SubsetOf: got %v, want %v", got, want)
		}

		// make it false
		if x.IsEmpty() {
			continue
		}
		a := x.AppendTo(nil)
		i := prng.Intn(len(a))
		y.Remove(a[i])

		if got, want := x.SubsetOf(y), false; got != want {
			t.Errorf("SubsetOf: got %v, want %v", got, want)
		}
	}
}

func TestBitString(t *testing.T) {
	for _, test := range []struct {
		input []int
		want  string
	}{
		{nil, "0"},
		{[]int{0}, "1"},
		{[]int{0, 4, 5}, "110001"},
		{[]int{0, 7, 177}, "1" + strings.Repeat("0", 169) + "10000001"},
		{[]int{-3, 0, 4, 5}, "110001.001"},
		{[]int{-3}, "0.001"},
	} {
		var set intsets.Sparse
		for _, x := range test.input {
			set.Insert(x)
		}
		if got := set.BitString(); got != test.want {
			t.Errorf("BitString(%s) = %s, want %s", set.String(), got, test.want)
		}
	}
}

func TestFailFastOnShallowCopy(t *testing.T) {
	var x intsets.Sparse
	x.Insert(1)

	y := x // shallow copy (breaks representation invariants)
	defer func() {
		got := fmt.Sprint(recover())
		want := "A Sparse has been copied without (*Sparse).Copy()"
		if got != want {
			t.Errorf("shallow copy: recover() = %q, want %q", got, want)
		}
	}()
	y.String() // panics
	t.Error("didn't panic as expected")
}

// -- Benchmarks -------------------------------------------------------

// TODO(adonovan):
// - Add benchmarks of each method.
// - Gather set distributions from pointer analysis.
// - Measure memory usage.

func BenchmarkSparseBitVector(b *testing.B) {
	prng := rand.New(rand.NewSource(0))
	for tries := 0; tries < b.N; tries++ {
		var x, y, z intsets.Sparse
		for i := 0; i < 1000; i++ {
			n := int(prng.Int()) % 100000
			if i%2 == 0 {
				x.Insert(n)
			} else {
				y.Insert(n)
			}
		}
		z.Union(&x, &y)
		z.Difference(&x, &y)
	}
}

func BenchmarkHashTable(b *testing.B) {
	prng := rand.New(rand.NewSource(0))
	for tries := 0; tries < b.N; tries++ {
		x, y, z := make(map[int]bool), make(map[int]bool), make(map[int]bool)
		for i := 0; i < 1000; i++ {
			n := int(prng.Int()) % 100000
			if i%2 == 0 {
				x[n] = true
			} else {
				y[n] = true
			}
		}
		// union
		for n := range x {
			z[n] = true
		}
		for n := range y {
			z[n] = true
		}
		// difference
		z = make(map[int]bool)
		for n := range y {
			if !x[n] {
				z[n] = true
			}
		}
	}
}

func BenchmarkAppendTo(b *testing.B) {
	prng := rand.New(rand.NewSource(0))
	var x intsets.Sparse
	for i := 0; i < 1000; i++ {
		x.Insert(int(prng.Int()) % 10000)
	}
	var space [1000]int
	for tries := 0; tries < b.N; tries++ {
		x.AppendTo(space[:0])
	}
}
