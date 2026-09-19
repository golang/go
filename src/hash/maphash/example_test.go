// Copyright 2025 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

package maphash_test

import (
	"fmt"
	"hash/maphash"
	"strings"
)

// CaseInsensitive is a Hasher[string] whose
// equivalence relation ignores letter case.
type CaseInsensitive struct{}

func (CaseInsensitive) Hash(h *maphash.Hash, s string) {
	h.WriteString(strings.ToLower(s))
}

func (CaseInsensitive) Equal(x, y string) bool {
	return strings.ToLower(x) == strings.ToLower(y)
}

// ExampleHasher demonstrates using the Hasher interface to create
// a custom hash function for case-insensitive string comparison.
func ExampleHasher() {
	var hasher CaseInsensitive

	// Create a hash with a fixed seed for reproducibility
	seed := maphash.MakeSeed()
	var h1, h2 maphash.Hash
	h1.SetSeed(seed)
	h2.SetSeed(seed)

	// Hash "Hello" and "HELLO" - they should have the same hash
	hasher.Hash(&h1, "Hello")
	hasher.Hash(&h2, "HELLO")

	fmt.Printf("Equal: %v\n", hasher.Equal("Hello", "HELLO"))
	fmt.Printf("Same hash: %v\n", h1.Sum64() == h2.Sum64())

	// Output:
	// Equal: true
	// Same hash: true
}

// ExampleComparableHasher demonstrates using ComparableHasher
// for standard comparable types.
func ExampleComparableHasher() {
	var hasher maphash.ComparableHasher[int]

	seed := maphash.MakeSeed()
	var h1, h2 maphash.Hash
	h1.SetSeed(seed)
	h2.SetSeed(seed)

	// Hash the same value twice
	hasher.Hash(&h1, 42)
	hasher.Hash(&h2, 42)

	fmt.Printf("Equal: %v\n", hasher.Equal(42, 42))
	fmt.Printf("Same hash: %v\n", h1.Sum64() == h2.Sum64())
	fmt.Printf("Different values equal: %v\n", hasher.Equal(42, 43))

	// Output:
	// Equal: true
	// Same hash: true
	// Different values equal: false
}

type Person struct {
	Name string
	Age  int
	ID   int // ID is ignored in comparison
}

// PersonByNameAge compares Person values by Name and Age only
type PersonByNameAge struct{}

func (PersonByNameAge) Hash(h *maphash.Hash, p Person) {
	h.WriteString(p.Name)
	maphash.WriteComparable(h, p.Age)
}

func (PersonByNameAge) Equal(x, y Person) bool {
	return x.Name == y.Name && x.Age == y.Age
}

// ExampleHasher_structFields demonstrates creating a Hasher
// for struct types that compares only specific fields.
func ExampleHasher_structFields() {
	var hasher PersonByNameAge

	p1 := Person{Name: "Alice", Age: 30, ID: 1}
	p2 := Person{Name: "Alice", Age: 30, ID: 2} // Different ID

	fmt.Printf("Equal (ignoring ID): %v\n", hasher.Equal(p1, p2))

	// Output:
	// Equal (ignoring ID): true
}

// SimpleSet represents an unordered set of integers
type SimpleSet map[int]struct{}

type SetHasher struct{}

func (SetHasher) Hash(hash *maphash.Hash, set SimpleSet) {
	var accum uint64
	for elem := range set {
		// Initialize a hasher for the element,
		// using same seed as the outer hash.
		var sub maphash.Hash
		sub.SetSeed(hash.Seed())

		// Hash the element.
		maphash.WriteComparable(&sub, elem)

		// Mix the element's hash into the set's hash using XOR (commutative).
		accum ^= sub.Sum64()
	}
	maphash.WriteComparable(hash, accum)
}

func (SetHasher) Equal(x, y SimpleSet) bool {
	if len(x) != len(y) {
		return false
	}
	for elem := range x {
		if _, ok := y[elem]; !ok {
			return false
		}
	}
	return true
}

// ExampleHasher_unorderedSet demonstrates hashing an unordered collection
// using a commutative operation.
func ExampleHasher_unorderedSet() {
	var hasher SetHasher

	// Create two sets with same elements in different order
	set1 := SimpleSet{1: {}, 2: {}, 3: {}}
	set2 := SimpleSet{3: {}, 1: {}, 2: {}}

	seed := maphash.MakeSeed()
	var h1, h2 maphash.Hash
	h1.SetSeed(seed)
	h2.SetSeed(seed)

	hasher.Hash(&h1, set1)
	hasher.Hash(&h2, set2)

	fmt.Printf("Equal: %v\n", hasher.Equal(set1, set2))
	fmt.Printf("Same hash: %v\n", h1.Sum64() == h2.Sum64())

	// Output:
	// Equal: true
	// Same hash: true
}

type Data struct {
	Value int
}

// DataPtrHasher compares *Data by value, not by pointer address
type DataPtrHasher struct{}

func (DataPtrHasher) Hash(h *maphash.Hash, p *Data) {
	if p == nil {
		h.WriteByte(0)
		return
	}
	h.WriteByte(1)
	maphash.WriteComparable(h, p.Value)
}

func (DataPtrHasher) Equal(x, y *Data) bool {
	if x == nil && y == nil {
		return true
	}
	if x == nil || y == nil {
		return false
	}
	return x.Value == y.Value
}

// ExampleHasher_pointerComparison demonstrates creating a Hasher
// for pointer types that compares the pointed-to values.
func ExampleHasher_pointerComparison() {
	var hasher DataPtrHasher

	// Different pointers, same value
	p1 := &Data{Value: 42}
	p2 := &Data{Value: 42}

	fmt.Printf("Pointers equal: %v\n", p1 == p2)
	fmt.Printf("Values equal: %v\n", hasher.Equal(p1, p2))

	// Output:
	// Pointers equal: false
	// Values equal: true
}
