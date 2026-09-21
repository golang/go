// Copyright 2026 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

package maphash_test

// This file demonstrates a Bloom filter.

import (
	"fmt"
	"hash/maphash"
	"math"
	"math/bits"
)

// BloomFilter is a Bloom filter, a probabilistic space-efficient
// representation of a set of values of type V based on hashing.
//
// It provides two operations: Insert inserts an element and Contains
// queries whether a value is a member of the set.
//
// However, unlike a typical set, the size is independent of the
// number of elements. The catch: Contains is permitted to report
// true even for some elements that are not present. The trade-off
// between space and accuracy is determined by parameters at
// construction.
type BloomFilter[V any] struct {
	hasher maphash.Hasher[V]
	seeds  []maphash.Seed // each seed determines a hash function
	bytes  []byte         // bit vector
}

// NewBloomFilterComparable returns a new BloomFilter for the
// specified elements using their natural comparison.
func NewComparableBloomFilter[V comparable](n int, fpProb float64) *BloomFilter[V] {
	return NewBloomFilter(maphash.ComparableHasher[V]{}, n, fpProb)
}

// NewBloomFilter constructs a new BloomFilter capable of holding n
// elements with the specified probability of false positive results,
// assuming a well dispersed hash function.
func NewBloomFilter[V any](hasher maphash.Hasher[V], n int, fpProb float64) *BloomFilter[V] {
	nbytes, nseeds := calibrate(n, fpProb)

	seeds := make([]maphash.Seed, nseeds)
	for i := range nseeds {
		seeds[i] = maphash.MakeSeed()
	}

	return &BloomFilter[V]{
		hasher: hasher,
		bytes:  make([]byte, nbytes),
		seeds:  seeds,
	}
}

func (f *BloomFilter[V]) Contains(v V) bool {
	for _, seed := range f.seeds {
		index, bit := f.locate(seed, v)
		if f.bytes[index]&bit == 0 {
			return false
		}
	}
	return true
}

func (f *BloomFilter[V]) Insert(v V) {
	for _, seed := range f.seeds {
		index, bit := f.locate(seed, v)
		f.bytes[index] |= bit
	}
}

func (f *BloomFilter[V]) locate(seed maphash.Seed, v V) (uint64, byte) {
	// Optimization note: the dynamic call to hasher.Hash causes h
	// to escape. You can use a sync.Pool can help mitigate the
	// allocation cost.
	//
	// Alternatively, you could copy and specialize the filter logic
	// for a specific implementation of maphash.Hasher, allowing
	// the compiler's escape analysis to determine that the
	// hasher.Hash call does not in fact cause h to escape.
	var h maphash.Hash
	h.SetSeed(seed)
	f.hasher.Hash(&h, v)
	hash := h.Sum64()

	index := reduce(hash, uint64(len(f.bytes)))
	mask := byte(1 << (hash % 8))
	return index, mask
}

// reduce maps hash to a value in the interval [0, n).
// See https://lemire.me/blog/2016/06/27/a-fast-alternative-to-the-modulo-reduction
func reduce(hash, n uint64) uint64 {
	if n > 1<<32-1 {
		hi, _ := bits.Mul64(hash, n)
		return hi
	}
	return hash >> 32 * n >> 32
}

func calibrate(n int, fpProb float64) (bytes, seeds int) {
	// Following https://en.wikipedia.org/wiki/Bloom_filter:
	// - k is the number of hash functions,
	// - m is the size of the bit field;
	// - n is the number of set bits.

	if n == 0 {
		return 1, 1
	}

	logFpProb := math.Log(fpProb)
	m := -(float64(n) * logFpProb) / (math.Ln2 * math.Ln2)

	// Round up to a byte.
	// TODO(adonovan): opt: round up to the next allocation size
	// class (see bytes.growSlice) and then compute the possibly
	// smaller number of needed seeds based on this higher number.
	bytes = int(m) / 8
	if float64(bytes*8) < m {
		bytes++
	}

	k := -logFpProb / math.Ln2
	seeds = max(int(math.Round(k)), 1)
	return bytes, seeds
}

func Example_bloomFilter() {
	// Create a Bloom filter optimized for 2 elements with
	// a one-in-a-billion false-positive rate.
	//
	// (This low rate demands a lot of space: 88 bits and
	// 30 hash functions. More typical rates are 1-5%;
	// at 5%, we need only 16 bits and 4 hash functions.)
	f := NewComparableBloomFilter[string](2, 1e-9)

	// Insert two elements.
	f.Insert("apple")
	f.Insert("banana")

	// Check whether elements are present.
	//
	// "cherry" was not inserted, but Contains is probabilistic, so
	// this test will spuriously report Contains("cherry") = true
	// about once every billion runs.
	for _, fruit := range []string{"apple", "banana", "cherry"} {
		fmt.Printf("Contains(%q) = %v\n", fruit, f.Contains(fruit))
	}

	// Output:
	//
	// Contains("apple") = true
	// Contains("banana") = true
	// Contains("cherry") = false
}
