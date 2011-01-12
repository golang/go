// Copyright 2010 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

// The suffixarray package implements substring search in logarithmic time
// using an in-memory suffix array.
//
// Example use:
//
//	// create index for some data
//	index := suffixarray.New(data)
//
//	// lookup byte slice s
//	offsets1 := index.Lookup(s, -1) // the list of all indices where s occurs in data
//	offsets2 := index.Lookup(s, 3)  // the list of at most 3 indices where s occurs in data
//
package suffixarray

import (
	"bytes"
	"regexp"
	"sort"
)


// Index implements a suffix array for fast substring search.
type Index struct {
	data []byte
	sa   []int // suffix array for data
}


// New creates a new Index for data.
// Index creation time is O(N*log(N)) for N = len(data).
func New(data []byte) *Index {
	return &Index{data, qsufsort(data)}
}


// Bytes returns the data over which the index was created.
// It must not be modified.
//
func (x *Index) Bytes() []byte {
	return x.data
}


func (x *Index) at(i int) []byte {
	return x.data[x.sa[i]:]
}


func (x *Index) search(s []byte) int {
	return sort.Search(len(x.sa), func(i int) bool { return bytes.Compare(x.at(i), s) >= 0 })
}


// Lookup returns an unsorted list of at most n indices where the byte string s
// occurs in the indexed data. If n < 0, all occurrences are returned.
// The result is nil if s is empty, s is not found, or n == 0.
// Lookup time is O((log(N) + len(result))*len(s)) where N is the
// size of the indexed data.
//
func (x *Index) Lookup(s []byte, n int) (result []int) {
	if len(s) > 0 && n != 0 {
		// find matching suffix index i
		i := x.search(s)
		// x.at(i-1) < s <= x.at(i)

		// collect the following suffixes with matching prefixes
		for (n < 0 || len(result) < n) && i < len(x.sa) && bytes.HasPrefix(x.at(i), s) {
			result = append(result, x.sa[i])
			i++
		}
	}
	return
}


// FindAllIndex returns a sorted list of non-overlapping matches of the
// regular expression r, where a match is a pair of indices specifying
// the matched slice of x.Bytes(). If n < 0, all matches are returned
// in successive order. Otherwise, at most n matches are returned and
// they may not be successive. The result is nil if there are no matches,
// or if n == 0.
//
func (x *Index) FindAllIndex(r *regexp.Regexp, n int) (result [][]int) {
	// a non-empty literal prefix is used to determine possible
	// match start indices with Lookup
	prefix, complete := r.LiteralPrefix()
	lit := []byte(prefix)

	// worst-case scenario: no literal prefix
	if prefix == "" {
		return r.FindAllIndex(x.data, n)
	}

	// if regexp is a literal just use Lookup and convert its
	// result into match pairs
	if complete {
		// Lookup returns indices that may belong to overlapping matches.
		// After eliminating them, we may end up with fewer than n matches.
		// If we don't have enough at the end, redo the search with an
		// increased value n1, but only if Lookup returned all the requested
		// indices in the first place (if it returned fewer than that then
		// there cannot be more).
		for n1 := n; ; n1 += 2 * (n - len(result)) /* overflow ok */ {
			indices := x.Lookup(lit, n1)
			if len(indices) == 0 {
				return
			}
			sort.SortInts(indices)
			pairs := make([]int, 2*len(indices))
			result = make([][]int, len(indices))
			count := 0
			prev := 0
			for _, i := range indices {
				if count == n {
					break
				}
				// ignore indices leading to overlapping matches
				if prev <= i {
					j := 2 * count
					pairs[j+0] = i
					pairs[j+1] = i + len(lit)
					result[count] = pairs[j : j+2]
					count++
					prev = i + len(lit)
				}
			}
			result = result[0:count]
			if len(result) >= n || len(indices) != n1 {
				// found all matches or there's no chance to find more
				// (n and n1 can be negative)
				break
			}
		}
		if len(result) == 0 {
			result = nil
		}
		return
	}

	// regexp has a non-empty literal prefix; Lookup(lit) computes
	// the indices of possible complete matches; use these as starting
	// points for anchored searches
	// (regexp "^" matches beginning of input, not beginning of line)
	r = regexp.MustCompile("^" + r.String()) // compiles because r compiled

	// same comment about Lookup applies here as in the loop above
	for n1 := n; ; n1 += 2 * (n - len(result)) /* overflow ok */ {
		indices := x.Lookup(lit, n1)
		if len(indices) == 0 {
			return
		}
		sort.SortInts(indices)
		result = result[0:0]
		prev := 0
		for _, i := range indices {
			if len(result) == n {
				break
			}
			m := r.FindIndex(x.data[i:]) // anchored search - will not run off
			// ignore indices leading to overlapping matches
			if m != nil && prev <= i {
				m[0] = i // correct m
				m[1] += i
				result = append(result, m)
				prev = m[1]
			}
		}
		if len(result) >= n || len(indices) != n1 {
			// found all matches or there's no chance to find more
			// (n and n1 can be negative)
			break
		}
	}
	if len(result) == 0 {
		result = nil
	}
	return
}
