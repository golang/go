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
	"container/vector"
	"sort"
)

// BUG(gri): For larger data (10MB) which contains very long (say 100000)
// contiguous sequences of identical bytes, index creation time will be extremely slow.

// TODO(gri): Use a more sophisticated algorithm to create the suffix array.


// Index implements a suffix array for fast substring search.
type Index struct {
	data []byte
	sa   []int // suffix array for data
}


// New creates a new Index for data.
// Index creation time is approximately O(N*log(N)) for N = len(data).
//
func New(data []byte) *Index {
	sa := make([]int, len(data))
	for i, _ := range sa {
		sa[i] = i
	}
	x := &Index{data, sa}
	sort.Sort((*index)(x))
	return x
}


func (x *Index) at(i int) []byte {
	return x.data[x.sa[i]:]
}


// Binary search according to "A Method of Programming", E.W. Dijkstra.
func (x *Index) search(s []byte) int {
	i, j := 0, len(x.sa)
	// i < j for non-empty x
	for i+1 < j {
		// 0 <= i < j <= len(x.sa) && (x.at(i) <= s < x.at(j) || (s is not in x))
		h := i + (j-i)/2 // i < h < j
		if bytes.Compare(x.at(h), s) <= 0 {
			i = h
		} else { // s < x.at(h)
			j = h
		}
	}
	// i+1 == j for non-empty x
	return i
}


// Lookup returns an unsorted list of at most n indices where the byte string s
// occurs in the indexed data. If n < 0, all occurrences are returned.
// The result is nil if s is empty, s is not found, or n == 0.
// Lookup time is O((log(N) + len(result))*len(s)) where N is the
// size of the indexed data.
//
func (x *Index) Lookup(s []byte, n int) []int {
	var res vector.IntVector

	if len(s) > 0 && n != 0 {
		// find matching suffix index i
		i := x.search(s)
		// x.at(i) <= s < x.at(i+1)

		// ignore the first suffix if it is < s
		if i < len(x.sa) && bytes.Compare(x.at(i), s) < 0 {
			i++
		}

		// collect the following suffixes with matching prefixes
		for (n < 0 || len(res) < n) && i < len(x.sa) && bytes.HasPrefix(x.at(i), s) {
			res.Push(x.sa[i])
			i++
		}
	}

	return res
}


// index is used to hide the sort.Interface
type index Index

func (x *index) Len() int           { return len(x.sa) }
func (x *index) Less(i, j int) bool { return bytes.Compare(x.at(i), x.at(j)) < 0 }
func (x *index) Swap(i, j int)      { x.sa[i], x.sa[j] = x.sa[j], x.sa[i] }
func (a *index) at(i int) []byte    { return a.data[a.sa[i]:] }
