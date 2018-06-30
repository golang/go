// Copyright 2010 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

// Package suffixarray implements substring search in logarithmic time using
// an in-memory suffix array.
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
	"encoding/binary"
	"io"
	"regexp"
	"sort"
)

// Index implements a suffix array for fast substring search.
type Index struct {
	data []byte
	sa   []int // suffix array for data; len(sa) == len(data)
}

// New creates a new Index for data.
// Index creation time is O(N*log(N)) for N = len(data).
func New(data []byte) *Index {
	return &Index{data, qsufsort(data)}
}

// writeInt writes an int x to w using buf to buffer the write.
func writeInt(w io.Writer, buf []byte, x int) error {
	binary.PutVarint(buf, int64(x))
	_, err := w.Write(buf[0:binary.MaxVarintLen64])
	return err
}

// readInt reads an int x from r using buf to buffer the read and returns x.
func readInt(r io.Reader, buf []byte) (int, error) {
	_, err := io.ReadFull(r, buf[0:binary.MaxVarintLen64]) // ok to continue with error
	x, _ := binary.Varint(buf)
	return int(x), err
}

// writeSlice writes data[:n] to w and returns n.
// It uses buf to buffer the write.
func writeSlice(w io.Writer, buf []byte, data []int) (n int, err error) {
	// encode as many elements as fit into buf
	p := binary.MaxVarintLen64
	for ; n < len(data) && p+binary.MaxVarintLen64 <= len(buf); n++ {
		p += binary.PutUvarint(buf[p:], uint64(data[n]))
	}

	// update buffer size
	binary.PutVarint(buf, int64(p))

	// write buffer
	_, err = w.Write(buf[0:p])
	return
}

// readSlice reads data[:n] from r and returns n.
// It uses buf to buffer the read.
func readSlice(r io.Reader, buf []byte, data []int) (n int, err error) {
	// read buffer size
	var size int
	size, err = readInt(r, buf)
	if err != nil {
		return
	}

	// read buffer w/o the size
	if _, err = io.ReadFull(r, buf[binary.MaxVarintLen64:size]); err != nil {
		return
	}

	// decode as many elements as present in buf
	for p := binary.MaxVarintLen64; p < size; n++ {
		x, w := binary.Uvarint(buf[p:])
		data[n] = int(x)
		p += w
	}

	return
}

const bufSize = 16 << 10 // reasonable for BenchmarkSaveRestore

// Read reads the index from r into x; x must not be nil.
func (x *Index) Read(r io.Reader) error {
	// buffer for all reads
	buf := make([]byte, bufSize)

	// read length
	n, err := readInt(r, buf)
	if err != nil {
		return err
	}

	// allocate space
	if 2*n < cap(x.data) || cap(x.data) < n {
		// new data is significantly smaller or larger than
		// existing buffers - allocate new ones
		x.data = make([]byte, n)
		x.sa = make([]int, n)
	} else {
		// re-use existing buffers
		x.data = x.data[0:n]
		x.sa = x.sa[0:n]
	}

	// read data
	if _, err := io.ReadFull(r, x.data); err != nil {
		return err
	}

	// read index
	for sa := x.sa; len(sa) > 0; {
		n, err := readSlice(r, buf, sa)
		if err != nil {
			return err
		}
		sa = sa[n:]
	}
	return nil
}

// Write writes the index x to w.
func (x *Index) Write(w io.Writer) error {
	// buffer for all writes
	buf := make([]byte, bufSize)

	// write length
	if err := writeInt(w, buf, len(x.data)); err != nil {
		return err
	}

	// write data
	if _, err := w.Write(x.data); err != nil {
		return err
	}

	// write index
	for sa := x.sa; len(sa) > 0; {
		n, err := writeSlice(w, buf, sa)
		if err != nil {
			return err
		}
		sa = sa[n:]
	}
	return nil
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

// lookupAll returns a slice into the matching region of the index.
// The runtime is O(log(N)*len(s)).
func (x *Index) lookupAll(s []byte) []int {
	// find matching suffix index range [i:j]
	// find the first index where s would be the prefix
	i := sort.Search(len(x.sa), func(i int) bool { return bytes.Compare(x.at(i), s) >= 0 })
	// starting at i, find the first index at which s is not a prefix
	j := i + sort.Search(len(x.sa)-i, func(j int) bool { return !bytes.HasPrefix(x.at(j+i), s) })
	return x.sa[i:j]
}

// Lookup returns an unsorted list of at most n indices where the byte string s
// occurs in the indexed data. If n < 0, all occurrences are returned.
// The result is nil if s is empty, s is not found, or n == 0.
// Lookup time is O(log(N)*len(s) + len(result)) where N is the
// size of the indexed data.
//
func (x *Index) Lookup(s []byte, n int) (result []int) {
	if len(s) > 0 && n != 0 {
		matches := x.lookupAll(s)
		if n < 0 || len(matches) < n {
			n = len(matches)
		}
		// 0 <= n <= len(matches)
		if n > 0 {
			result = make([]int, n)
			copy(result, matches)
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
			sort.Ints(indices)
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
		sort.Ints(indices)
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
