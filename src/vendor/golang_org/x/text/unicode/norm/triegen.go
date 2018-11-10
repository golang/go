// Copyright 2011 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

// +build ignore

// Trie table generator.
// Used by make*tables tools to generate a go file with trie data structures
// for mapping UTF-8 to a 16-bit value. All but the last byte in a UTF-8 byte
// sequence are used to lookup offsets in the index table to be used for the
// next byte. The last byte is used to index into a table with 16-bit values.

package main

import (
	"fmt"
	"io"
)

const maxSparseEntries = 16

type normCompacter struct {
	sparseBlocks [][]uint64
	sparseOffset []uint16
	sparseCount  int
	name         string
}

func mostFrequentStride(a []uint64) int {
	counts := make(map[int]int)
	var v int
	for _, x := range a {
		if stride := int(x) - v; v != 0 && stride >= 0 {
			counts[stride]++
		}
		v = int(x)
	}
	var maxs, maxc int
	for stride, cnt := range counts {
		if cnt > maxc || (cnt == maxc && stride < maxs) {
			maxs, maxc = stride, cnt
		}
	}
	return maxs
}

func countSparseEntries(a []uint64) int {
	stride := mostFrequentStride(a)
	var v, count int
	for _, tv := range a {
		if int(tv)-v != stride {
			if tv != 0 {
				count++
			}
		}
		v = int(tv)
	}
	return count
}

func (c *normCompacter) Size(v []uint64) (sz int, ok bool) {
	if n := countSparseEntries(v); n <= maxSparseEntries {
		return (n+1)*4 + 2, true
	}
	return 0, false
}

func (c *normCompacter) Store(v []uint64) uint32 {
	h := uint32(len(c.sparseOffset))
	c.sparseBlocks = append(c.sparseBlocks, v)
	c.sparseOffset = append(c.sparseOffset, uint16(c.sparseCount))
	c.sparseCount += countSparseEntries(v) + 1
	return h
}

func (c *normCompacter) Handler() string {
	return c.name + "Sparse.lookup"
}

func (c *normCompacter) Print(w io.Writer) (retErr error) {
	p := func(f string, x ...interface{}) {
		if _, err := fmt.Fprintf(w, f, x...); retErr == nil && err != nil {
			retErr = err
		}
	}

	ls := len(c.sparseBlocks)
	p("// %sSparseOffset: %d entries, %d bytes\n", c.name, ls, ls*2)
	p("var %sSparseOffset = %#v\n\n", c.name, c.sparseOffset)

	ns := c.sparseCount
	p("// %sSparseValues: %d entries, %d bytes\n", c.name, ns, ns*4)
	p("var %sSparseValues = [%d]valueRange {", c.name, ns)
	for i, b := range c.sparseBlocks {
		p("\n// Block %#x, offset %#x", i, c.sparseOffset[i])
		var v int
		stride := mostFrequentStride(b)
		n := countSparseEntries(b)
		p("\n{value:%#04x,lo:%#02x},", stride, uint8(n))
		for i, nv := range b {
			if int(nv)-v != stride {
				if v != 0 {
					p(",hi:%#02x},", 0x80+i-1)
				}
				if nv != 0 {
					p("\n{value:%#04x,lo:%#02x", nv, 0x80+i)
				}
			}
			v = int(nv)
		}
		if v != 0 {
			p(",hi:%#02x},", 0x80+len(b)-1)
		}
	}
	p("\n}\n\n")
	return
}
