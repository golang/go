// Copyright 2016 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

package ld

import (
	"bytes"
	"cmd/internal/obj"
	"index/suffixarray"
	"sort"
	"strings"
)

// mergestrings merges all go.string.* character data into a single symbol.
//
// Combining string data symbols reduces the total binary size and
// makes deduplication possible.
func mergestrings() {
	if Buildmode == BuildmodeShared {
		return
	}

	strs := make([]*LSym, 0, 256)
	seenStr := make(map[string]bool, 256)         // symbol name -> in strs slice
	relocsToStrs := make(map[*LSym][]*Reloc, 256) // string -> relocation to string
	size := 0                                     // number of bytes in all strings

	// Collect strings and relocations that point to strings.
	for _, s := range Ctxt.Allsym {
		if !s.Attr.Reachable() || s.Attr.Special() {
			continue
		}
		for i := range s.R {
			r := &s.R[i]
			if r.Sym == nil {
				continue
			}
			if !seenStr[r.Sym.Name] {
				if !strings.HasPrefix(r.Sym.Name, "go.string.") {
					continue
				}
				if strings.HasPrefix(r.Sym.Name, "go.string.hdr") {
					continue
				}
				strs = append(strs, r.Sym)
				seenStr[r.Sym.Name] = true
				size += len(r.Sym.P)
			}
			relocsToStrs[r.Sym] = append(relocsToStrs[r.Sym], r)
		}
	}

	// Sort the strings, shortest first.
	//
	// Ordering by length lets us use the largest matching substring
	// index when there are multiple matches. This means we will not
	// use a substring of a string that we will later in the pass
	// mark as unreachable.
	sort.Sort(strSymsByLen(strs))

	// Build a suffix array.
	dataOff := make([]int, len(strs))
	data := make([]byte, 0, size)
	for i := range strs {
		dataOff[i] = len(data)
		data = append(data, strs[i].P...)
	}
	index := suffixarray.New(data)

	// Search for substring replacements.
	type replacement struct {
		str *LSym
		off int
	}
	replacements := make(map[*LSym]replacement)
	for i, s := range strs {
		results := index.Lookup(s.P, -1)
		if len(results) == 0 {
			continue
		}
		var res int
		for _, result := range results {
			if result > res {
				res = result
			}
		}
		var off int
		x := sort.SearchInts(dataOff, res)
		if x == len(dataOff) || dataOff[x] > res {
			x--
			off = res - dataOff[x]
		}
		if x == i {
			continue // found ourself
		}
		if len(s.P) > len(strs[x].P[off:]) {
			// Do not use substrings that match across strings.
			// In theory it is possible, but it would
			// complicate accounting for which future strings
			// are already used. It doesn't appear to be common
			// enough to do the extra work.
			continue
		}
		if off%Thearch.Minalign != 0 {
			continue // Cannot relcate to this substring.
		}
		replacements[s] = replacement{
			str: strs[x],
			off: off,
		}
	}

	// Put all string data into a single symbol and update the relocations.
	alldata := Linklookup(Ctxt, "go.string.alldata", 0)
	alldata.Type = obj.SGOSTRING
	alldata.Attr |= AttrReachable
	alldata.P = make([]byte, 0, size)
	for _, str := range strs {
		str.Attr.Set(AttrReachable, false)
		if rep, isReplaced := replacements[str]; isReplaced {
			// As strs is sorted, the replacement string
			// is always later in the strs range. Shift the
			// relocations to the replacement string symbol
			// and process then.
			relocs := relocsToStrs[rep.str]
			for _, r := range relocsToStrs[str] {
				r.Add += int64(rep.off)
				relocs = append(relocs, r)
			}
			relocsToStrs[rep.str] = relocs
			continue
		}

		off := len(alldata.P)
		alldata.P = append(alldata.P, str.P...)
		// Architectures with Minalign > 1 cannot have relocations pointing
		// to arbitrary locations, so make sure each string is appropriately
		// aligned.
		for r := len(alldata.P) % Thearch.Minalign; r > 0; r-- {
			alldata.P = append(alldata.P, 0)
		}
		for _, r := range relocsToStrs[str] {
			r.Add += int64(off)
			r.Sym = alldata
		}
	}
	alldata.Size = int64(len(alldata.P))
}

// strSymsByLen implements sort.Interface. It sorts *LSym by the length of P.
type strSymsByLen []*LSym

func (s strSymsByLen) Len() int      { return len(s) }
func (s strSymsByLen) Swap(i, j int) { s[i], s[j] = s[j], s[i] }
func (s strSymsByLen) Less(i, j int) bool {
	if len(s[i].P) == len(s[j].P) {
		return bytes.Compare(s[i].P, s[j].P) == -1
	}
	return len(s[i].P) < len(s[j].P)
}
