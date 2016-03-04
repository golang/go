// Copyright 2016 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

package ld

import (
	"cmd/internal/obj"
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

	// Put all string data into a single symbol and update the relocations.
	alldata := Linklookup(Ctxt, "go.string.alldata", 0)
	alldata.Type = obj.SGOSTRING
	alldata.Attr |= AttrReachable
	alldata.P = make([]byte, 0, size)
	for _, str := range strs {
		off := len(alldata.P)
		alldata.P = append(alldata.P, str.P...)
		// Architectures with Minalign > 1 cannot have relocations pointing
		// to arbitrary locations, so make sure each string is appropriately
		// aligned.
		for r := len(alldata.P) % Thearch.Minalign; r > 0; r-- {
			alldata.P = append(alldata.P, 0)
		}
		str.Attr.Set(AttrReachable, false)
		for _, r := range relocsToStrs[str] {
			r.Add += int64(off)
			r.Sym = alldata
		}
	}
	alldata.Size = int64(len(alldata.P))
}
