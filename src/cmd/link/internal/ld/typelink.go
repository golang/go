// Copyright 2016 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

package ld

import (
	"sort"

	"cmd/internal/obj"
)

type byTypeStr []typelinkSortKey

type typelinkSortKey struct {
	TypeStr string
	Type    *Symbol
}

func (s byTypeStr) Less(i, j int) bool { return s[i].TypeStr < s[j].TypeStr }
func (s byTypeStr) Len() int           { return len(s) }
func (s byTypeStr) Swap(i, j int)      { s[i], s[j] = s[j], s[i] }

// typelink generates the typelink table which is used by reflect.typelinks().
// Types that should be added to the typelinks table are marked with the
// MakeTypelink attribute by the compiler.
func (ctxt *Link) typelink() {
	typelinks := byTypeStr{}
	for _, s := range ctxt.Syms.Allsym {
		if s.Attr.Reachable() && s.Attr.MakeTypelink() {
			typelinks = append(typelinks, typelinkSortKey{decodetypeStr(s), s})
		}
	}
	sort.Sort(typelinks)

	tl := ctxt.Syms.Lookup("runtime.typelink", 0)
	tl.Type = obj.STYPELINK
	tl.Attr |= AttrReachable | AttrLocal
	tl.Size = int64(4 * len(typelinks))
	tl.P = make([]byte, tl.Size)
	tl.R = make([]Reloc, len(typelinks))
	for i, s := range typelinks {
		r := &tl.R[i]
		r.Sym = s.Type
		r.Off = int32(i * 4)
		r.Siz = 4
		r.Type = obj.R_ADDROFF
	}
}
