// Copyright 2016 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

package ld

import (
	"cmd/internal/objabi"
	"cmd/link/internal/loader"
	"cmd/link/internal/sym"
	"sort"
)

type byTypeStr []typelinkSortKey

type typelinkSortKey struct {
	TypeStr string
	Type    loader.Sym
}

func (s byTypeStr) Less(i, j int) bool { return s[i].TypeStr < s[j].TypeStr }
func (s byTypeStr) Len() int           { return len(s) }
func (s byTypeStr) Swap(i, j int)      { s[i], s[j] = s[j], s[i] }

// typelink generates the typelink table which is used by reflect.typelinks().
// Types that should be added to the typelinks table are marked with the
// MakeTypelink attribute by the compiler.
func (ctxt *Link) typelink() {
	ldr := ctxt.loader
	typelinks := byTypeStr{}
	for s := loader.Sym(1); s < loader.Sym(ldr.NSym()); s++ {
		if ldr.AttrReachable(s) && ldr.IsTypelink(s) {
			typelinks = append(typelinks, typelinkSortKey{decodetypeStr(ldr, ctxt.Arch, s), s})
		}
	}
	sort.Sort(typelinks)

	tl := ldr.CreateSymForUpdate("runtime.typelink", 0)
	tl.SetType(sym.STYPELINK)
	ldr.SetAttrReachable(tl.Sym(), true)
	ldr.SetAttrLocal(tl.Sym(), true)
	tl.SetSize(int64(4 * len(typelinks)))
	tl.Grow(tl.Size())
	relocs := tl.AddRelocs(len(typelinks))
	for i, s := range typelinks {
		r := relocs.At2(i)
		r.SetSym(s.Type)
		r.SetOff(int32(i * 4))
		r.SetSiz(4)
		r.SetType(objabi.R_ADDROFF)
	}
}
