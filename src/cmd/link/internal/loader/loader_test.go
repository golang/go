// Copyright 2019 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

package loader

import (
	"bytes"
	"cmd/internal/goobj"
	"cmd/internal/objabi"
	"cmd/internal/sys"
	"cmd/link/internal/sym"
	"fmt"
	"testing"
)

// dummyAddSym adds the named symbol to the loader as if it had been
// read from a Go object file. Note that it allocates a global
// index without creating an associated object reader, so one can't
// do anything interesting with this symbol (such as look at its
// data or relocations).
func addDummyObjSym(t *testing.T, ldr *Loader, or *oReader, name string) Sym {
	idx := uint32(len(ldr.objSyms))
	st := loadState{l: ldr}
	return st.addSym(name, 0, or, idx, nonPkgDef, &goobj.Sym{})
}

func mkLoader() *Loader {
	er := ErrorReporter{}
	ldr := NewLoader(0, &er)
	er.ldr = ldr
	return ldr
}

func TestAddMaterializedSymbol(t *testing.T) {
	ldr := mkLoader()
	dummyOreader := oReader{version: -1, syms: make([]Sym, 100)}
	or := &dummyOreader

	// Create some syms from a dummy object file symbol to get things going.
	ts1 := addDummyObjSym(t, ldr, or, "type:uint8")
	ts2 := addDummyObjSym(t, ldr, or, "mumble")
	ts3 := addDummyObjSym(t, ldr, or, "type:string")

	// Create some external symbols.
	es1 := ldr.LookupOrCreateSym("extnew1", 0)
	if es1 == 0 {
		t.Fatalf("LookupOrCreateSym failed for extnew1")
	}
	es1x := ldr.LookupOrCreateSym("extnew1", 0)
	if es1x != es1 {
		t.Fatalf("LookupOrCreateSym lookup: expected %d got %d for second lookup", es1, es1x)
	}
	es2 := ldr.LookupOrCreateSym("go:info.type.uint8", 0)
	if es2 == 0 {
		t.Fatalf("LookupOrCreateSym failed for go.info.type.uint8")
	}
	// Create a nameless symbol
	es3 := ldr.CreateStaticSym("")
	if es3 == 0 {
		t.Fatalf("CreateStaticSym failed for nameless sym")
	}

	// Grab symbol builder pointers
	sb1 := ldr.MakeSymbolUpdater(es1)
	sb2 := ldr.MakeSymbolUpdater(es2)
	sb3 := ldr.MakeSymbolUpdater(es3)

	// Suppose we create some more symbols, which triggers a grow.
	// Make sure the symbol builder's payload pointer is valid,
	// even across a grow.
	for i := 0; i < 9999; i++ {
		ldr.CreateStaticSym("dummy")
	}

	// Check get/set symbol type
	es3typ := sb3.Type()
	if es3typ != sym.Sxxx {
		t.Errorf("SymType(es3): expected %v, got %v", sym.Sxxx, es3typ)
	}
	sb3.SetType(sym.SRODATA)
	es3typ = sb3.Type()
	if es3typ != sym.SRODATA {
		t.Errorf("SymType(es3): expected %v, got %v", sym.SRODATA, es3typ)
	}
	es3typ = ldr.SymType(es3)
	if es3typ != sym.SRODATA {
		t.Errorf("SymType(es3): expected %v, got %v", sym.SRODATA, es3typ)
	}

	// New symbols should not initially be reachable.
	if ldr.AttrReachable(es1) || ldr.AttrReachable(es2) || ldr.AttrReachable(es3) {
		t.Errorf("newly materialized symbols should not be reachable")
	}

	// ... however it should be possible to set/unset their reachability.
	ldr.SetAttrReachable(es3, true)
	if !ldr.AttrReachable(es3) {
		t.Errorf("expected reachable symbol after update")
	}
	ldr.SetAttrReachable(es3, false)
	if ldr.AttrReachable(es3) {
		t.Errorf("expected unreachable symbol after update")
	}

	// Test expansion of attr bitmaps
	for idx := 0; idx < 36; idx++ {
		es := ldr.LookupOrCreateSym(fmt.Sprintf("zext%d", idx), 0)
		if ldr.AttrOnList(es) {
			t.Errorf("expected OnList after creation")
		}
		ldr.SetAttrOnList(es, true)
		if !ldr.AttrOnList(es) {
			t.Errorf("expected !OnList after update")
		}
		if ldr.AttrDuplicateOK(es) {
			t.Errorf("expected DupOK after creation")
		}
		ldr.SetAttrDuplicateOK(es, true)
		if !ldr.AttrDuplicateOK(es) {
			t.Errorf("expected !DupOK after update")
		}
	}

	sb1 = ldr.MakeSymbolUpdater(es1)
	sb2 = ldr.MakeSymbolUpdater(es2)

	// Get/set a few other attributes
	if ldr.AttrVisibilityHidden(es3) {
		t.Errorf("expected initially not hidden")
	}
	ldr.SetAttrVisibilityHidden(es3, true)
	if !ldr.AttrVisibilityHidden(es3) {
		t.Errorf("expected hidden after update")
	}

	// Test get/set symbol value.
	toTest := []Sym{ts2, es3}
	for i, s := range toTest {
		if v := ldr.SymValue(s); v != 0 {
			t.Errorf("ldr.Value(%d): expected 0 got %d\n", s, v)
		}
		nv := int64(i + 101)
		ldr.SetSymValue(s, nv)
		if v := ldr.SymValue(s); v != nv {
			t.Errorf("ldr.SetValue(%d,%d): expected %d got %d\n", s, nv, nv, v)
		}
	}

	// Check/set alignment
	es3al := ldr.SymAlign(es3)
	if es3al != 0 {
		t.Errorf("SymAlign(es3): expected 0, got %d", es3al)
	}
	ldr.SetSymAlign(es3, 128)
	es3al = ldr.SymAlign(es3)
	if es3al != 128 {
		t.Errorf("SymAlign(es3): expected 128, got %d", es3al)
	}

	// Add some relocations to the new symbols.
	r1, _ := sb1.AddRel(objabi.R_ADDR)
	r1.SetOff(0)
	r1.SetSiz(1)
	r1.SetSym(ts1)
	r2, _ := sb1.AddRel(objabi.R_CALL)
	r2.SetOff(3)
	r2.SetSiz(8)
	r2.SetSym(ts2)
	r3, _ := sb2.AddRel(objabi.R_USETYPE)
	r3.SetOff(7)
	r3.SetSiz(1)
	r3.SetSym(ts3)

	// Add some data to the symbols.
	d1 := []byte{1, 2, 3}
	d2 := []byte{4, 5, 6, 7}
	sb1.AddBytes(d1)
	sb2.AddBytes(d2)

	// Now invoke the usual loader interfaces to make sure
	// we're getting the right things back for these symbols.
	// First relocations...
	expRel := [][]Reloc{{r1, r2}, {r3}}
	for k, sb := range []*SymbolBuilder{sb1, sb2} {
		rsl := sb.Relocs()
		exp := expRel[k]
		if !sameRelocSlice(&rsl, exp) {
			t.Errorf("expected relocs %v, got %v", exp, rsl)
		}
	}

	// ... then data.
	dat := sb2.Data()
	if !bytes.Equal(dat, d2) {
		t.Errorf("expected es2 data %v, got %v", d2, dat)
	}

	// Nameless symbol should still be nameless.
	es3name := ldr.SymName(es3)
	if "" != es3name {
		t.Errorf("expected es3 name of '', got '%s'", es3name)
	}

	// Read value of materialized symbol.
	es1val := sb1.Value()
	if 0 != es1val {
		t.Errorf("expected es1 value of 0, got %v", es1val)
	}

	// Test other misc methods
	irm := ldr.IsReflectMethod(es1)
	if 0 != es1val {
		t.Errorf("expected IsReflectMethod(es1) value of 0, got %v", irm)
	}
}

func sameRelocSlice(s1 *Relocs, s2 []Reloc) bool {
	if s1.Count() != len(s2) {
		return false
	}
	for i := 0; i < s1.Count(); i++ {
		r1 := s1.At(i)
		r2 := &s2[i]
		if r1.Sym() != r2.Sym() ||
			r1.Type() != r2.Type() ||
			r1.Off() != r2.Off() ||
			r1.Add() != r2.Add() ||
			r1.Siz() != r2.Siz() {
			return false
		}
	}
	return true
}

type addFunc func(l *Loader, s Sym, s2 Sym) Sym

func mkReloc(l *Loader, typ objabi.RelocType, off int32, siz uint8, add int64, sym Sym) Reloc {
	r := Reloc{&goobj.Reloc{}, l.extReader, l}
	r.SetType(typ)
	r.SetOff(off)
	r.SetSiz(siz)
	r.SetAdd(add)
	r.SetSym(sym)
	return r
}

func TestAddDataMethods(t *testing.T) {
	ldr := mkLoader()
	dummyOreader := oReader{version: -1, syms: make([]Sym, 100)}
	or := &dummyOreader

	// Populate loader with some symbols.
	addDummyObjSym(t, ldr, or, "type:uint8")
	ldr.LookupOrCreateSym("hello", 0)

	arch := sys.ArchAMD64
	var testpoints = []struct {
		which       string
		addDataFunc addFunc
		expData     []byte
		expKind     sym.SymKind
		expRel      []Reloc
	}{
		{
			which: "AddUint8",
			addDataFunc: func(l *Loader, s Sym, _ Sym) Sym {
				sb := l.MakeSymbolUpdater(s)
				sb.AddUint8('a')
				return s
			},
			expData: []byte{'a'},
			expKind: sym.SDATA,
		},
		{
			which: "AddUintXX",
			addDataFunc: func(l *Loader, s Sym, _ Sym) Sym {
				sb := l.MakeSymbolUpdater(s)
				sb.AddUintXX(arch, 25185, 2)
				return s
			},
			expData: []byte{'a', 'b'},
			expKind: sym.SDATA,
		},
		{
			which: "SetUint8",
			addDataFunc: func(l *Loader, s Sym, _ Sym) Sym {
				sb := l.MakeSymbolUpdater(s)
				sb.AddUint8('a')
				sb.AddUint8('b')
				sb.SetUint8(arch, 1, 'c')
				return s
			},
			expData: []byte{'a', 'c'},
			expKind: sym.SDATA,
		},
		{
			which: "AddString",
			addDataFunc: func(l *Loader, s Sym, _ Sym) Sym {
				sb := l.MakeSymbolUpdater(s)
				sb.Addstring("hello")
				return s
			},
			expData: []byte{'h', 'e', 'l', 'l', 'o', 0},
			expKind: sym.SNOPTRDATA,
		},
		{
			which: "AddAddrPlus",
			addDataFunc: func(l *Loader, s Sym, s2 Sym) Sym {
				sb := l.MakeSymbolUpdater(s)
				sb.AddAddrPlus(arch, s2, 3)
				return s
			},
			expData: []byte{0, 0, 0, 0, 0, 0, 0, 0},
			expKind: sym.SDATA,
			expRel:  []Reloc{mkReloc(ldr, objabi.R_ADDR, 0, 8, 3, 6)},
		},
		{
			which: "AddAddrPlus4",
			addDataFunc: func(l *Loader, s Sym, s2 Sym) Sym {
				sb := l.MakeSymbolUpdater(s)
				sb.AddAddrPlus4(arch, s2, 3)
				return s
			},
			expData: []byte{0, 0, 0, 0},
			expKind: sym.SDATA,
			expRel:  []Reloc{mkReloc(ldr, objabi.R_ADDR, 0, 4, 3, 7)},
		},
		{
			which: "AddCURelativeAddrPlus",
			addDataFunc: func(l *Loader, s Sym, s2 Sym) Sym {
				sb := l.MakeSymbolUpdater(s)
				sb.AddCURelativeAddrPlus(arch, s2, 7)
				return s
			},
			expData: []byte{0, 0, 0, 0, 0, 0, 0, 0},
			expKind: sym.SDATA,
			expRel:  []Reloc{mkReloc(ldr, objabi.R_ADDRCUOFF, 0, 8, 7, 8)},
		},
		{
			which: "AddPEImageRelativeAddrPlus",
			addDataFunc: func(l *Loader, s Sym, s2 Sym) Sym {
				sb := l.MakeSymbolUpdater(s)
				sb.AddPEImageRelativeAddrPlus(arch, s2, 3)
				return s
			},
			expData: []byte{0, 0, 0, 0},
			expKind: sym.SDATA,
			expRel:  []Reloc{mkReloc(ldr, objabi.R_PEIMAGEOFF, 0, 4, 3, 9)},
		},
	}

	var pmi Sym
	for k, tp := range testpoints {
		name := fmt.Sprintf("new%d", k+1)
		mi := ldr.LookupOrCreateSym(name, 0)
		if mi == 0 {
			t.Fatalf("LookupOrCreateSym failed for '" + name + "'")
		}
		mi = tp.addDataFunc(ldr, mi, pmi)
		if ldr.SymType(mi) != tp.expKind {
			t.Errorf("testing Loader.%s: expected kind %s got %s",
				tp.which, tp.expKind, ldr.SymType(mi))
		}
		if !bytes.Equal(ldr.Data(mi), tp.expData) {
			t.Errorf("testing Loader.%s: expected data %v got %v",
				tp.which, tp.expData, ldr.Data(mi))
		}
		relocs := ldr.Relocs(mi)
		if !sameRelocSlice(&relocs, tp.expRel) {
			t.Fatalf("testing Loader.%s: got relocslice %+v wanted %+v",
				tp.which, relocs, tp.expRel)
		}
		pmi = mi
	}
}

func TestOuterSub(t *testing.T) {
	ldr := mkLoader()
	dummyOreader := oReader{version: -1, syms: make([]Sym, 100)}
	or := &dummyOreader

	// Populate loader with some symbols.
	addDummyObjSym(t, ldr, or, "type:uint8")
	es1 := ldr.LookupOrCreateSym("outer", 0)
	ldr.MakeSymbolUpdater(es1).SetSize(101)
	es2 := ldr.LookupOrCreateSym("sub1", 0)
	es3 := ldr.LookupOrCreateSym("sub2", 0)
	es4 := ldr.LookupOrCreateSym("sub3", 0)
	es5 := ldr.LookupOrCreateSym("sub4", 0)
	es6 := ldr.LookupOrCreateSym("sub5", 0)

	// Should not have an outer sym initially
	if ldr.OuterSym(es1) != 0 {
		t.Errorf("es1 outer sym set ")
	}
	if ldr.SubSym(es2) != 0 {
		t.Errorf("es2 outer sym set ")
	}

	// Establish first outer/sub relationship
	ldr.AddInteriorSym(es1, es2)
	if ldr.OuterSym(es1) != 0 {
		t.Errorf("ldr.OuterSym(es1) got %d wanted %d", ldr.OuterSym(es1), 0)
	}
	if ldr.OuterSym(es2) != es1 {
		t.Errorf("ldr.OuterSym(es2) got %d wanted %d", ldr.OuterSym(es2), es1)
	}
	if ldr.SubSym(es1) != es2 {
		t.Errorf("ldr.SubSym(es1) got %d wanted %d", ldr.SubSym(es1), es2)
	}
	if ldr.SubSym(es2) != 0 {
		t.Errorf("ldr.SubSym(es2) got %d wanted %d", ldr.SubSym(es2), 0)
	}

	// Establish second outer/sub relationship
	ldr.AddInteriorSym(es1, es3)
	if ldr.OuterSym(es1) != 0 {
		t.Errorf("ldr.OuterSym(es1) got %d wanted %d", ldr.OuterSym(es1), 0)
	}
	if ldr.OuterSym(es2) != es1 {
		t.Errorf("ldr.OuterSym(es2) got %d wanted %d", ldr.OuterSym(es2), es1)
	}
	if ldr.OuterSym(es3) != es1 {
		t.Errorf("ldr.OuterSym(es3) got %d wanted %d", ldr.OuterSym(es3), es1)
	}
	if ldr.SubSym(es1) != es3 {
		t.Errorf("ldr.SubSym(es1) got %d wanted %d", ldr.SubSym(es1), es3)
	}
	if ldr.SubSym(es3) != es2 {
		t.Errorf("ldr.SubSym(es3) got %d wanted %d", ldr.SubSym(es3), es2)
	}

	// Some more
	ldr.AddInteriorSym(es1, es4)
	ldr.AddInteriorSym(es1, es5)
	ldr.AddInteriorSym(es1, es6)

	// Set values.
	ldr.SetSymValue(es2, 7)
	ldr.SetSymValue(es3, 1)
	ldr.SetSymValue(es4, 13)
	ldr.SetSymValue(es5, 101)
	ldr.SetSymValue(es6, 3)

	// Sort
	news := ldr.SortSub(es1)
	if news != es3 {
		t.Errorf("ldr.SortSub leader got %d wanted %d", news, es3)
	}
	pv := int64(-1)
	count := 0
	for ss := ldr.SubSym(es1); ss != 0; ss = ldr.SubSym(ss) {
		v := ldr.SymValue(ss)
		if v <= pv {
			t.Errorf("ldr.SortSub sortfail at %d: val %d >= prev val %d",
				ss, v, pv)
		}
		pv = v
		count++
	}
	if count != 5 {
		t.Errorf("expected %d in sub list got %d", 5, count)
	}
}
