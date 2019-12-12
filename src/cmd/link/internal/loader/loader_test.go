// Copyright 2019 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

package loader

import (
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
	idx := ldr.max + 1
	ldr.max++
	if ok := ldr.AddSym(name, 0, idx, or, false, sym.SRODATA); !ok {
		t.Errorf("AddrSym failed for '" + name + "'")
	}

	return idx
}

func TestAddMaterializedSymbol(t *testing.T) {
	ldr := NewLoader(0)
	dummyOreader := oReader{version: -1}
	or := &dummyOreader

	// Create some syms from a dummy object file symbol to get things going.
	addDummyObjSym(t, ldr, or, "type.uint8")
	ts2 := addDummyObjSym(t, ldr, or, "mumble")
	addDummyObjSym(t, ldr, or, "type.string")

	// Create some external symbols.
	es1 := ldr.AddExtSym("extnew1", 0)
	if es1 == 0 {
		t.Fatalf("AddExtSym failed for extnew1")
	}
	es1x := ldr.AddExtSym("extnew1", 0)
	if es1x != 0 {
		t.Fatalf("AddExtSym lookup: expected 0 got %d for second lookup", es1x)
	}
	es2 := ldr.AddExtSym("go.info.type.uint8", 0)
	if es2 == 0 {
		t.Fatalf("AddExtSym failed for go.info.type.uint8")
	}
	// Create a nameless symbol
	es3 := ldr.CreateExtSym("")
	if es3 == 0 {
		t.Fatalf("CreateExtSym failed for nameless sym")
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
		es := ldr.AddExtSym(fmt.Sprintf("zext%d", idx), 0)
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
}

func TestOuterSub(t *testing.T) {
	ldr := NewLoader(0)
	dummyOreader := oReader{version: -1}
	or := &dummyOreader

	// Populate loader with some symbols.
	addDummyObjSym(t, ldr, or, "type.uint8")
	es1 := ldr.AddExtSym("outer", 0)
	es2 := ldr.AddExtSym("sub1", 0)
	es3 := ldr.AddExtSym("sub2", 0)
	es4 := ldr.AddExtSym("sub3", 0)
	es5 := ldr.AddExtSym("sub4", 0)
	es6 := ldr.AddExtSym("sub5", 0)

	// Should not have an outer sym initially
	if ldr.OuterSym(es1) != 0 {
		t.Errorf("es1 outer sym set ")
	}
	if ldr.SubSym(es2) != 0 {
		t.Errorf("es2 outer sym set ")
	}

	// Establish first outer/sub relationship
	ldr.PrependSub(es1, es2)
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
	ldr.PrependSub(es1, es3)
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
	ldr.PrependSub(es1, es4)
	ldr.PrependSub(es1, es5)
	ldr.PrependSub(es1, es6)

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
