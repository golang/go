// Copyright 2019 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

package loader

import (
	"cmd/link/internal/sym"
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
	addDummyObjSym(t, ldr, or, "mumble")
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
}
