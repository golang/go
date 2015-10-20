// Copyright 2009 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

package dwarf_test

import (
	. "debug/dwarf"
	"testing"
)

func TestSplit(t *testing.T) {
	// debug/dwarf doesn't (currently) support split DWARF, but
	// the attributes that pointed to the split DWARF used to
	// cause loading the DWARF data to fail entirely (issue
	// #12592). Test that we can at least read the DWARF data.
	d := elfData(t, "testdata/split.elf")
	r := d.Reader()
	e, err := r.Next()
	if err != nil {
		t.Fatal(err)
	}
	if e.Tag != TagCompileUnit {
		t.Fatalf("bad tag: have %s, want %s", e.Tag, TagCompileUnit)
	}
	// Check that we were able to parse the unknown section offset
	// field, even if we can't figure out its DWARF class.
	const AttrGNUAddrBase Attr = 0x2133
	f := e.AttrField(AttrGNUAddrBase)
	if _, ok := f.Val.(int64); !ok {
		t.Fatalf("bad attribute value type: have %T, want int64", f.Val)
	}
	if f.Class != ClassUnknown {
		t.Fatalf("bad class: have %s, want %s", f.Class, ClassUnknown)
	}
}
