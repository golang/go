// Copyright 2009 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

package dwarf_test

import (
	. "debug/dwarf"
	"reflect"
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

// wantRange maps from a PC to the ranges of the compilation unit
// containing that PC.
type wantRange struct {
	pc     uint64
	ranges [][2]uint64
}

func TestReaderSeek(t *testing.T) {
	want := []wantRange{
		{0x40059d, [][2]uint64{{0x40059d, 0x400601}}},
		{0x400600, [][2]uint64{{0x40059d, 0x400601}}},
		{0x400601, [][2]uint64{{0x400601, 0x400611}}},
		{0x4005f0, [][2]uint64{{0x40059d, 0x400601}}}, // loop test
		{0x10, nil},
		{0x400611, nil},
	}
	testRanges(t, "testdata/line-gcc.elf", want)
}

func TestRangesSection(t *testing.T) {
	want := []wantRange{
		{0x400500, [][2]uint64{{0x400500, 0x400549}, {0x400400, 0x400408}}},
		{0x400400, [][2]uint64{{0x400500, 0x400549}, {0x400400, 0x400408}}},
		{0x400548, [][2]uint64{{0x400500, 0x400549}, {0x400400, 0x400408}}},
		{0x400407, [][2]uint64{{0x400500, 0x400549}, {0x400400, 0x400408}}},
		{0x400408, nil},
		{0x400449, nil},
		{0x4003ff, nil},
	}
	testRanges(t, "testdata/ranges.elf", want)
}

func testRanges(t *testing.T, name string, want []wantRange) {
	d := elfData(t, name)
	r := d.Reader()
	for _, w := range want {
		entry, err := r.SeekPC(w.pc)
		if err != nil {
			if w.ranges != nil {
				t.Errorf("%s: missing Entry for %#x", name, w.pc)
			}
			if err != ErrUnknownPC {
				t.Errorf("%s: expected ErrUnknownPC for %#x, got %v", name, w.pc, err)
			}
			continue
		}

		ranges, err := d.Ranges(entry)
		if err != nil {
			t.Errorf("%s: %v", name, err)
			continue
		}
		if !reflect.DeepEqual(ranges, w.ranges) {
			t.Errorf("%s: for %#x got %x, expected %x", name, w.pc, ranges, w.ranges)
		}
	}
}

func TestReaderRanges(t *testing.T) {
	d := elfData(t, "testdata/line-gcc.elf")

	subprograms := []struct {
		name   string
		ranges [][2]uint64
	}{
		{"f1", [][2]uint64{{0x40059d, 0x4005e7}}},
		{"main", [][2]uint64{{0x4005e7, 0x400601}}},
		{"f2", [][2]uint64{{0x400601, 0x400611}}},
	}

	r := d.Reader()
	i := 0
	for entry, err := r.Next(); entry != nil && err == nil; entry, err = r.Next() {
		if entry.Tag != TagSubprogram {
			continue
		}

		if i > len(subprograms) {
			t.Fatalf("too many subprograms (expected at most %d)", i)
		}

		if got := entry.Val(AttrName).(string); got != subprograms[i].name {
			t.Errorf("subprogram %d name is %s, expected %s", i, got, subprograms[i].name)
		}
		ranges, err := d.Ranges(entry)
		if err != nil {
			t.Errorf("subprogram %d: %v", i, err)
			continue
		}
		if !reflect.DeepEqual(ranges, subprograms[i].ranges) {
			t.Errorf("subprogram %d ranges are %x, expected %x", i, ranges, subprograms[i].ranges)
		}
		i++
	}

	if i < len(subprograms) {
		t.Errorf("saw only %d subprograms, expected %d", i, len(subprograms))
	}
}

func Test64Bit(t *testing.T) {
	// I don't know how to generate a 64-bit DWARF debug
	// compilation unit except by using XCOFF, so this is
	// hand-written.
	tests := []struct {
		name string
		info []byte
	}{
		{
			"32-bit little",
			[]byte{0x30, 0, 0, 0, // comp unit length
				4, 0, // DWARF version 4
				0, 0, 0, 0, // abbrev offset
				8, // address size
				0,
				0, 0, 0, 0, 0, 0, 0, 0,
				0, 0, 0, 0, 0, 0, 0, 0,
				0, 0, 0, 0, 0, 0, 0, 0,
				0, 0, 0, 0, 0, 0, 0, 0,
				0, 0, 0, 0, 0, 0, 0, 0,
			},
		},
		{
			"64-bit little",
			[]byte{0xff, 0xff, 0xff, 0xff, // 64-bit DWARF
				0x30, 0, 0, 0, 0, 0, 0, 0, // comp unit length
				4, 0, // DWARF version 4
				0, 0, 0, 0, 0, 0, 0, 0, // abbrev offset
				8, // address size
				0, 0, 0, 0, 0,
				0, 0, 0, 0, 0, 0, 0, 0,
				0, 0, 0, 0, 0, 0, 0, 0,
				0, 0, 0, 0, 0, 0, 0, 0,
				0, 0, 0, 0, 0, 0, 0, 0,
			},
		},
		{
			"64-bit big",
			[]byte{0xff, 0xff, 0xff, 0xff, // 64-bit DWARF
				0, 0, 0, 0, 0, 0, 0, 0x30, // comp unit length
				0, 4, // DWARF version 4
				0, 0, 0, 0, 0, 0, 0, 0, // abbrev offset
				8, // address size
				0, 0, 0, 0, 0,
				0, 0, 0, 0, 0, 0, 0, 0,
				0, 0, 0, 0, 0, 0, 0, 0,
				0, 0, 0, 0, 0, 0, 0, 0,
				0, 0, 0, 0, 0, 0, 0, 0,
			},
		},
	}

	for _, test := range tests {
		_, err := New(nil, nil, nil, test.info, nil, nil, nil, nil)
		if err != nil {
			t.Errorf("%s: %v", test.name, err)
		}
	}
}
