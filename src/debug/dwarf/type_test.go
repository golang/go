// Copyright 2009 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

package dwarf_test

import (
	. "debug/dwarf"
	"debug/elf"
	"debug/macho"
	"debug/pe"
	"fmt"
	"strconv"
	"testing"
)

var typedefTests = map[string]string{
	"t_ptr_volatile_int":                    "*volatile int",
	"t_ptr_const_char":                      "*const char",
	"t_long":                                "long int",
	"t_ushort":                              "short unsigned int",
	"t_func_int_of_float_double":            "func(float, double) int",
	"t_ptr_func_int_of_float_double":        "*func(float, double) int",
	"t_ptr_func_int_of_float_complex":       "*func(complex float) int",
	"t_ptr_func_int_of_double_complex":      "*func(complex double) int",
	"t_ptr_func_int_of_long_double_complex": "*func(complex long double) int",
	"t_func_ptr_int_of_char_schar_uchar":    "func(char, signed char, unsigned char) *int",
	"t_func_void_of_char":                   "func(char) void",
	"t_func_void_of_void":                   "func() void",
	"t_func_void_of_ptr_char_dots":          "func(*char, ...) void",
	"t_my_struct":                           "struct my_struct {vi volatile int@0; x char@4 : 1@7; y int@4 : 4@27; z [0]int@8; array [40]long long int@8; zz [0]int@328}",
	"t_my_struct1":                          "struct my_struct1 {zz [1]int@0}",
	"t_my_union":                            "union my_union {vi volatile int@0; x char@0 : 1@7; y int@0 : 4@28; array [40]long long int@0}",
	"t_my_enum":                             "enum my_enum {e1=1; e2=2; e3=-5; e4=1000000000000000}",
	"t_my_list":                             "struct list {val short int@0; next *t_my_list@8}",
	"t_my_tree":                             "struct tree {left *struct tree@0; right *struct tree@8; val long long unsigned int@16}",
}

// As Apple converts gcc to a clang-based front end
// they keep breaking the DWARF output. This map lists the
// conversion from real answer to Apple answer.
var machoBug = map[string]string{
	"func(*char, ...) void":                                 "func(*char) void",
	"enum my_enum {e1=1; e2=2; e3=-5; e4=1000000000000000}": "enum my_enum {e1=1; e2=2; e3=-5; e4=-1530494976}",
}

func elfData(t *testing.T, name string) *Data {
	f, err := elf.Open(name)
	if err != nil {
		t.Fatal(err)
	}

	d, err := f.DWARF()
	if err != nil {
		t.Fatal(err)
	}
	return d
}

func machoData(t *testing.T, name string) *Data {
	f, err := macho.Open(name)
	if err != nil {
		t.Fatal(err)
	}

	d, err := f.DWARF()
	if err != nil {
		t.Fatal(err)
	}
	return d
}

func peData(t *testing.T, name string) *Data {
	f, err := pe.Open(name)
	if err != nil {
		t.Fatal(err)
	}

	d, err := f.DWARF()
	if err != nil {
		t.Fatal(err)
	}
	return d
}

func TestTypedefsELF(t *testing.T) {
	testTypedefs(t, elfData(t, "testdata/typedef.elf"), "elf", typedefTests)
}

func TestTypedefsMachO(t *testing.T) {
	testTypedefs(t, machoData(t, "testdata/typedef.macho"), "macho", typedefTests)
}

func TestTypedefsELFDwarf4(t *testing.T) {
	testTypedefs(t, elfData(t, "testdata/typedef.elf4"), "elf", typedefTests)
}

func testTypedefs(t *testing.T, d *Data, kind string, testcases map[string]string) {
	r := d.Reader()
	seen := make(map[string]bool)
	for {
		e, err := r.Next()
		if err != nil {
			t.Fatal("r.Next:", err)
		}
		if e == nil {
			break
		}
		if e.Tag == TagTypedef {
			typ, err := d.Type(e.Offset)
			if err != nil {
				t.Fatal("d.Type:", err)
			}
			t1 := typ.(*TypedefType)
			var typstr string
			if ts, ok := t1.Type.(*StructType); ok {
				typstr = ts.Defn()
			} else {
				typstr = t1.Type.String()
			}

			if want, ok := testcases[t1.Name]; ok {
				if seen[t1.Name] {
					t.Errorf("multiple definitions for %s", t1.Name)
				}
				seen[t1.Name] = true
				if typstr != want && (kind != "macho" || typstr != machoBug[want]) {
					t.Errorf("%s:\n\thave %s\n\twant %s", t1.Name, typstr, want)
				}
			}
		}
		if e.Tag != TagCompileUnit {
			r.SkipChildren()
		}
	}

	for k := range testcases {
		if !seen[k] {
			t.Errorf("missing %s", k)
		}
	}
}

func TestTypedefCycle(t *testing.T) {
	// See issue #13039: reading a typedef cycle starting from a
	// different place than the size needed to be computed from
	// used to crash.
	//
	// cycle.elf built with GCC 4.8.4:
	//    gcc -g -c -o cycle.elf cycle.c
	d := elfData(t, "testdata/cycle.elf")
	r := d.Reader()
	offsets := []Offset{}
	for {
		e, err := r.Next()
		if err != nil {
			t.Fatal("r.Next:", err)
		}
		if e == nil {
			break
		}
		switch e.Tag {
		case TagBaseType, TagTypedef, TagPointerType, TagStructType:
			offsets = append(offsets, e.Offset)
		}
	}

	// Parse each type with a fresh type cache.
	for _, offset := range offsets {
		d := elfData(t, "testdata/cycle.elf")
		_, err := d.Type(offset)
		if err != nil {
			t.Fatalf("d.Type(0x%x): %s", offset, err)
		}
	}
}

var unsupportedTypeTests = []string{
	// varname:typename:string:size
	"culprit::(unsupported type ReferenceType):8",
	"pdm::(unsupported type PtrToMemberType):-1",
}

func TestUnsupportedTypes(t *testing.T) {
	// Issue 29601:
	// When reading DWARF from C++ load modules, we can encounter
	// oddball type DIEs. These will be returned as "UnsupportedType"
	// objects; check to make sure this works properly.
	d := elfData(t, "testdata/cppunsuptypes.elf")
	r := d.Reader()
	seen := make(map[string]bool)
	for {
		e, err := r.Next()
		if err != nil {
			t.Fatal("r.Next:", err)
		}
		if e == nil {
			break
		}
		if e.Tag == TagVariable {
			vname, _ := e.Val(AttrName).(string)
			tAttr := e.Val(AttrType)
			typOff, ok := tAttr.(Offset)
			if !ok {
				t.Errorf("variable at offset %v has no type", e.Offset)
				continue
			}
			typ, err := d.Type(typOff)
			if err != nil {
				t.Errorf("err in type decode: %v\n", err)
				continue
			}
			unsup, isok := typ.(*UnsupportedType)
			if !isok {
				continue
			}
			tag := vname + ":" + unsup.Name + ":" + unsup.String() +
				":" + strconv.FormatInt(unsup.Size(), 10)
			seen[tag] = true
		}
	}
	dumpseen := false
	for _, v := range unsupportedTypeTests {
		if !seen[v] {
			t.Errorf("missing %s", v)
			dumpseen = true
		}
	}
	if dumpseen {
		for k := range seen {
			fmt.Printf("seen: %s\n", k)
		}
	}
}

var expectedBitOffsets1 = map[string]string{
	"x": "S:1 DBO:32",
	"y": "S:4 DBO:33",
}

var expectedBitOffsets2 = map[string]string{
	"x": "S:1 BO:7",
	"y": "S:4 BO:27",
}

func TestBitOffsetsELF(t *testing.T) {
	f := "testdata/typedef.elf"
	testBitOffsets(t, elfData(t, f), f, expectedBitOffsets2)
}

func TestBitOffsetsMachO(t *testing.T) {
	f := "testdata/typedef.macho"
	testBitOffsets(t, machoData(t, f), f, expectedBitOffsets2)
}

func TestBitOffsetsMachO4(t *testing.T) {
	f := "testdata/typedef.macho4"
	testBitOffsets(t, machoData(t, f), f, expectedBitOffsets1)
}

func TestBitOffsetsELFDwarf4(t *testing.T) {
	f := "testdata/typedef.elf4"
	testBitOffsets(t, elfData(t, f), f, expectedBitOffsets1)
}

func TestBitOffsetsELFDwarf5(t *testing.T) {
	f := "testdata/typedef.elf5"
	testBitOffsets(t, elfData(t, f), f, expectedBitOffsets1)
}

func testBitOffsets(t *testing.T, d *Data, tag string, expectedBitOffsets map[string]string) {
	r := d.Reader()
	for {
		e, err := r.Next()
		if err != nil {
			t.Fatal("r.Next:", err)
		}
		if e == nil {
			break
		}

		if e.Tag == TagStructType {
			typ, err := d.Type(e.Offset)
			if err != nil {
				t.Fatal("d.Type:", err)
			}

			t1 := typ.(*StructType)

			bitInfoDump := func(f *StructField) string {
				res := fmt.Sprintf("S:%d", f.BitSize)
				if f.BitOffset != 0 {
					res += fmt.Sprintf(" BO:%d", f.BitOffset)
				}
				if f.DataBitOffset != 0 {
					res += fmt.Sprintf(" DBO:%d", f.DataBitOffset)
				}
				return res
			}

			for _, field := range t1.Field {
				// We're only testing for bitfields
				if field.BitSize == 0 {
					continue
				}
				got := bitInfoDump(field)
				want := expectedBitOffsets[field.Name]
				if got != want {
					t.Errorf("%s: field %s in %s: got info %q want %q", tag, field.Name, t1.StructName, got, want)
				}
			}
		}
		if e.Tag != TagCompileUnit {
			r.SkipChildren()
		}
	}
}

var bitfieldTests = map[string]string{
	"t_another_struct": "struct another_struct {quix short unsigned int@0; xyz [0]int@4; x unsigned int@4 : 1@31; array [40]long long int@8}",
}

// TestBitFieldZeroArrayIssue50685 checks to make sure that the DWARF
// type reading code doesn't get confused by the presence of a
// specifically-sized bitfield member immediately following a field
// whose type is a zero-length array. Prior to the fix for issue
// 50685, we would get this type for the case in testdata/bitfields.c:
//
// another_struct {quix short unsigned int@0; xyz [-1]int@4; x unsigned int@4 : 1@31; array [40]long long int@8}
//
// Note the "-1" for the xyz field, which should be zero.
func TestBitFieldZeroArrayIssue50685(t *testing.T) {
	f := "testdata/bitfields.elf4"
	testTypedefs(t, elfData(t, f), "elf", bitfieldTests)
}
