// Copyright 2009 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

package gob

import (
"fmt";
	"gob";
	"os";
	"testing";
)

func checkType(ti Type, expected string, t *testing.T) {
	if ti.String() != expected {
		t.Errorf("checkType: expected %q got %s", expected, ti.String())
	}
	if ti.id() == 0 {
		t.Errorf("id for %q is zero", expected)
	}
}

type typeT struct {
	typ	Type;
	str	string;
}
var basicTypes = []typeT {
	typeT { tBool, "bool" },
	typeT { tInt, "int" },
	typeT { tUint, "uint" },
	typeT { tFloat32, "float32" },
	typeT { tFloat64, "float64" },
	typeT { tString, "string" },
}

// Sanity checks
func TestBasic(t *testing.T) {
	for _, tt := range basicTypes {
		if tt.typ.String() != tt.str {
			t.Errorf("checkType: expected %q got %s", tt.str, tt.typ.String())
		}
		if tt.typ.id() == 0 {
			t.Errorf("id for %q is zero", tt.str)
		}
	}
}

// Reregister some basic types to check registration is idempotent.
func TestReregistration(t *testing.T) {
	newtyp := GetType("int", 0);
	if newtyp != tInt {
		t.Errorf("reregistration of %s got new type", newtyp.String())
	}
	newtyp = GetType("uint", uint(0));
	if newtyp != tUint {
		t.Errorf("reregistration of %s got new type", newtyp.String())
	}
	newtyp = GetType("string", "hello");
	if newtyp != tString {
		t.Errorf("reregistration of %s got new type", newtyp.String())
	}
}

func TestArrayType(t *testing.T) {
	var a3 [3]int;
	a3int := GetType("foo", a3);
	var newa3 [3]int;
	newa3int := GetType("bar", a3);
	if a3int != newa3int {
		t.Errorf("second registration of [3]int creates new type");
	}
	var a4 [4]int;
	a4int := GetType("goo", a4);
	if a3int == a4int {
		t.Errorf("registration of [3]int creates same type as [4]int");
	}
	var b3 [3]bool;
	a3bool := GetType("", b3);
	if a3int == a3bool {
		t.Errorf("registration of [3]bool creates same type as [3]int");
	}
	str := a3bool.String();
	expected := "[3]bool";
	if str != expected {
		t.Errorf("array printed as %q; expected %q", str, expected);
	}
}

func TestSliceType(t *testing.T) {
	var s []int;
	sint := GetType("slice", s);
	var news []int;
	newsint := GetType("slice1", news);
	if sint != newsint {
		t.Errorf("second registration of []int creates new type");
	}
	var b []bool;
	sbool := GetType("", b);
	if sbool == sint {
		t.Errorf("registration of []bool creates same type as []int");
	}
	str := sbool.String();
	expected := "[]bool";
	if str != expected {
		t.Errorf("slice printed as %q; expected %q", str, expected);
	}
}

type Bar struct {
	x string
}

// This structure has pointers and refers to itself, making it a good test case.
type Foo struct {
	a int;
	b int32;	// will become int
	c string;
	d *float;	// will become float32
	e ****float64;	// will become float64
	f *Bar;
	g *Foo;	// will not explode
}

func TestStructType(t *testing.T) {
	sstruct := GetType("Foo", Foo{});
	str := sstruct.String();
	// If we can print it correctly, we built it correctly.
	expected := "struct { a int; b int; c string; d float32; e float64; f struct { x string; }; g Foo; }";
	if str != expected {
		t.Errorf("struct printed as %q; expected %q", str, expected);
	}
}
