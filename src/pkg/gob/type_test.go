// Copyright 2009 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

package gob

import (
	"reflect"
	"testing"
)

type typeT struct {
	id  typeId
	str string
}

var basicTypes = []typeT{
	{tBool, "bool"},
	{tInt, "int"},
	{tUint, "uint"},
	{tFloat, "float"},
	{tBytes, "bytes"},
	{tString, "string"},
}

func getTypeUnlocked(name string, rt reflect.Type) gobType {
	typeLock.Lock()
	defer typeLock.Unlock()
	t, err := getType(name, rt)
	if err != nil {
		panic("getTypeUnlocked: " + err.String())
	}
	return t
}

// Sanity checks
func TestBasic(t *testing.T) {
	for _, tt := range basicTypes {
		if tt.id.string() != tt.str {
			t.Errorf("checkType: expected %q got %s", tt.str, tt.id.string())
		}
		if tt.id == 0 {
			t.Errorf("id for %q is zero", tt.str)
		}
	}
}

// Reregister some basic types to check registration is idempotent.
func TestReregistration(t *testing.T) {
	newtyp := getTypeUnlocked("int", reflect.Typeof(int(0)))
	if newtyp != tInt.gobType() {
		t.Errorf("reregistration of %s got new type", newtyp.string())
	}
	newtyp = getTypeUnlocked("uint", reflect.Typeof(uint(0)))
	if newtyp != tUint.gobType() {
		t.Errorf("reregistration of %s got new type", newtyp.string())
	}
	newtyp = getTypeUnlocked("string", reflect.Typeof("hello"))
	if newtyp != tString.gobType() {
		t.Errorf("reregistration of %s got new type", newtyp.string())
	}
}

func TestArrayType(t *testing.T) {
	var a3 [3]int
	a3int := getTypeUnlocked("foo", reflect.Typeof(a3))
	newa3int := getTypeUnlocked("bar", reflect.Typeof(a3))
	if a3int != newa3int {
		t.Errorf("second registration of [3]int creates new type")
	}
	var a4 [4]int
	a4int := getTypeUnlocked("goo", reflect.Typeof(a4))
	if a3int == a4int {
		t.Errorf("registration of [3]int creates same type as [4]int")
	}
	var b3 [3]bool
	a3bool := getTypeUnlocked("", reflect.Typeof(b3))
	if a3int == a3bool {
		t.Errorf("registration of [3]bool creates same type as [3]int")
	}
	str := a3bool.string()
	expected := "[3]bool"
	if str != expected {
		t.Errorf("array printed as %q; expected %q", str, expected)
	}
}

func TestSliceType(t *testing.T) {
	var s []int
	sint := getTypeUnlocked("slice", reflect.Typeof(s))
	var news []int
	newsint := getTypeUnlocked("slice1", reflect.Typeof(news))
	if sint != newsint {
		t.Errorf("second registration of []int creates new type")
	}
	var b []bool
	sbool := getTypeUnlocked("", reflect.Typeof(b))
	if sbool == sint {
		t.Errorf("registration of []bool creates same type as []int")
	}
	str := sbool.string()
	expected := "[]bool"
	if str != expected {
		t.Errorf("slice printed as %q; expected %q", str, expected)
	}
}

func TestMapType(t *testing.T) {
	var m map[string]int
	mapStringInt := getTypeUnlocked("map", reflect.Typeof(m))
	var newm map[string]int
	newMapStringInt := getTypeUnlocked("map1", reflect.Typeof(newm))
	if mapStringInt != newMapStringInt {
		t.Errorf("second registration of map[string]int creates new type")
	}
	var b map[string]bool
	mapStringBool := getTypeUnlocked("", reflect.Typeof(b))
	if mapStringBool == mapStringInt {
		t.Errorf("registration of map[string]bool creates same type as map[string]int")
	}
	str := mapStringBool.string()
	expected := "map[string]bool"
	if str != expected {
		t.Errorf("map printed as %q; expected %q", str, expected)
	}
}

type Bar struct {
	x string
}

// This structure has pointers and refers to itself, making it a good test case.
type Foo struct {
	a int
	b int32 // will become int
	c string
	d []byte
	e *float64    // will become float64
	f ****float64 // will become float64
	g *Bar
	h *Bar // should not interpolate the definition of Bar again
	i *Foo // will not explode
}

func TestStructType(t *testing.T) {
	sstruct := getTypeUnlocked("Foo", reflect.Typeof(Foo{}))
	str := sstruct.string()
	// If we can print it correctly, we built it correctly.
	expected := "Foo = struct { a int; b int; c string; d bytes; e float; f float; g Bar = struct { x string; }; h Bar; i Foo; }"
	if str != expected {
		t.Errorf("struct printed as %q; expected %q", str, expected)
	}
}
