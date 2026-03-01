// Copyright 2009 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

package gob

import (
	"bytes"
	"reflect"
	"sync"
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
	t, err := getBaseType(name, rt)
	if err != nil {
		panic("getTypeUnlocked: " + err.Error())
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
	newtyp := getTypeUnlocked("int", reflect.TypeFor[int]())
	if newtyp != tInt.gobType() {
		t.Errorf("reregistration of %s got new type", newtyp.string())
	}
	newtyp = getTypeUnlocked("uint", reflect.TypeFor[uint]())
	if newtyp != tUint.gobType() {
		t.Errorf("reregistration of %s got new type", newtyp.string())
	}
	newtyp = getTypeUnlocked("string", reflect.TypeFor[string]())
	if newtyp != tString.gobType() {
		t.Errorf("reregistration of %s got new type", newtyp.string())
	}
}

func TestArrayType(t *testing.T) {
	var a3 [3]int
	a3int := getTypeUnlocked("foo", reflect.TypeOf(a3))
	newa3int := getTypeUnlocked("bar", reflect.TypeOf(a3))
	if a3int != newa3int {
		t.Errorf("second registration of [3]int creates new type")
	}
	var a4 [4]int
	a4int := getTypeUnlocked("goo", reflect.TypeOf(a4))
	if a3int == a4int {
		t.Errorf("registration of [3]int creates same type as [4]int")
	}
	var b3 [3]bool
	a3bool := getTypeUnlocked("", reflect.TypeOf(b3))
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
	sint := getTypeUnlocked("slice", reflect.TypeOf(s))
	var news []int
	newsint := getTypeUnlocked("slice1", reflect.TypeOf(news))
	if sint != newsint {
		t.Errorf("second registration of []int creates new type")
	}
	var b []bool
	sbool := getTypeUnlocked("", reflect.TypeOf(b))
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
	mapStringInt := getTypeUnlocked("map", reflect.TypeOf(m))
	var newm map[string]int
	newMapStringInt := getTypeUnlocked("map1", reflect.TypeOf(newm))
	if mapStringInt != newMapStringInt {
		t.Errorf("second registration of map[string]int creates new type")
	}
	var b map[string]bool
	mapStringBool := getTypeUnlocked("", reflect.TypeOf(b))
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
	X string
}

// This structure has pointers and refers to itself, making it a good test case.
type Foo struct {
	A int
	B int32 // will become int
	C string
	D []byte
	E *float64    // will become float64
	F ****float64 // will become float64
	G *Bar
	H *Bar // should not interpolate the definition of Bar again
	I *Foo // will not explode
}

func TestStructType(t *testing.T) {
	sstruct := getTypeUnlocked("Foo", reflect.TypeFor[Foo]())
	str := sstruct.string()
	// If we can print it correctly, we built it correctly.
	expected := "Foo = struct { A int; B int; C string; D bytes; E float; F float; G Bar = struct { X string; }; H Bar; I Foo; }"
	if str != expected {
		t.Errorf("struct printed as %q; expected %q", str, expected)
	}
}

// Should be OK to register the same type multiple times, as long as they're
// at the same level of indirection.
func TestRegistration(t *testing.T) {
	type T struct{ a int }
	Register(new(T))
	Register(new(T))
}

type N1 struct{}
type N2 struct{}

// See comment in type.go/Register.
func TestRegistrationNaming(t *testing.T) {
	testCases := []struct {
		t    any
		name string
	}{
		{&N1{}, "*gob.N1"},
		{N2{}, "encoding/gob.N2"},
	}

	for _, tc := range testCases {
		Register(tc.t)

		tct := reflect.TypeOf(tc.t)
		ct, _ := nameToConcreteType.Load(tc.name)
		if ct != tct {
			t.Errorf("nameToConcreteType[%q] = %v, want %v", tc.name, ct, tct)
		}
		// concreteTypeToName is keyed off the base type.
		if tct.Kind() == reflect.Pointer {
			tct = tct.Elem()
		}
		if n, _ := concreteTypeToName.Load(tct); n != tc.name {
			t.Errorf("concreteTypeToName[%v] got %v, want %v", tct, n, tc.name)
		}
	}
}

func TestStressParallel(t *testing.T) {
	type T2 struct{ A int }
	c := make(chan bool)
	const N = 10
	for i := 0; i < N; i++ {
		go func() {
			p := new(T2)
			Register(p)
			b := new(bytes.Buffer)
			enc := NewEncoder(b)
			err := enc.Encode(p)
			if err != nil {
				t.Error("encoder fail:", err)
			}
			dec := NewDecoder(b)
			err = dec.Decode(p)
			if err != nil {
				t.Error("decoder fail:", err)
			}
			c <- true
		}()
	}
	for i := 0; i < N; i++ {
		<-c
	}
}

// Issue 23328. Note that this test name is known to cmd/dist/test.go.
func TestTypeRace(t *testing.T) {
	c := make(chan bool)
	var wg sync.WaitGroup
	for i := 0; i < 2; i++ {
		wg.Add(1)
		go func(i int) {
			defer wg.Done()
			var buf bytes.Buffer
			enc := NewEncoder(&buf)
			dec := NewDecoder(&buf)
			var x any
			switch i {
			case 0:
				x = &N1{}
			case 1:
				x = &N2{}
			default:
				t.Errorf("bad i %d", i)
				return
			}
			m := make(map[string]string)
			<-c
			if err := enc.Encode(x); err != nil {
				t.Error(err)
				return
			}
			if err := enc.Encode(x); err != nil {
				t.Error(err)
				return
			}
			if err := dec.Decode(&m); err == nil {
				t.Error("decode unexpectedly succeeded")
				return
			}
		}(i)
	}
	close(c)
	wg.Wait()
}

func TestRegisteredNames(t *testing.T) {
	var names []string
	for name, _ := range RegisteredTypes() {
		names = append(names, name)
	}
	if len(names) < 34 {
		t.Errorf("registered names should contains all primitive type")
	}
}
