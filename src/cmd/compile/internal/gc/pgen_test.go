// Copyright 2015 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

package gc

import (
	"reflect"
	"sort"
	"testing"
)

func typeWithoutPointers() *Type {
	return &Type{Etype: TSTRUCT, Extra: &StructType{Haspointers: 1}} // haspointers -> false
}

func typeWithPointers() *Type {
	return &Type{Etype: TSTRUCT, Extra: &StructType{Haspointers: 2}} // haspointers -> true
}

// Test all code paths for cmpstackvarlt.
func TestCmpstackvar(t *testing.T) {
	testdata := []struct {
		a, b Node
		lt   bool
	}{
		{
			Node{Class: PAUTO},
			Node{Class: PFUNC},
			false,
		},
		{
			Node{Class: PFUNC},
			Node{Class: PAUTO},
			true,
		},
		{
			Node{Class: PFUNC, Xoffset: 0},
			Node{Class: PFUNC, Xoffset: 10},
			true,
		},
		{
			Node{Class: PFUNC, Xoffset: 20},
			Node{Class: PFUNC, Xoffset: 10},
			false,
		},
		{
			Node{Class: PFUNC, Xoffset: 10},
			Node{Class: PFUNC, Xoffset: 10},
			false,
		},
		{
			Node{Class: PPARAM, Xoffset: 10},
			Node{Class: PPARAMOUT, Xoffset: 20},
			true,
		},
		{
			Node{Class: PPARAMOUT, Xoffset: 10},
			Node{Class: PPARAM, Xoffset: 20},
			true,
		},
		{
			Node{Class: PAUTO, Used: true},
			Node{Class: PAUTO, Used: false},
			true,
		},
		{
			Node{Class: PAUTO, Used: false},
			Node{Class: PAUTO, Used: true},
			false,
		},
		{
			Node{Class: PAUTO, Type: typeWithoutPointers()},
			Node{Class: PAUTO, Type: typeWithPointers()},
			false,
		},
		{
			Node{Class: PAUTO, Type: typeWithPointers()},
			Node{Class: PAUTO, Type: typeWithoutPointers()},
			true,
		},
		{
			Node{Class: PAUTO, Type: &Type{}, Name: &Name{Needzero: true}},
			Node{Class: PAUTO, Type: &Type{}, Name: &Name{Needzero: false}},
			true,
		},
		{
			Node{Class: PAUTO, Type: &Type{}, Name: &Name{Needzero: false}},
			Node{Class: PAUTO, Type: &Type{}, Name: &Name{Needzero: true}},
			false,
		},
		{
			Node{Class: PAUTO, Type: &Type{Width: 1}, Name: &Name{}},
			Node{Class: PAUTO, Type: &Type{Width: 2}, Name: &Name{}},
			false,
		},
		{
			Node{Class: PAUTO, Type: &Type{Width: 2}, Name: &Name{}},
			Node{Class: PAUTO, Type: &Type{Width: 1}, Name: &Name{}},
			true,
		},
		{
			Node{Class: PAUTO, Type: &Type{}, Name: &Name{}, Sym: &Sym{Name: "abc"}},
			Node{Class: PAUTO, Type: &Type{}, Name: &Name{}, Sym: &Sym{Name: "xyz"}},
			true,
		},
		{
			Node{Class: PAUTO, Type: &Type{}, Name: &Name{}, Sym: &Sym{Name: "abc"}},
			Node{Class: PAUTO, Type: &Type{}, Name: &Name{}, Sym: &Sym{Name: "abc"}},
			false,
		},
		{
			Node{Class: PAUTO, Type: &Type{}, Name: &Name{}, Sym: &Sym{Name: "xyz"}},
			Node{Class: PAUTO, Type: &Type{}, Name: &Name{}, Sym: &Sym{Name: "abc"}},
			false,
		},
	}
	for _, d := range testdata {
		got := cmpstackvarlt(&d.a, &d.b)
		if got != d.lt {
			t.Errorf("want %#v < %#v", d.a, d.b)
		}
		// If we expect a < b to be true, check that b < a is false.
		if d.lt && cmpstackvarlt(&d.b, &d.a) {
			t.Errorf("unexpected %#v < %#v", d.b, d.a)
		}
	}
}

func TestStackvarSort(t *testing.T) {
	inp := []*Node{
		{Class: PFUNC, Type: &Type{}, Name: &Name{}, Sym: &Sym{}},
		{Class: PAUTO, Type: &Type{}, Name: &Name{}, Sym: &Sym{}},
		{Class: PFUNC, Xoffset: 0, Type: &Type{}, Name: &Name{}, Sym: &Sym{}},
		{Class: PFUNC, Xoffset: 10, Type: &Type{}, Name: &Name{}, Sym: &Sym{}},
		{Class: PFUNC, Xoffset: 20, Type: &Type{}, Name: &Name{}, Sym: &Sym{}},
		{Class: PAUTO, Used: true, Type: &Type{}, Name: &Name{}, Sym: &Sym{}},
		{Class: PAUTO, Type: typeWithoutPointers(), Name: &Name{}, Sym: &Sym{}},
		{Class: PAUTO, Type: &Type{}, Name: &Name{}, Sym: &Sym{}},
		{Class: PAUTO, Type: &Type{}, Name: &Name{Needzero: true}, Sym: &Sym{}},
		{Class: PAUTO, Type: &Type{Width: 1}, Name: &Name{}, Sym: &Sym{}},
		{Class: PAUTO, Type: &Type{Width: 2}, Name: &Name{}, Sym: &Sym{}},
		{Class: PAUTO, Type: &Type{}, Name: &Name{}, Sym: &Sym{Name: "abc"}},
		{Class: PAUTO, Type: &Type{}, Name: &Name{}, Sym: &Sym{Name: "xyz"}},
	}
	want := []*Node{
		{Class: PFUNC, Type: &Type{}, Name: &Name{}, Sym: &Sym{}},
		{Class: PFUNC, Xoffset: 0, Type: &Type{}, Name: &Name{}, Sym: &Sym{}},
		{Class: PFUNC, Xoffset: 10, Type: &Type{}, Name: &Name{}, Sym: &Sym{}},
		{Class: PFUNC, Xoffset: 20, Type: &Type{}, Name: &Name{}, Sym: &Sym{}},
		{Class: PAUTO, Used: true, Type: &Type{}, Name: &Name{}, Sym: &Sym{}},
		{Class: PAUTO, Type: &Type{}, Name: &Name{Needzero: true}, Sym: &Sym{}},
		{Class: PAUTO, Type: &Type{Width: 2}, Name: &Name{}, Sym: &Sym{}},
		{Class: PAUTO, Type: &Type{Width: 1}, Name: &Name{}, Sym: &Sym{}},
		{Class: PAUTO, Type: &Type{}, Name: &Name{}, Sym: &Sym{}},
		{Class: PAUTO, Type: &Type{}, Name: &Name{}, Sym: &Sym{}},
		{Class: PAUTO, Type: &Type{}, Name: &Name{}, Sym: &Sym{Name: "abc"}},
		{Class: PAUTO, Type: &Type{}, Name: &Name{}, Sym: &Sym{Name: "xyz"}},
		{Class: PAUTO, Type: typeWithoutPointers(), Name: &Name{}, Sym: &Sym{}},
	}
	// haspointers updates Type.Haspointers as a side effect, so
	// exercise this function on all inputs so that reflect.DeepEqual
	// doesn't produce false positives.
	for i := range want {
		haspointers(want[i].Type)
		haspointers(inp[i].Type)
	}

	sort.Sort(byStackVar(inp))
	if !reflect.DeepEqual(want, inp) {
		t.Error("sort failed")
		for i := range inp {
			g := inp[i]
			w := want[i]
			eq := reflect.DeepEqual(w, g)
			if !eq {
				t.Log(i, w, g)
			}
		}
	}
}
