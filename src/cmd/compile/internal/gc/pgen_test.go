// Copyright 2015 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

package gc

import (
	"cmd/compile/internal/types"
	"reflect"
	"sort"
	"testing"
)

func typeWithoutPointers() *types.Type {
	t := types.New(TSTRUCT)
	f := &types.Field{Type: types.New(TINT)}
	t.SetFields([]*types.Field{f})
	return t
}

func typeWithPointers() *types.Type {
	t := types.New(TSTRUCT)
	f := &types.Field{Type: types.New(TPTR64)}
	t.SetFields([]*types.Field{f})
	return t
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
			Node{Class: PAUTO, flags: nodeUsed},
			Node{Class: PAUTO},
			true,
		},
		{
			Node{Class: PAUTO},
			Node{Class: PAUTO, flags: nodeUsed},
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
			Node{Class: PAUTO, Type: &types.Type{}, Name: &Name{flags: nameNeedzero}},
			Node{Class: PAUTO, Type: &types.Type{}, Name: &Name{}},
			true,
		},
		{
			Node{Class: PAUTO, Type: &types.Type{}, Name: &Name{}},
			Node{Class: PAUTO, Type: &types.Type{}, Name: &Name{flags: nameNeedzero}},
			false,
		},
		{
			Node{Class: PAUTO, Type: &types.Type{Width: 1}, Name: &Name{}},
			Node{Class: PAUTO, Type: &types.Type{Width: 2}, Name: &Name{}},
			false,
		},
		{
			Node{Class: PAUTO, Type: &types.Type{Width: 2}, Name: &Name{}},
			Node{Class: PAUTO, Type: &types.Type{Width: 1}, Name: &Name{}},
			true,
		},
		{
			Node{Class: PAUTO, Type: &types.Type{}, Name: &Name{}, Sym: &types.Sym{Name: "abc"}},
			Node{Class: PAUTO, Type: &types.Type{}, Name: &Name{}, Sym: &types.Sym{Name: "xyz"}},
			true,
		},
		{
			Node{Class: PAUTO, Type: &types.Type{}, Name: &Name{}, Sym: &types.Sym{Name: "abc"}},
			Node{Class: PAUTO, Type: &types.Type{}, Name: &Name{}, Sym: &types.Sym{Name: "abc"}},
			false,
		},
		{
			Node{Class: PAUTO, Type: &types.Type{}, Name: &Name{}, Sym: &types.Sym{Name: "xyz"}},
			Node{Class: PAUTO, Type: &types.Type{}, Name: &Name{}, Sym: &types.Sym{Name: "abc"}},
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
		{Class: PFUNC, Type: &types.Type{}, Name: &Name{}, Sym: &types.Sym{}},
		{Class: PAUTO, Type: &types.Type{}, Name: &Name{}, Sym: &types.Sym{}},
		{Class: PFUNC, Xoffset: 0, Type: &types.Type{}, Name: &Name{}, Sym: &types.Sym{}},
		{Class: PFUNC, Xoffset: 10, Type: &types.Type{}, Name: &Name{}, Sym: &types.Sym{}},
		{Class: PFUNC, Xoffset: 20, Type: &types.Type{}, Name: &Name{}, Sym: &types.Sym{}},
		{Class: PAUTO, flags: nodeUsed, Type: &types.Type{}, Name: &Name{}, Sym: &types.Sym{}},
		{Class: PAUTO, Type: typeWithoutPointers(), Name: &Name{}, Sym: &types.Sym{}},
		{Class: PAUTO, Type: &types.Type{}, Name: &Name{}, Sym: &types.Sym{}},
		{Class: PAUTO, Type: &types.Type{}, Name: &Name{flags: nameNeedzero}, Sym: &types.Sym{}},
		{Class: PAUTO, Type: &types.Type{Width: 1}, Name: &Name{}, Sym: &types.Sym{}},
		{Class: PAUTO, Type: &types.Type{Width: 2}, Name: &Name{}, Sym: &types.Sym{}},
		{Class: PAUTO, Type: &types.Type{}, Name: &Name{}, Sym: &types.Sym{Name: "abc"}},
		{Class: PAUTO, Type: &types.Type{}, Name: &Name{}, Sym: &types.Sym{Name: "xyz"}},
	}
	want := []*Node{
		{Class: PFUNC, Type: &types.Type{}, Name: &Name{}, Sym: &types.Sym{}},
		{Class: PFUNC, Xoffset: 0, Type: &types.Type{}, Name: &Name{}, Sym: &types.Sym{}},
		{Class: PFUNC, Xoffset: 10, Type: &types.Type{}, Name: &Name{}, Sym: &types.Sym{}},
		{Class: PFUNC, Xoffset: 20, Type: &types.Type{}, Name: &Name{}, Sym: &types.Sym{}},
		{Class: PAUTO, flags: nodeUsed, Type: &types.Type{}, Name: &Name{}, Sym: &types.Sym{}},
		{Class: PAUTO, Type: &types.Type{}, Name: &Name{flags: nameNeedzero}, Sym: &types.Sym{}},
		{Class: PAUTO, Type: &types.Type{Width: 2}, Name: &Name{}, Sym: &types.Sym{}},
		{Class: PAUTO, Type: &types.Type{Width: 1}, Name: &Name{}, Sym: &types.Sym{}},
		{Class: PAUTO, Type: &types.Type{}, Name: &Name{}, Sym: &types.Sym{}},
		{Class: PAUTO, Type: &types.Type{}, Name: &Name{}, Sym: &types.Sym{}},
		{Class: PAUTO, Type: &types.Type{}, Name: &Name{}, Sym: &types.Sym{Name: "abc"}},
		{Class: PAUTO, Type: &types.Type{}, Name: &Name{}, Sym: &types.Sym{Name: "xyz"}},
		{Class: PAUTO, Type: typeWithoutPointers(), Name: &Name{}, Sym: &types.Sym{}},
	}
	// haspointers updates Type.Haspointers as a side effect, so
	// exercise this function on all inputs so that reflect.DeepEqual
	// doesn't produce false positives.
	for i := range want {
		types.Haspointers(want[i].Type)
		types.Haspointers(inp[i].Type)
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
