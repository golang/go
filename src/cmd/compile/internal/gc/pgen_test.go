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

func nodeWithClass(n Node, c Class) *Node {
	n.SetClass(c)
	return &n
}

// Test all code paths for cmpstackvarlt.
func TestCmpstackvar(t *testing.T) {
	testdata := []struct {
		a, b *Node
		lt   bool
	}{
		{
			nodeWithClass(Node{}, PAUTO),
			nodeWithClass(Node{}, PFUNC),
			false,
		},
		{
			nodeWithClass(Node{}, PFUNC),
			nodeWithClass(Node{}, PAUTO),
			true,
		},
		{
			nodeWithClass(Node{Xoffset: 0}, PFUNC),
			nodeWithClass(Node{Xoffset: 10}, PFUNC),
			true,
		},
		{
			nodeWithClass(Node{Xoffset: 20}, PFUNC),
			nodeWithClass(Node{Xoffset: 10}, PFUNC),
			false,
		},
		{
			nodeWithClass(Node{Xoffset: 10}, PFUNC),
			nodeWithClass(Node{Xoffset: 10}, PFUNC),
			false,
		},
		{
			nodeWithClass(Node{Xoffset: 10}, PPARAM),
			nodeWithClass(Node{Xoffset: 20}, PPARAMOUT),
			true,
		},
		{
			nodeWithClass(Node{Xoffset: 10}, PPARAMOUT),
			nodeWithClass(Node{Xoffset: 20}, PPARAM),
			true,
		},
		{
			nodeWithClass(Node{flags: nodeUsed}, PAUTO),
			nodeWithClass(Node{}, PAUTO),
			true,
		},
		{
			nodeWithClass(Node{}, PAUTO),
			nodeWithClass(Node{flags: nodeUsed}, PAUTO),
			false,
		},
		{
			nodeWithClass(Node{Type: typeWithoutPointers()}, PAUTO),
			nodeWithClass(Node{Type: typeWithPointers()}, PAUTO),
			false,
		},
		{
			nodeWithClass(Node{Type: typeWithPointers()}, PAUTO),
			nodeWithClass(Node{Type: typeWithoutPointers()}, PAUTO),
			true,
		},
		{
			nodeWithClass(Node{Type: &types.Type{}, Name: &Name{flags: nameNeedzero}}, PAUTO),
			nodeWithClass(Node{Type: &types.Type{}, Name: &Name{}}, PAUTO),
			true,
		},
		{
			nodeWithClass(Node{Type: &types.Type{}, Name: &Name{}}, PAUTO),
			nodeWithClass(Node{Type: &types.Type{}, Name: &Name{flags: nameNeedzero}}, PAUTO),
			false,
		},
		{
			nodeWithClass(Node{Type: &types.Type{Width: 1}, Name: &Name{}}, PAUTO),
			nodeWithClass(Node{Type: &types.Type{Width: 2}, Name: &Name{}}, PAUTO),
			false,
		},
		{
			nodeWithClass(Node{Type: &types.Type{Width: 2}, Name: &Name{}}, PAUTO),
			nodeWithClass(Node{Type: &types.Type{Width: 1}, Name: &Name{}}, PAUTO),
			true,
		},
		{
			nodeWithClass(Node{Type: &types.Type{}, Name: &Name{}, Sym: &types.Sym{Name: "abc"}}, PAUTO),
			nodeWithClass(Node{Type: &types.Type{}, Name: &Name{}, Sym: &types.Sym{Name: "xyz"}}, PAUTO),
			true,
		},
		{
			nodeWithClass(Node{Type: &types.Type{}, Name: &Name{}, Sym: &types.Sym{Name: "abc"}}, PAUTO),
			nodeWithClass(Node{Type: &types.Type{}, Name: &Name{}, Sym: &types.Sym{Name: "abc"}}, PAUTO),
			false,
		},
		{
			nodeWithClass(Node{Type: &types.Type{}, Name: &Name{}, Sym: &types.Sym{Name: "xyz"}}, PAUTO),
			nodeWithClass(Node{Type: &types.Type{}, Name: &Name{}, Sym: &types.Sym{Name: "abc"}}, PAUTO),
			false,
		},
	}
	for _, d := range testdata {
		got := cmpstackvarlt(d.a, d.b)
		if got != d.lt {
			t.Errorf("want %#v < %#v", d.a, d.b)
		}
		// If we expect a < b to be true, check that b < a is false.
		if d.lt && cmpstackvarlt(d.b, d.a) {
			t.Errorf("unexpected %#v < %#v", d.b, d.a)
		}
	}
}

func TestStackvarSort(t *testing.T) {
	inp := []*Node{
		nodeWithClass(Node{Type: &types.Type{}, Name: &Name{}, Sym: &types.Sym{}}, PFUNC),
		nodeWithClass(Node{Type: &types.Type{}, Name: &Name{}, Sym: &types.Sym{}}, PAUTO),
		nodeWithClass(Node{Xoffset: 0, Type: &types.Type{}, Name: &Name{}, Sym: &types.Sym{}}, PFUNC),
		nodeWithClass(Node{Xoffset: 10, Type: &types.Type{}, Name: &Name{}, Sym: &types.Sym{}}, PFUNC),
		nodeWithClass(Node{Xoffset: 20, Type: &types.Type{}, Name: &Name{}, Sym: &types.Sym{}}, PFUNC),
		nodeWithClass(Node{flags: nodeUsed, Type: &types.Type{}, Name: &Name{}, Sym: &types.Sym{}}, PAUTO),
		nodeWithClass(Node{Type: typeWithoutPointers(), Name: &Name{}, Sym: &types.Sym{}}, PAUTO),
		nodeWithClass(Node{Type: &types.Type{}, Name: &Name{}, Sym: &types.Sym{}}, PAUTO),
		nodeWithClass(Node{Type: &types.Type{}, Name: &Name{flags: nameNeedzero}, Sym: &types.Sym{}}, PAUTO),
		nodeWithClass(Node{Type: &types.Type{Width: 1}, Name: &Name{}, Sym: &types.Sym{}}, PAUTO),
		nodeWithClass(Node{Type: &types.Type{Width: 2}, Name: &Name{}, Sym: &types.Sym{}}, PAUTO),
		nodeWithClass(Node{Type: &types.Type{}, Name: &Name{}, Sym: &types.Sym{Name: "abc"}}, PAUTO),
		nodeWithClass(Node{Type: &types.Type{}, Name: &Name{}, Sym: &types.Sym{Name: "xyz"}}, PAUTO),
	}
	want := []*Node{
		nodeWithClass(Node{Type: &types.Type{}, Name: &Name{}, Sym: &types.Sym{}}, PFUNC),
		nodeWithClass(Node{Xoffset: 0, Type: &types.Type{}, Name: &Name{}, Sym: &types.Sym{}}, PFUNC),
		nodeWithClass(Node{Xoffset: 10, Type: &types.Type{}, Name: &Name{}, Sym: &types.Sym{}}, PFUNC),
		nodeWithClass(Node{Xoffset: 20, Type: &types.Type{}, Name: &Name{}, Sym: &types.Sym{}}, PFUNC),
		nodeWithClass(Node{flags: nodeUsed, Type: &types.Type{}, Name: &Name{}, Sym: &types.Sym{}}, PAUTO),
		nodeWithClass(Node{Type: &types.Type{}, Name: &Name{flags: nameNeedzero}, Sym: &types.Sym{}}, PAUTO),
		nodeWithClass(Node{Type: &types.Type{Width: 2}, Name: &Name{}, Sym: &types.Sym{}}, PAUTO),
		nodeWithClass(Node{Type: &types.Type{Width: 1}, Name: &Name{}, Sym: &types.Sym{}}, PAUTO),
		nodeWithClass(Node{Type: &types.Type{}, Name: &Name{}, Sym: &types.Sym{}}, PAUTO),
		nodeWithClass(Node{Type: &types.Type{}, Name: &Name{}, Sym: &types.Sym{}}, PAUTO),
		nodeWithClass(Node{Type: &types.Type{}, Name: &Name{}, Sym: &types.Sym{Name: "abc"}}, PAUTO),
		nodeWithClass(Node{Type: &types.Type{}, Name: &Name{}, Sym: &types.Sym{Name: "xyz"}}, PAUTO),
		nodeWithClass(Node{Type: typeWithoutPointers(), Name: &Name{}, Sym: &types.Sym{}}, PAUTO),
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
