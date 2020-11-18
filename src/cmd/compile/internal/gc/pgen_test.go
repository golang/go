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
	f := &types.Field{Type: types.NewPtr(types.New(TINT))}
	t.SetFields([]*types.Field{f})
	return t
}

func markUsed(n *Node) *Node {
	n.Name.SetUsed(true)
	return n
}

func markNeedZero(n *Node) *Node {
	n.Name.SetNeedzero(true)
	return n
}

// Test all code paths for cmpstackvarlt.
func TestCmpstackvar(t *testing.T) {
	nod := func(xoffset int64, t *types.Type, s *types.Sym, cl Class) *Node {
		if s == nil {
			s = &types.Sym{Name: "."}
		}
		n := newname(s)
		n.Type = t
		n.Xoffset = xoffset
		n.SetClass(cl)
		return n
	}
	testdata := []struct {
		a, b *Node
		lt   bool
	}{
		{
			nod(0, nil, nil, PAUTO),
			nod(0, nil, nil, PFUNC),
			false,
		},
		{
			nod(0, nil, nil, PFUNC),
			nod(0, nil, nil, PAUTO),
			true,
		},
		{
			nod(0, nil, nil, PFUNC),
			nod(10, nil, nil, PFUNC),
			true,
		},
		{
			nod(20, nil, nil, PFUNC),
			nod(10, nil, nil, PFUNC),
			false,
		},
		{
			nod(10, nil, nil, PFUNC),
			nod(10, nil, nil, PFUNC),
			false,
		},
		{
			nod(10, nil, nil, PPARAM),
			nod(20, nil, nil, PPARAMOUT),
			true,
		},
		{
			nod(10, nil, nil, PPARAMOUT),
			nod(20, nil, nil, PPARAM),
			true,
		},
		{
			markUsed(nod(0, nil, nil, PAUTO)),
			nod(0, nil, nil, PAUTO),
			true,
		},
		{
			nod(0, nil, nil, PAUTO),
			markUsed(nod(0, nil, nil, PAUTO)),
			false,
		},
		{
			nod(0, typeWithoutPointers(), nil, PAUTO),
			nod(0, typeWithPointers(), nil, PAUTO),
			false,
		},
		{
			nod(0, typeWithPointers(), nil, PAUTO),
			nod(0, typeWithoutPointers(), nil, PAUTO),
			true,
		},
		{
			markNeedZero(nod(0, &types.Type{}, nil, PAUTO)),
			nod(0, &types.Type{}, nil, PAUTO),
			true,
		},
		{
			nod(0, &types.Type{}, nil, PAUTO),
			markNeedZero(nod(0, &types.Type{}, nil, PAUTO)),
			false,
		},
		{
			nod(0, &types.Type{Width: 1}, nil, PAUTO),
			nod(0, &types.Type{Width: 2}, nil, PAUTO),
			false,
		},
		{
			nod(0, &types.Type{Width: 2}, nil, PAUTO),
			nod(0, &types.Type{Width: 1}, nil, PAUTO),
			true,
		},
		{
			nod(0, &types.Type{}, &types.Sym{Name: "abc"}, PAUTO),
			nod(0, &types.Type{}, &types.Sym{Name: "xyz"}, PAUTO),
			true,
		},
		{
			nod(0, &types.Type{}, &types.Sym{Name: "abc"}, PAUTO),
			nod(0, &types.Type{}, &types.Sym{Name: "abc"}, PAUTO),
			false,
		},
		{
			nod(0, &types.Type{}, &types.Sym{Name: "xyz"}, PAUTO),
			nod(0, &types.Type{}, &types.Sym{Name: "abc"}, PAUTO),
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
	nod := func(xoffset int64, t *types.Type, s *types.Sym, cl Class) *Node {
		n := newname(s)
		n.Type = t
		n.Xoffset = xoffset
		n.SetClass(cl)
		return n
	}
	inp := []*Node{
		nod(0, &types.Type{}, &types.Sym{}, PFUNC),
		nod(0, &types.Type{}, &types.Sym{}, PAUTO),
		nod(0, &types.Type{}, &types.Sym{}, PFUNC),
		nod(10, &types.Type{}, &types.Sym{}, PFUNC),
		nod(20, &types.Type{}, &types.Sym{}, PFUNC),
		markUsed(nod(0, &types.Type{}, &types.Sym{}, PAUTO)),
		nod(0, typeWithoutPointers(), &types.Sym{}, PAUTO),
		nod(0, &types.Type{}, &types.Sym{}, PAUTO),
		markNeedZero(nod(0, &types.Type{}, &types.Sym{}, PAUTO)),
		nod(0, &types.Type{Width: 1}, &types.Sym{}, PAUTO),
		nod(0, &types.Type{Width: 2}, &types.Sym{}, PAUTO),
		nod(0, &types.Type{}, &types.Sym{Name: "abc"}, PAUTO),
		nod(0, &types.Type{}, &types.Sym{Name: "xyz"}, PAUTO),
	}
	want := []*Node{
		nod(0, &types.Type{}, &types.Sym{}, PFUNC),
		nod(0, &types.Type{}, &types.Sym{}, PFUNC),
		nod(10, &types.Type{}, &types.Sym{}, PFUNC),
		nod(20, &types.Type{}, &types.Sym{}, PFUNC),
		markUsed(nod(0, &types.Type{}, &types.Sym{}, PAUTO)),
		markNeedZero(nod(0, &types.Type{}, &types.Sym{}, PAUTO)),
		nod(0, &types.Type{Width: 2}, &types.Sym{}, PAUTO),
		nod(0, &types.Type{Width: 1}, &types.Sym{}, PAUTO),
		nod(0, &types.Type{}, &types.Sym{}, PAUTO),
		nod(0, &types.Type{}, &types.Sym{}, PAUTO),
		nod(0, &types.Type{}, &types.Sym{Name: "abc"}, PAUTO),
		nod(0, &types.Type{}, &types.Sym{Name: "xyz"}, PAUTO),
		nod(0, typeWithoutPointers(), &types.Sym{}, PAUTO),
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
