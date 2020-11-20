// Copyright 2015 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

package gc

import (
	"cmd/compile/internal/ir"
	"cmd/compile/internal/types"
	"reflect"
	"sort"
	"testing"
)

func typeWithoutPointers() *types.Type {
	t := types.New(types.TSTRUCT)
	f := &types.Field{Type: types.New(types.TINT)}
	t.SetFields([]*types.Field{f})
	return t
}

func typeWithPointers() *types.Type {
	t := types.New(types.TSTRUCT)
	f := &types.Field{Type: types.NewPtr(types.New(types.TINT))}
	t.SetFields([]*types.Field{f})
	return t
}

func markUsed(n *ir.Node) *ir.Node {
	n.Name.SetUsed(true)
	return n
}

func markNeedZero(n *ir.Node) *ir.Node {
	n.Name.SetNeedzero(true)
	return n
}

// Test all code paths for cmpstackvarlt.
func TestCmpstackvar(t *testing.T) {
	nod := func(xoffset int64, t *types.Type, s *types.Sym, cl ir.Class) *ir.Node {
		if s == nil {
			s = &types.Sym{Name: "."}
		}
		n := NewName(s)
		n.Type = t
		n.Xoffset = xoffset
		n.SetClass(cl)
		return n
	}
	testdata := []struct {
		a, b *ir.Node
		lt   bool
	}{
		{
			nod(0, nil, nil, ir.PAUTO),
			nod(0, nil, nil, ir.PFUNC),
			false,
		},
		{
			nod(0, nil, nil, ir.PFUNC),
			nod(0, nil, nil, ir.PAUTO),
			true,
		},
		{
			nod(0, nil, nil, ir.PFUNC),
			nod(10, nil, nil, ir.PFUNC),
			true,
		},
		{
			nod(20, nil, nil, ir.PFUNC),
			nod(10, nil, nil, ir.PFUNC),
			false,
		},
		{
			nod(10, nil, nil, ir.PFUNC),
			nod(10, nil, nil, ir.PFUNC),
			false,
		},
		{
			nod(10, nil, nil, ir.PPARAM),
			nod(20, nil, nil, ir.PPARAMOUT),
			true,
		},
		{
			nod(10, nil, nil, ir.PPARAMOUT),
			nod(20, nil, nil, ir.PPARAM),
			true,
		},
		{
			markUsed(nod(0, nil, nil, ir.PAUTO)),
			nod(0, nil, nil, ir.PAUTO),
			true,
		},
		{
			nod(0, nil, nil, ir.PAUTO),
			markUsed(nod(0, nil, nil, ir.PAUTO)),
			false,
		},
		{
			nod(0, typeWithoutPointers(), nil, ir.PAUTO),
			nod(0, typeWithPointers(), nil, ir.PAUTO),
			false,
		},
		{
			nod(0, typeWithPointers(), nil, ir.PAUTO),
			nod(0, typeWithoutPointers(), nil, ir.PAUTO),
			true,
		},
		{
			markNeedZero(nod(0, &types.Type{}, nil, ir.PAUTO)),
			nod(0, &types.Type{}, nil, ir.PAUTO),
			true,
		},
		{
			nod(0, &types.Type{}, nil, ir.PAUTO),
			markNeedZero(nod(0, &types.Type{}, nil, ir.PAUTO)),
			false,
		},
		{
			nod(0, &types.Type{Width: 1}, nil, ir.PAUTO),
			nod(0, &types.Type{Width: 2}, nil, ir.PAUTO),
			false,
		},
		{
			nod(0, &types.Type{Width: 2}, nil, ir.PAUTO),
			nod(0, &types.Type{Width: 1}, nil, ir.PAUTO),
			true,
		},
		{
			nod(0, &types.Type{}, &types.Sym{Name: "abc"}, ir.PAUTO),
			nod(0, &types.Type{}, &types.Sym{Name: "xyz"}, ir.PAUTO),
			true,
		},
		{
			nod(0, &types.Type{}, &types.Sym{Name: "abc"}, ir.PAUTO),
			nod(0, &types.Type{}, &types.Sym{Name: "abc"}, ir.PAUTO),
			false,
		},
		{
			nod(0, &types.Type{}, &types.Sym{Name: "xyz"}, ir.PAUTO),
			nod(0, &types.Type{}, &types.Sym{Name: "abc"}, ir.PAUTO),
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
	nod := func(xoffset int64, t *types.Type, s *types.Sym, cl ir.Class) *ir.Node {
		n := NewName(s)
		n.Type = t
		n.Xoffset = xoffset
		n.SetClass(cl)
		return n
	}
	inp := []*ir.Node{
		nod(0, &types.Type{}, &types.Sym{}, ir.PFUNC),
		nod(0, &types.Type{}, &types.Sym{}, ir.PAUTO),
		nod(0, &types.Type{}, &types.Sym{}, ir.PFUNC),
		nod(10, &types.Type{}, &types.Sym{}, ir.PFUNC),
		nod(20, &types.Type{}, &types.Sym{}, ir.PFUNC),
		markUsed(nod(0, &types.Type{}, &types.Sym{}, ir.PAUTO)),
		nod(0, typeWithoutPointers(), &types.Sym{}, ir.PAUTO),
		nod(0, &types.Type{}, &types.Sym{}, ir.PAUTO),
		markNeedZero(nod(0, &types.Type{}, &types.Sym{}, ir.PAUTO)),
		nod(0, &types.Type{Width: 1}, &types.Sym{}, ir.PAUTO),
		nod(0, &types.Type{Width: 2}, &types.Sym{}, ir.PAUTO),
		nod(0, &types.Type{}, &types.Sym{Name: "abc"}, ir.PAUTO),
		nod(0, &types.Type{}, &types.Sym{Name: "xyz"}, ir.PAUTO),
	}
	want := []*ir.Node{
		nod(0, &types.Type{}, &types.Sym{}, ir.PFUNC),
		nod(0, &types.Type{}, &types.Sym{}, ir.PFUNC),
		nod(10, &types.Type{}, &types.Sym{}, ir.PFUNC),
		nod(20, &types.Type{}, &types.Sym{}, ir.PFUNC),
		markUsed(nod(0, &types.Type{}, &types.Sym{}, ir.PAUTO)),
		markNeedZero(nod(0, &types.Type{}, &types.Sym{}, ir.PAUTO)),
		nod(0, &types.Type{Width: 2}, &types.Sym{}, ir.PAUTO),
		nod(0, &types.Type{Width: 1}, &types.Sym{}, ir.PAUTO),
		nod(0, &types.Type{}, &types.Sym{}, ir.PAUTO),
		nod(0, &types.Type{}, &types.Sym{}, ir.PAUTO),
		nod(0, &types.Type{}, &types.Sym{Name: "abc"}, ir.PAUTO),
		nod(0, &types.Type{}, &types.Sym{Name: "xyz"}, ir.PAUTO),
		nod(0, typeWithoutPointers(), &types.Sym{}, ir.PAUTO),
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
