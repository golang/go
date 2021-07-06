// Copyright 2015 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

package ssagen

import (
	"reflect"
	"sort"
	"testing"

	"cmd/compile/internal/ir"
	"cmd/compile/internal/typecheck"
	"cmd/compile/internal/types"
	"cmd/internal/src"
)

func typeWithoutPointers() *types.Type {
	return types.NewStruct(types.NoPkg, []*types.Field{
		types.NewField(src.NoXPos, nil, types.New(types.TINT)),
	})
}

func typeWithPointers() *types.Type {
	return types.NewStruct(types.NoPkg, []*types.Field{
		types.NewField(src.NoXPos, nil, types.NewPtr(types.New(types.TINT))),
	})
}

func markUsed(n *ir.Name) *ir.Name {
	n.SetUsed(true)
	return n
}

func markNeedZero(n *ir.Name) *ir.Name {
	n.SetNeedzero(true)
	return n
}

// Test all code paths for cmpstackvarlt.
func TestCmpstackvar(t *testing.T) {
	nod := func(xoffset int64, t *types.Type, s *types.Sym, cl ir.Class) *ir.Name {
		if s == nil {
			s = &types.Sym{Name: "."}
		}
		n := typecheck.NewName(s)
		n.SetType(t)
		n.SetFrameOffset(xoffset)
		n.Class = cl
		return n
	}
	testdata := []struct {
		a, b *ir.Name
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
			t.Errorf("want %v < %v", d.a, d.b)
		}
		// If we expect a < b to be true, check that b < a is false.
		if d.lt && cmpstackvarlt(d.b, d.a) {
			t.Errorf("unexpected %v < %v", d.b, d.a)
		}
	}
}

func TestStackvarSort(t *testing.T) {
	nod := func(xoffset int64, t *types.Type, s *types.Sym, cl ir.Class) *ir.Name {
		n := typecheck.NewName(s)
		n.SetType(t)
		n.SetFrameOffset(xoffset)
		n.Class = cl
		return n
	}
	inp := []*ir.Name{
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
	want := []*ir.Name{
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
