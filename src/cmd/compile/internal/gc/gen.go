// Copyright 2009 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

package gc

import (
	"cmd/compile/internal/types"
	"cmd/internal/obj"
	"cmd/internal/src"
	"strconv"
)

func Sysfunc(name string) *obj.LSym {
	return Runtimepkg.Lookup(name).Linksym()
}

// isParamStackCopy reports whether this is the on-stack copy of a
// function parameter that moved to the heap.
func (n *Node) isParamStackCopy() bool {
	return n.Op == ONAME && (n.Class() == PPARAM || n.Class() == PPARAMOUT) && n.Name.Param.Heapaddr != nil
}

// isParamHeapCopy reports whether this is the on-heap copy of
// a function parameter that moved to the heap.
func (n *Node) isParamHeapCopy() bool {
	return n.Op == ONAME && n.Class() == PAUTOHEAP && n.Name.Param.Stackcopy != nil
}

// autotmpname returns the name for an autotmp variable numbered n.
func autotmpname(n int) string {
	// Give each tmp a different name so that they can be registerized.
	// Add a preceding . to avoid clashing with legal names.
	const prefix = ".autotmp_"
	// Start with a buffer big enough to hold a large n.
	b := []byte(prefix + "      ")[:len(prefix)]
	b = strconv.AppendInt(b, int64(n), 10)
	return types.InternString(b)
}

// make a new Node off the books
func tempnamel(pos src.XPos, curfn *Node, nn *Node, t *types.Type) {
	if curfn == nil {
		Fatalf("no curfn for tempname")
	}
	if curfn.Func.Closure != nil && curfn.Op == OCLOSURE {
		Dump("tempname", curfn)
		Fatalf("adding tempname to wrong closure function")
	}
	if t == nil {
		Fatalf("tempname called with nil type")
	}

	s := &types.Sym{
		Name: autotmpname(len(curfn.Func.Dcl)),
		Pkg:  localpkg,
	}
	n := newnamel(pos, s)
	s.Def = asTypesNode(n)
	n.Type = t
	n.SetClass(PAUTO)
	n.Esc = EscNever
	n.Name.Curfn = curfn
	n.Name.SetAutoTemp(true)
	curfn.Func.Dcl = append(curfn.Func.Dcl, n)

	dowidth(t)
	*nn = *n
}

func temp(t *types.Type) *Node {
	var n Node
	tempnamel(lineno, Curfn, &n, t)
	asNode(n.Sym.Def).Name.SetUsed(true)
	return n.Orig
}

func tempAt(pos src.XPos, curfn *Node, t *types.Type) *Node {
	var n Node
	tempnamel(pos, curfn, &n, t)
	asNode(n.Sym.Def).Name.SetUsed(true)
	return n.Orig
}
