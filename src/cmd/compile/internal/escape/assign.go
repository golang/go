// Copyright 2018 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

package escape

import (
	"cmd/compile/internal/base"
	"cmd/compile/internal/ir"
)

// addr evaluates an addressable expression n and returns a hole
// that represents storing into the represented location.
func (e *escape) addr(n ir.Node) hole {
	if n == nil || ir.IsBlank(n) {
		// Can happen in select case, range, maybe others.
		return e.discardHole()
	}

	k := e.heapHole()

	switch n.Op() {
	default:
		base.Fatalf("unexpected addr: %v", n)
	case ir.ONAME:
		n := n.(*ir.Name)
		if n.Class == ir.PEXTERN {
			break
		}
		k = e.oldLoc(n).asHole()
	case ir.OLINKSYMOFFSET:
		break
	case ir.ODOT:
		n := n.(*ir.SelectorExpr)
		k = e.addr(n.X)
	case ir.OINDEX:
		n := n.(*ir.IndexExpr)
		e.discard(n.Index)
		if n.X.Type().IsArray() {
			k = e.addr(n.X)
		} else {
			e.mutate(n.X)
		}
	case ir.ODEREF:
		n := n.(*ir.StarExpr)
		e.mutate(n.X)
	case ir.ODOTPTR:
		n := n.(*ir.SelectorExpr)
		e.mutate(n.X)
	case ir.OINDEXMAP:
		n := n.(*ir.IndexExpr)
		e.discard(n.X)
		e.assignHeap(n.Index, "key of map put", n)
	}

	return k
}

func (e *escape) mutate(n ir.Node) {
	e.expr(e.mutatorHole(), n)
}

func (e *escape) addrs(l ir.Nodes) []hole {
	var ks []hole
	for _, n := range l {
		ks = append(ks, e.addr(n))
	}
	return ks
}

func (e *escape) assignHeap(src ir.Node, why string, where ir.Node) {
	e.expr(e.heapHole().note(where, why), src)
}

// assignList evaluates the assignment dsts... = srcs....
func (e *escape) assignList(dsts, srcs []ir.Node, why string, where ir.Node) {
	ks := e.addrs(dsts)
	for i, k := range ks {
		var src ir.Node
		if i < len(srcs) {
			src = srcs[i]
		}

		if dst := dsts[i]; dst != nil {
			// Detect implicit conversion of uintptr to unsafe.Pointer when
			// storing into reflect.{Slice,String}Header.
			if dst.Op() == ir.ODOTPTR && ir.IsReflectHeaderDataField(dst) {
				e.unsafeValue(e.heapHole().note(where, why), src)
				continue
			}

			// Filter out some no-op assignments for escape analysis.
			if src != nil && isSelfAssign(dst, src) {
				if base.Flag.LowerM != 0 {
					base.WarnfAt(where.Pos(), "%v ignoring self-assignment in %v", e.curfn, where)
				}
				k = e.discardHole()
			}
		}

		e.expr(k.note(where, why), src)
	}

	e.reassigned(ks, where)
}

// reassigned marks the locations associated with the given holes as
// reassigned, unless the location represents a variable declared and
// assigned exactly once by where.
func (e *escape) reassigned(ks []hole, where ir.Node) {
	if as, ok := where.(*ir.AssignStmt); ok && as.Op() == ir.OAS && as.Y == nil {
		if dst, ok := as.X.(*ir.Name); ok && dst.Op() == ir.ONAME && dst.Defn == nil {
			// Zero-value assignment for variable declared without an
			// explicit initial value. Assume this is its initialization
			// statement.
			return
		}
	}

	for _, k := range ks {
		loc := k.dst
		// Variables declared by range statements are assigned on every iteration.
		if n, ok := loc.n.(*ir.Name); ok && n.Defn == where && where.Op() != ir.ORANGE {
			continue
		}
		loc.reassigned = true
	}
}
