// Copyright 2026 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

package escape

import (
	"cmd/compile/internal/base"
	"cmd/compile/internal/ir"
	"cmd/compile/internal/types"
	"slices"
)

var logosArenaObjects = make(map[ir.Node]*ir.Func)

func logosArenaObjectType(n ir.Node) *types.Type {
	if n == nil || n.Type() == nil {
		return nil
	}
	switch n.Op() {
	case ir.ONEW, ir.OPTRLIT:
		return n.Type().Elem()
	case ir.OCONVIFACE:
		t := n.(*ir.ConvExpr).X.Type()
		if !t.IsInterface() && !types.IsDirectIface(t) && t.Size() > 1024 {
			return t
		}
	}
	return nil
}

func logosArenaObjectCandidate(n ir.Node, fn *ir.Func) bool {
	t := logosArenaObjectType(n)
	if t == nil || !LogosArenaPlaceable(fn) || base.Ctxt.Arch.Name != "amd64" || t.Size() <= 0 || t.Size() > LogosArenaMaxBytes || t.Alignment() > 8 {
		return false
	}
	return n.Op() == ir.OCONVIFACE || HeapAllocReason(n) == "too large for stack"
}

func (b *batch) logosArenaObjectShadows() map[*location]*location {
	if !LogosArenaEnabled() {
		return nil
	}
	var shadows map[*location]*location
	for _, loc := range b.allLocs {
		if logosArenaObjectCandidate(loc.n, loc.curfn) {
			if shadows == nil {
				shadows = make(map[*location]*location)
			}
			shadows[loc] = &location{n: loc.n, curfn: loc.curfn, loopDepth: loc.loopDepth, attrs: loc.attrs}
		}
	}
	if len(shadows) == 0 {
		return nil
	}
	for _, dst := range b.allLocs {
		var edges []edge
		for _, edge := range dst.edges {
			if shadow := shadows[edge.src]; shadow != nil {
				edge.src = shadow
				edges = append(edges, edge)
			}
		}
		dst.edges = append(dst.edges, edges...)
	}
	return shadows
}

func (b *batch) logosArenaMarkObjects(shadows map[*location]*location) {
	if len(shadows) == 0 {
		return
	}
	logosArenaReg.mu.Lock()
	defer logosArenaReg.mu.Unlock()
	for loc, shadow := range shadows {
		if !shadow.hasAttr(attrEscapes) && (loc.n.Op() != ir.OCONVIFACE || loc.n.Esc() == ir.EscNone) {
			logosArenaObjects[loc.n] = loc.curfn
		}
		for _, stmt := range loc.curfn.Body {
			logosArenaObjectLoops(stmt, shadow, nil)
		}
	}
}

func logosArenaObjectLoops(n ir.Node, loc *location, loops []ir.Node) {
	if n == nil || n.Op() == ir.OCLOSURE {
		return
	}
	if n.Op() == ir.OFOR || n.Op() == ir.ORANGE {
		loops = append(loops, n)
	}
	match := n == loc.n
	if n.Op() == ir.ODCL {
		match = n.(*ir.Decl).X == loc.n
	}
	if match && len(loops) != 0 {
		logosArenaObjectLoopVerdict(loc, loops)
		return
	}
	ir.DoChildren(n, func(child ir.Node) bool {
		if logosArenaObjectLoopHeader(n, child) {
			if ir.Any(child, func(x ir.Node) bool { return x == loc.n }) {
				logosArenaVetoLoopsLocked(loc.curfn, loops)
			}
			return false
		}
		logosArenaObjectLoops(child, loc, loops)
		return false
	})
}

func logosArenaObjectLoopVerdict(loc *location, loops []ir.Node) {
	if loc.hasAttr(attrEscapes) {
		logosArenaVetoLoopsLocked(loc.curfn, loops)
		return
	}
	restores := logosArenaReg.loopRestores[loc.curfn]
	if restores == nil {
		restores = make(map[ir.Node]bool)
		logosArenaReg.loopRestores[loc.curfn] = restores
	}
	loop := loops[len(loops)-1]
	if _, seen := restores[loop]; !seen {
		restores[loop] = true
	}
}

func logosArenaObjectLoopHeader(n, child ir.Node) bool {
	switch loop := n.(type) {
	case *ir.ForStmt:
		return !slices.Contains(loop.Body, child)
	case *ir.RangeStmt:
		return !slices.Contains(loop.Body, child)
	}
	return false
}

func LogosArenaObject(n ir.Node, fn *ir.Func) bool {
	if !LogosArenaPlaceable(fn) {
		return false
	}
	logosArenaReg.mu.RLock()
	defer logosArenaReg.mu.RUnlock()
	return logosArenaObjects[n] == fn
}

func LogosArenaTransferObject(from, to ir.Node, fn *ir.Func) {
	if !LogosArenaObject(from, fn) {
		return
	}
	logosArenaReg.mu.Lock()
	logosArenaObjects[to] = fn
	logosArenaReg.mu.Unlock()
}
