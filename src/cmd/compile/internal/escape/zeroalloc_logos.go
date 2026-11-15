// Copyright 2026 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

package escape

import (
	"fmt"
	"go/constant"

	"cmd/compile/internal/base"
	"cmd/compile/internal/ir"
	"cmd/compile/internal/logopt"
	"cmd/internal/src"
)

type LogosFuncKey struct {
	File string
	Line uint
}

var LogosZeroAlloc map[LogosFuncKey]string

type LogosAlloc struct {
	Pos    src.XPos
	Func   string
	Reason string
}

var logosAllocs []LogosAlloc

func LogosAllocs() []LogosAlloc {
	return logosAllocs
}

func logosNotes() bool {
	return len(LogosZeroAlloc) != 0 || LogosFuncAllocs != nil
}

func logosNote(k hole, where ir.Node, why string) hole {
	if !logosNotes() || base.Flag.LowerM >= 2 || logopt.Enabled() {
		return k
	}
	k.notes = &note{next: k.notes, where: where, why: why}
	return k
}

func (b *batch) logosEscapeWalk(l, root *location) {
	if !logosNotes() {
		return
	}
	b.logosEscaped(l, root, b.logosLastNotes(l, root))
}

func (b *batch) logosEscaped(l, dst *location, notes *note) {
	if !logosNotes() || l.hasAttr(attrEscapes) || l.n == nil || !logosAllocates(l.n) {
		return
	}
	name, ok := logosMarked(l.curfn)
	if !ok && LogosFuncAllocs == nil {
		return
	}
	logosRecordFuncAlloc(l.curfn)
	if !ok {
		return
	}
	logosAllocs = append(logosAllocs, LogosAlloc{
		Pos:    l.n.Pos(),
		Func:   name,
		Reason: b.logosReason(l.n, dst, notes),
	})
}

func (e *escape) logosMakeChan(n *ir.MakeExpr) {
	if !logosNotes() {
		return
	}
	logosRecordFuncAlloc(e.curfn)
	name, ok := logosMarked(e.curfn)
	if !ok {
		return
	}
	logosAllocs = append(logosAllocs, LogosAlloc{
		Pos:    n.Pos(),
		Func:   name,
		Reason: fmt.Sprintf("%v escapes to heap: channels always live on the heap", n),
	})
}

func logosMarked(fn *ir.Func) (string, bool) {
	for fn.ClosureParent != nil {
		fn = fn.ClosureParent
	}
	pos := base.Ctxt.InnermostPos(fn.Pos())
	name, ok := LogosZeroAlloc[LogosFuncKey{File: pos.RelFilename(), Line: pos.Line()}]
	return name, ok
}

func (b *batch) logosLastNotes(l, root *location) *note {
	for range len(b.allLocs) + 1 {
		dst := l.dst
		if dst == nil {
			return nil
		}
		if dst == root {
			return dst.edges[l.dstEdgeIdx].notes
		}
		l = dst
	}
	return nil
}

func (b *batch) logosReason(n ir.Node, dst *location, notes *note) string {
	what := fmt.Sprintf("%v escapes to heap", n)
	if n.Op() == ir.ONAME {
		what = fmt.Sprintf("moved to heap: %v", n)
	}
	reason := what + ": flows to " + b.logosDest(dst)
	if last := logosTail(notes); last != nil {
		reason += " (" + last.why + ")"
	}
	return reason
}

func (b *batch) logosDest(dst *location) string {
	switch {
	case dst == &b.heapLoc:
		return "the heap"
	case dst.paramOut:
		return "result " + b.explainLoc(dst)
	}
	return b.explainLoc(dst)
}

func logosTail(n *note) *note {
	for n != nil && n.next != nil {
		n = n.next
	}
	return n
}

func logosAllocates(n ir.Node) bool {
	switch n.Op() {
	case ir.ONAME:
		return !ir.IsBlank(n) && n.Type().Size() != 0
	case ir.OPTRLIT, ir.ONEW:
		return n.Type().Elem().Size() != 0
	case ir.OSLICELIT:
		return len(n.(*ir.CompLitExpr).List) != 0
	case ir.OCLOSURE:
		return len(n.(*ir.ClosureExpr).Func.ClosureVars) != 0
	case ir.OCONVIFACE:
		return !logosStaticIface(n.(*ir.ConvExpr))
	case ir.OMAKESLICE:
		return !logosZeroSlice(n.(*ir.MakeExpr))
	case ir.OMAKEMAP, ir.OMAPLIT, ir.OMETHVALUE, ir.OADDSTR,
		ir.OSTR2BYTES, ir.OBYTES2STR, ir.ORUNES2STR, ir.OSTR2RUNES, ir.ORUNESTR:
		return true
	}
	return false
}

func logosZeroSlice(n *ir.MakeExpr) bool {
	bound, ok := logosSliceBound(n.Len)
	if !ok || bound != 0 {
		return false
	}
	if n.Cap == nil {
		return true
	}
	bound, ok = logosSliceBound(n.Cap)
	return ok && bound == 0
}

func logosSliceBound(n ir.Node) (int64, bool) {
	lit, ok := n.(*ir.BasicLit)
	if !ok || lit.Op() != ir.OLITERAL {
		return 0, false
	}
	bound, ok := constant.Int64Val(lit.Val())
	return bound, ok
}

func logosStaticIface(n *ir.ConvExpr) bool {
	t := n.X.Type()
	switch {
	case n.X.Op() == ir.OLITERAL || n.X.Op() == ir.ONIL:
		return true
	case t.Size() == 0:
		return true
	case t.IsBoolean(), t.Size() == 1 && t.IsInteger():
		return true
	}
	return false
}
