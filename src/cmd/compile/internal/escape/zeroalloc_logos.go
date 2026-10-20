// Copyright 2026 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

package escape

import (
	"fmt"

	"cmd/compile/internal/base"
	"cmd/compile/internal/ir"
	"cmd/compile/internal/logopt"
	"cmd/internal/src"
)

// LogosFuncKey names a function by the source position of its declaration:
// the file and the line of the token after func.
type LogosFuncKey struct {
	File string
	Line uint
}

// LogosZeroAlloc maps each function marked //logos:zeroalloc (SPEC Sec 22.4)
// to its declared name. The loGo/s lint hook sets it to the marked functions
// of the package; it is empty for every other compile, so the flow graph and
// the -m output stay those of plain Go.
//
// Escape analysis then records every heap allocation site it decides inside
// a marked function, including its function literals and the bodies inlined
// into it: an allocation expression that escapes, a variable moved to the
// heap, and make(chan T), which no flow edge models because a channel always
// lives on the heap. Escape analysis decides these sites only. An operation
// that falls back to the heap at run time, such as append past the capacity,
// a string concatenation or conversion longer than the stack buffer, or an
// insert that grows a map, is not an allocation site.
var LogosZeroAlloc map[LogosFuncKey]string

// LogosAlloc is one heap allocation site inside a marked function: the
// position of the allocating expression or variable, the declared name of
// the marked function that performs it, and why escape analysis puts it on
// the heap.
type LogosAlloc struct {
	Pos    src.XPos
	Func   string
	Reason string
}

var logosAllocs []LogosAlloc

// LogosAllocs returns the allocation sites found so far, in analysis order.
// A source expression the compiler analyzes more than once, such as an
// inlined body or a generic instantiation, can appear more than once.
func LogosAllocs() []LogosAlloc {
	return logosAllocs
}

// logosNotes reports whether the package marks a function, so escape
// analysis records allocation sites.
func logosNotes() bool {
	return len(LogosZeroAlloc) != 0
}

// logosNote adds the note of k to the hole when the reason of an allocation
// needs it: the note that carries the value to where it escapes. Upstream
// keeps notes only for -m=2 and the optimization log, and then k already
// holds this note.
func logosNote(k hole, where ir.Node, why string) hole {
	if !logosNotes() || base.Flag.LowerM >= 2 || logopt.Enabled() {
		return k
	}
	k.notes = &note{next: k.notes, where: where, why: why}
	return k
}

// logosEscapeWalk records the allocation l, which escapes because the solver
// found that its address reaches root. It runs before l gains the escape
// attribute, so a location reports once.
func (b *batch) logosEscapeWalk(l, root *location) {
	if !logosNotes() {
		return
	}
	b.logosEscaped(l, root, b.logosLastNotes(l, root))
}

// logosEscaped records l when it is an allocation of a marked function that
// escapes to dst through the flow edge that carries notes. It runs before l
// gains the escape attribute, so a location reports once.
func (b *batch) logosEscaped(l, dst *location, notes *note) {
	if !logosNotes() || l.hasAttr(attrEscapes) || l.n == nil || !logosAllocates(l.n) {
		return
	}
	name, ok := logosMarked(l.curfn)
	if !ok {
		return
	}
	logosAllocs = append(logosAllocs, LogosAlloc{
		Pos:    l.n.Pos(),
		Func:   name,
		Reason: b.logosReason(l.n, dst, notes),
	})
}

// logosMakeChan records make(chan T) in a marked function. A channel always
// lives on the heap, so escape analysis keeps no location for it.
func (e *escape) logosMakeChan(n *ir.MakeExpr) {
	if !logosNotes() {
		return
	}
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

// logosMarked returns the declared name of the marked function that
// lexically holds fn, which is fn itself unless fn is a function literal.
func logosMarked(fn *ir.Func) (string, bool) {
	for fn.ClosureParent != nil {
		fn = fn.ClosureParent
	}
	pos := base.Ctxt.InnermostPos(fn.Pos())
	name, ok := LogosZeroAlloc[LogosFuncKey{File: pos.RelFilename(), Line: pos.Line()}]
	return name, ok
}

// logosLastNotes returns the notes of the last flow edge on the path from l
// to root, the edge that tells which statement carried the value to root, or
// nil when the path is unknown.
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

// logosReason describes why n escapes: what it is, where its address flows,
// and the statement that carries it there, in the words of -m=2.
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

// logosDest names the location an allocation escapes to.
func (b *batch) logosDest(dst *location) string {
	switch {
	case dst == &b.heapLoc:
		return "the heap"
	case dst.paramOut:
		return "result " + b.explainLoc(dst)
	}
	return b.explainLoc(dst)
}

// logosTail returns the last note of the list: the context that started the
// flow, such as return or call parameter. The list holds the innermost note
// first.
func logosTail(n *note) *note {
	for n != nil && n.next != nil {
		n = n.next
	}
	return n
}

// logosAllocates reports whether n, a location the solver moved to the heap,
// costs a heap allocation at run time. A conversion of a constant, of a value
// that needs no storage, and a function literal that captures nothing are
// static data, and a zero-size object has no address of its own.
func logosAllocates(n ir.Node) bool {
	switch n.Op() {
	case ir.ONAME:
		return !ir.IsBlank(n)
	case ir.OPTRLIT, ir.ONEW:
		return n.Type().Elem().Size() != 0
	case ir.OSLICELIT:
		return len(n.(*ir.CompLitExpr).List) != 0
	case ir.OCLOSURE:
		return len(n.(*ir.ClosureExpr).Func.ClosureVars) != 0
	case ir.OCONVIFACE:
		return !logosStaticIface(n.(*ir.ConvExpr))
	case ir.OMAKESLICE, ir.OMAKEMAP, ir.OMAPLIT, ir.OMETHVALUE, ir.OADDSTR,
		ir.OSTR2BYTES, ir.OBYTES2STR, ir.ORUNES2STR, ir.OSTR2RUNES, ir.ORUNESTR:
		return true
	}
	return false
}

// logosStaticIface reports whether the interface conversion n needs no
// allocation whatever its result does: a constant lives in read-only data, a
// zero-size value shares one address, and a bool or byte indexes a static
// table.
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
