// Copyright 2018 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

package escape

import (
	"cmd/compile/internal/base"
	"cmd/compile/internal/ir"
	"cmd/compile/internal/logopt"
	"cmd/compile/internal/types"
	"fmt"
)

// Below we implement the methods for walking the AST and recording
// data flow edges. Note that because a sub-expression might have
// side-effects, it's important to always visit the entire AST.
//
// For example, write either:
//
//     if x {
//         e.discard(n.Left)
//     } else {
//         e.value(k, n.Left)
//     }
//
// or
//
//     if x {
//         k = e.discardHole()
//     }
//     e.value(k, n.Left)
//
// Do NOT write:
//
//    // BAD: possibly loses side-effects within n.Left
//    if !x {
//        e.value(k, n.Left)
//    }

// A location represents an abstract location that stores a Go
// variable.
type location struct {
	n         ir.Node  // represented variable or expression, if any
	curfn     *ir.Func // enclosing function
	edges     []edge   // incoming edges
	loopDepth int      // loopDepth at declaration

	// resultIndex records the tuple index (starting at 1) for
	// PPARAMOUT variables within their function's result type.
	// For non-PPARAMOUT variables it's 0.
	resultIndex int

	// derefs and walkgen are used during walkOne to track the
	// minimal dereferences from the walk root.
	derefs  int // >= -1
	walkgen uint32

	// dst and dstEdgeindex track the next immediate assignment
	// destination location during walkone, along with the index
	// of the edge pointing back to this location.
	dst        *location
	dstEdgeIdx int

	// queuedWalkAll is used by walkAll to track whether this location is
	// in its work queue.
	queuedWalkAll bool

	// queuedWalkOne is used by walkOne to track whether this location is
	// in its work queue. The value is the walkgen when this location was
	// last queued for walkOne, or 0 if it's not currently queued.
	queuedWalkOne uint32

	// attrs is a bitset of location attributes.
	attrs locAttr

	// paramEsc records the represented parameter's leak set.
	paramEsc leaks

	captured   bool // has a closure captured this variable?
	reassigned bool // has this variable been reassigned?
	addrtaken  bool // has this variable's address been taken?
	param      bool // is this variable a parameter (ONAME of class ir.PPARAM)?
	paramOut   bool // is this variable an out parameter (ONAME of class ir.PPARAMOUT)?
}

type locAttr uint8

const (
	// attrEscapes indicates whether the represented variable's address
	// escapes; that is, whether the variable must be heap allocated.
	attrEscapes locAttr = 1 << iota

	// attrPersists indicates whether the represented expression's
	// address outlives the statement; that is, whether its storage
	// cannot be immediately reused.
	attrPersists

	// attrMutates indicates whether pointers that are reachable from
	// this location may have their addressed memory mutated. This is
	// used to detect string->[]byte conversions that can be safely
	// optimized away.
	attrMutates

	// attrCalls indicates whether closures that are reachable from this
	// location may be called without tracking their results. This is
	// used to better optimize indirect closure calls.
	attrCalls
)

func (l *location) hasAttr(attr locAttr) bool { return l.attrs&attr != 0 }

// An edge represents an assignment edge between two Go variables.
type edge struct {
	src    *location
	derefs int // >= -1
	notes  *note
}

func (l *location) asHole() hole {
	return hole{dst: l}
}

// leak records that parameter l leaks to sink.
func (l *location) leakTo(sink *location, derefs int) {
	// If sink is a result parameter that doesn't escape (#44614)
	// and we can fit return bits into the escape analysis tag,
	// then record as a result leak.
	if !sink.hasAttr(attrEscapes) && sink.isName(ir.PPARAMOUT) && sink.curfn == l.curfn {
		ri := sink.resultIndex - 1
		if ri < numEscResults {
			// Leak to result parameter.
			l.paramEsc.AddResult(ri, derefs)
			return
		}
	}

	// Otherwise, record as heap leak.
	l.paramEsc.AddHeap(derefs)
}

// leakTo records that parameter l leaks to sink.
func (b *batch) leakTo(l, sink *location, derefs int) {
	if (logopt.Enabled() || base.Flag.LowerM >= 2) && !l.hasAttr(attrEscapes) {
		if base.Flag.LowerM >= 2 {
			fmt.Printf("%s: parameter %v leaks to %s with derefs=%d:\n", base.FmtPos(l.n.Pos()), l.n, b.explainLoc(sink), derefs)
		}
		explanation := b.explainPath(sink, l)
		if logopt.Enabled() {
			var e_curfn *ir.Func // TODO(mdempsky): Fix.
			logopt.LogOpt(l.n.Pos(), "leak", "escape", ir.FuncName(e_curfn),
				fmt.Sprintf("parameter %v leaks to %s with derefs=%d", l.n, b.explainLoc(sink), derefs), explanation)
		}
	}

	// If sink is a result parameter that doesn't escape (#44614)
	// and we can fit return bits into the escape analysis tag,
	// then record as a result leak.
	if !sink.hasAttr(attrEscapes) && sink.isName(ir.PPARAMOUT) && sink.curfn == l.curfn {
		if ri := sink.resultIndex - 1; ri < numEscResults {
			// Leak to result parameter.
			l.paramEsc.AddResult(ri, derefs)
			return
		}
	}

	// Otherwise, record as heap leak.
	l.paramEsc.AddHeap(derefs)
}

func (l *location) isName(c ir.Class) bool {
	return l.n != nil && l.n.Op() == ir.ONAME && l.n.(*ir.Name).Class == c
}

// A hole represents a context for evaluation of a Go
// expression. E.g., when evaluating p in "x = **p", we'd have a hole
// with dst==x and derefs==2.
type hole struct {
	dst    *location
	derefs int // >= -1
	notes  *note

	// addrtaken indicates whether this context is taking the address of
	// the expression, independent of whether the address will actually
	// be stored into a variable.
	addrtaken bool
}

type note struct {
	next  *note
	where ir.Node
	why   string
}

func (k hole) note(where ir.Node, why string) hole {
	if where == nil || why == "" {
		base.Fatalf("note: missing where/why")
	}
	if base.Flag.LowerM >= 2 || logopt.Enabled() {
		k.notes = &note{
			next:  k.notes,
			where: where,
			why:   why,
		}
	}
	return k
}

func (k hole) shift(delta int) hole {
	k.derefs += delta
	if k.derefs < -1 {
		base.Fatalf("derefs underflow: %v", k.derefs)
	}
	k.addrtaken = delta < 0
	return k
}

func (k hole) deref(where ir.Node, why string) hole { return k.shift(1).note(where, why) }
func (k hole) addr(where ir.Node, why string) hole  { return k.shift(-1).note(where, why) }

func (k hole) dotType(t *types.Type, where ir.Node, why string) hole {
	if !t.IsInterface() && !types.IsDirectIface(t) {
		k = k.shift(1)
	}
	return k.note(where, why)
}

func (b *batch) flow(k hole, src *location) {
	if k.addrtaken {
		src.addrtaken = true
	}

	dst := k.dst
	if dst == &b.blankLoc {
		return
	}
	if dst == src && k.derefs >= 0 { // dst = dst, dst = *dst, ...
		return
	}
	if dst.hasAttr(attrEscapes) && k.derefs < 0 { // dst = &src
		if base.Flag.LowerM >= 2 || logopt.Enabled() {
			pos := base.FmtPos(src.n.Pos())
			if base.Flag.LowerM >= 2 {
				fmt.Printf("%s: %v escapes to heap:\n", pos, src.n)
			}
			explanation := b.explainFlow(pos, dst, src, k.derefs, k.notes, []*logopt.LoggedOpt{})
			if logopt.Enabled() {
				var e_curfn *ir.Func // TODO(mdempsky): Fix.
				logopt.LogOpt(src.n.Pos(), "escapes", "escape", ir.FuncName(e_curfn), fmt.Sprintf("%v escapes to heap", src.n), explanation)
			}

		}
		src.attrs |= attrEscapes | attrPersists | attrMutates | attrCalls
		return
	}

	// TODO(mdempsky): Deduplicate edges?
	dst.edges = append(dst.edges, edge{src: src, derefs: k.derefs, notes: k.notes})
}

func (b *batch) heapHole() hole    { return b.heapLoc.asHole() }
func (b *batch) mutatorHole() hole { return b.mutatorLoc.asHole() }
func (b *batch) calleeHole() hole  { return b.calleeLoc.asHole() }
func (b *batch) discardHole() hole { return b.blankLoc.asHole() }

func (b *batch) oldLoc(n *ir.Name) *location {
	if n.Canonical().Opt == nil {
		base.FatalfAt(n.Pos(), "%v has no location", n)
	}
	return n.Canonical().Opt.(*location)
}

func (e *escape) newLoc(n ir.Node, persists bool) *location {
	if e.curfn == nil {
		base.Fatalf("e.curfn isn't set")
	}
	if n != nil && n.Type() != nil && n.Type().NotInHeap() {
		base.ErrorfAt(n.Pos(), 0, "%v is incomplete (or unallocatable); stack allocation disallowed", n.Type())
	}

	if n != nil && n.Op() == ir.ONAME {
		if canon := n.(*ir.Name).Canonical(); n != canon {
			base.FatalfAt(n.Pos(), "newLoc on non-canonical %v (canonical is %v)", n, canon)
		}
	}
	loc := &location{
		n:         n,
		curfn:     e.curfn,
		loopDepth: e.loopDepth,
	}
	if loc.isName(ir.PPARAM) {
		loc.param = true
	} else if loc.isName(ir.PPARAMOUT) {
		loc.paramOut = true
	}

	if persists {
		loc.attrs |= attrPersists
	}
	e.allLocs = append(e.allLocs, loc)
	if n != nil {
		if n.Op() == ir.ONAME {
			n := n.(*ir.Name)
			if n.Class == ir.PPARAM && n.Curfn == nil {
				// ok; hidden parameter
			} else if n.Curfn != e.curfn {
				base.FatalfAt(n.Pos(), "curfn mismatch: %v != %v for %v", n.Curfn, e.curfn, n)
			}

			if n.Opt != nil {
				base.FatalfAt(n.Pos(), "%v already has a location", n)
			}
			n.Opt = loc
		}
	}
	return loc
}

// teeHole returns a new hole that flows into each hole of ks,
// similar to the Unix tee(1) command.
func (e *escape) teeHole(ks ...hole) hole {
	if len(ks) == 0 {
		return e.discardHole()
	}
	if len(ks) == 1 {
		return ks[0]
	}
	// TODO(mdempsky): Optimize if there's only one non-discard hole?

	// Given holes "l1 = _", "l2 = **_", "l3 = *_", ..., create a
	// new temporary location ltmp, wire it into place, and return
	// a hole for "ltmp = _".
	loc := e.newLoc(nil, false)
	for _, k := range ks {
		// N.B., "p = &q" and "p = &tmp; tmp = q" are not
		// semantically equivalent. To combine holes like "l1
		// = _" and "l2 = &_", we'd need to wire them as "l1 =
		// *ltmp" and "l2 = ltmp" and return "ltmp = &_"
		// instead.
		if k.derefs < 0 {
			base.Fatalf("teeHole: negative derefs")
		}

		e.flow(k, loc)
	}
	return loc.asHole()
}

// later returns a new hole that flows into k, but some time later.
// Its main effect is to prevent immediate reuse of temporary
// variables introduced during Order.
func (e *escape) later(k hole) hole {
	loc := e.newLoc(nil, true)
	e.flow(k, loc)
	return loc.asHole()
}

// Fmt is called from node printing to print information about escape analysis results.
func Fmt(n ir.Node) string {
	text := ""
	switch n.Esc() {
	case ir.EscUnknown:
		break

	case ir.EscHeap:
		text = "esc(h)"

	case ir.EscNone:
		text = "esc(no)"

	case ir.EscNever:
		text = "esc(N)"

	default:
		text = fmt.Sprintf("esc(%d)", n.Esc())
	}

	if n.Op() == ir.ONAME {
		n := n.(*ir.Name)
		if loc, ok := n.Opt.(*location); ok && loc.loopDepth != 0 {
			if text != "" {
				text += " "
			}
			text += fmt.Sprintf("ld(%d)", loc.loopDepth)
		}
	}

	return text
}
