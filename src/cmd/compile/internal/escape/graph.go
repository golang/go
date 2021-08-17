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

// An location represents an abstract location that stores a Go
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

	// queued is used by walkAll to track whether this location is
	// in the walk queue.
	queued bool

	// escapes reports whether the represented variable's address
	// escapes; that is, whether the variable must be heap
	// allocated.
	escapes bool

	// transient reports whether the represented expression's
	// address does not outlive the statement; that is, whether
	// its storage can be immediately reused.
	transient bool

	// paramEsc records the represented parameter's leak set.
	paramEsc leaks

	captured   bool // has a closure captured this variable?
	reassigned bool // has this variable been reassigned?
	addrtaken  bool // has this variable's address been taken?
}

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
	if !sink.escapes && sink.isName(ir.PPARAMOUT) && sink.curfn == l.curfn {
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
	if dst.escapes && k.derefs < 0 { // dst = &src
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
		src.escapes = true
		return
	}

	// TODO(mdempsky): Deduplicate edges?
	dst.edges = append(dst.edges, edge{src: src, derefs: k.derefs, notes: k.notes})
}

func (b *batch) heapHole() hole    { return b.heapLoc.asHole() }
func (b *batch) discardHole() hole { return b.blankLoc.asHole() }

func (b *batch) oldLoc(n *ir.Name) *location {
	if n.Canonical().Opt == nil {
		base.Fatalf("%v has no location", n)
	}
	return n.Canonical().Opt.(*location)
}

func (e *escape) newLoc(n ir.Node, transient bool) *location {
	if e.curfn == nil {
		base.Fatalf("e.curfn isn't set")
	}
	if n != nil && n.Type() != nil && n.Type().NotInHeap() {
		base.ErrorfAt(n.Pos(), "%v is incomplete (or unallocatable); stack allocation disallowed", n.Type())
	}

	if n != nil && n.Op() == ir.ONAME {
		if canon := n.(*ir.Name).Canonical(); n != canon {
			base.Fatalf("newLoc on non-canonical %v (canonical is %v)", n, canon)
		}
	}
	loc := &location{
		n:         n,
		curfn:     e.curfn,
		loopDepth: e.loopDepth,
		transient: transient,
	}
	e.allLocs = append(e.allLocs, loc)
	if n != nil {
		if n.Op() == ir.ONAME {
			n := n.(*ir.Name)
			if n.Class == ir.PPARAM && n.Curfn == nil {
				// ok; hidden parameter
			} else if n.Curfn != e.curfn {
				base.Fatalf("curfn mismatch: %v != %v for %v", n.Curfn, e.curfn, n)
			}

			if n.Opt != nil {
				base.Fatalf("%v already has a location", n)
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
	loc := e.newLoc(nil, true)
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
	loc := e.newLoc(nil, false)
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
