// Copyright 2026 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

package escape

import (
	"fmt"

	"cmd/compile/internal/base"
	"cmd/compile/internal/ir"
	"cmd/internal/src"
)

// LogosHintDecision is one escape decision that -logos.mode=hints reports
// (SPEC Sec 21.2): where the allocation sits, whether it reaches the heap,
// the text -m prints for it, and the reason the compiler forced it onto the
// heap when it had one.
type LogosHintDecision struct {
	Pos     src.XPos
	Heap    bool
	Message string
	Reason  string
}

var logosHintDecisions []LogosHintDecision

// LogosHintDecisions returns the decisions recorded so far, in analysis
// order. A source allocation that the compiler analyzes more than once, such
// as an inlined body, appears once per analysis.
func LogosHintDecisions() []LogosHintDecision {
	return logosHintDecisions
}

// logosHintOps are the nodes that allocate: the ones a programmer writes as
// an allocation, and the conversions and concatenations that copy.
var logosHintOps = map[ir.Op]bool{
	ir.OPTRLIT:    true,
	ir.ONEW:       true,
	ir.OMAKESLICE: true,
	ir.OMAKEMAP:   true,
	ir.OMAKECHAN:  true,
	ir.OSLICELIT:  true,
	ir.OMAPLIT:    true,
	ir.OCLOSURE:   true,
	ir.OAPPEND:    true,
	ir.OSTR2BYTES: true,
	ir.OBYTES2STR: true,
	ir.OSTR2RUNES: true,
	ir.ORUNES2STR: true,
	ir.OADDSTR:    true,
}

// logosCollectHints records the decisions of the solved batch in hints mode.
// It reads the results the way the -m diagnostics of finish do, and prints
// and changes nothing.
func (b *batch) logosCollectHints() {
	if base.LogosMode != "hints" {
		return
	}
	for _, loc := range b.allLocs {
		if d, ok := logosHintOf(loc); ok {
			logosHintDecisions = append(logosHintDecisions, d)
		}
	}
}

// logosHintOf returns the decision for loc, or false when loc is not an
// allocation or a named variable that moved to the heap.
func logosHintOf(loc *location) (LogosHintDecision, bool) {
	n := loc.n
	heap := loc.hasAttr(attrEscapes)
	if n == nil || !n.Pos().IsKnown() || !logosHintNode(n, heap) {
		return LogosHintDecision{}, false
	}
	d := LogosHintDecision{Pos: n.Pos(), Heap: heap, Message: logosHintMessage(n, heap)}
	if heap {
		d.Reason = HeapAllocReason(n)
	}
	return d, true
}

// logosHintNode reports whether n is reported: a user variable only when it
// moved to the heap, an allocation node always, and never a go or defer
// wrapper closure, which -m omits too.
func logosHintNode(n ir.Node, heap bool) bool {
	if name, ok := n.(*ir.Name); ok {
		return heap && !name.AutoTemp() && name.Sym() != nil && !name.Sym().IsBlank()
	}
	if clo, ok := n.(*ir.ClosureExpr); ok && clo.Func.Wrapper() {
		return false
	}
	return logosHintOps[n.Op()]
}

// logosHintMessage returns the -m text for n and its decision.
func logosHintMessage(n ir.Node, heap bool) string {
	switch {
	case n.Op() == ir.ONAME:
		return fmt.Sprintf("moved to heap: %v", n)
	case n.Op() == ir.OAPPEND && heap:
		return "append escapes to heap"
	case n.Op() == ir.OAPPEND:
		return "append does not escape"
	case heap:
		return fmt.Sprintf("%v escapes to heap", n)
	}
	return fmt.Sprintf("%v does not escape", n)
}
