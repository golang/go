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

type LogosHintDecision struct {
	Pos     src.XPos
	Heap    bool
	Arena   bool
	Message string
	Reason  string
}

var logosHintDecisions []LogosHintDecision

func LogosHintDecisions() []LogosHintDecision {
	return logosHintDecisions
}

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

func logosArenaHint(n ir.Node, fn *ir.Func) bool {
	if !LogosArenaEnabled() || fn == nil {
		return false
	}
	m, ok := n.(*ir.MakeExpr)
	if !ok || m.Op() != ir.OMAKESLICE {
		return false
	}
	return LogosArenaMake(m, fn) || LogosArenaConstMake(m, fn) || LogosArenaRetMake(m, fn)
}

func logosHintOf(loc *location) (LogosHintDecision, bool) {
	n := loc.n
	heap := loc.hasAttr(attrEscapes)
	if n == nil || !n.Pos().IsKnown() || !logosHintNode(n, heap) {
		return LogosHintDecision{}, false
	}
	arena := logosArenaHint(n, loc.curfn)
	d := LogosHintDecision{
		Pos:     n.Pos(),
		Heap:    heap,
		Arena:   arena,
		Message: logosHintMessage(n, heap),
	}
	if heap {
		d.Reason = HeapAllocReason(n)
	}
	return d, true
}

func logosHintNode(n ir.Node, heap bool) bool {
	if name, ok := n.(*ir.Name); ok {
		return heap && !name.AutoTemp() && name.Sym() != nil && !name.Sym().IsBlank()
	}
	if clo, ok := n.(*ir.ClosureExpr); ok && clo.Func.Wrapper() {
		return false
	}
	return logosHintOps[n.Op()]
}

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
