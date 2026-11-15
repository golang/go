// Copyright 2026 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

package escape

import (
	"cmd/compile/internal/base"
	"cmd/compile/internal/ir"
)

func (b *batch) logosEscapeCapture(clo *ir.ClosureExpr, cv, n *ir.Name, loc *location) (skip, pooled bool) {
	skip = b.logosRegionSkipGroupCapture(clo, cv)
	pooled = capturedByPooledTryDefer(clo, n)
	n.SetByval(!loc.addrtaken && !loc.reassigned && (pooled || n.Type().Size() <= 128))
	if clo.DeferTryErr != nil && !pooled && n.Byval() {
		base.FatalfAt(n.Pos(), "pooled try defer captures its error result by value: %v", n)
	}
	return skip, pooled
}

func (b *batch) logosEscapeCaptureHole(k hole, pooled bool) hole {
	if pooled {
		k = b.heapHole()
	}
	return k
}

func (b *batch) logosEscapeFlowCapture(k hole, cv *ir.Name, loc *location, skip bool) {
	if skip {
		return
	}
	b.flow(k.note(cv, "captured by a closure"), loc)
}
