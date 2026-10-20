// Copyright 2026 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

package lint

import (
	"cmd/compile/internal/syntax"
	"cmd/compile/internal/types2"
)

// timeAfterInLoop is perf::time_after_in_loop (SPEC Sec 1.2, #170, #172): a
// call to time.After inside the body of a for statement. Each pass through
// the body creates a timer that stays allocated until it fires, and the
// compiler never hoists the call out of the loop, so the program behaves the
// same at every optimization level. A function literal inside the body
// counts as inside it; the header of the loop (init, condition, post
// statement, range operand) runs outside the body. The rule is
// language-neutral: it reads Go syntax and Go types only.
var timeAfterInLoop = Rule{
	ID:    "perf::time_after_in_loop",
	Group: Perf,
	Go:    true,
	Check: checkTimeAfterInLoop,
}

// checkTimeAfterInLoop reports each time.After call inside a loop body.
func checkTimeAfterInLoop(pass *Pass) {
	for _, f := range pass.Files {
		w := &timeAfterWalk{pass: pass, file: f}
		w.visit(f)
	}
}

// timeAfterWalk walks one file; depth counts the loop bodies around the node
// being visited.
type timeAfterWalk struct {
	pass  *Pass
	file  *syntax.File
	depth int
}

// visit inspects the subtree n. A for statement is handled by visitFor,
// which splits its header from its body.
func (w *timeAfterWalk) visit(n syntax.Node) {
	syntax.Inspect(n, func(c syntax.Node) bool {
		switch x := c.(type) {
		case *syntax.ForStmt:
			w.visitFor(x)
			return false
		case *syntax.CallExpr:
			if w.depth > 0 && isTimeAfter(w.pass, x) {
				w.report(x)
			}
		}
		return true
	})
}

// visitFor visits the header of loop at the enclosing depth and its body one
// loop deeper.
func (w *timeAfterWalk) visitFor(loop *syntax.ForStmt) {
	if loop.Init != nil {
		w.visit(loop.Init)
	}
	if loop.Cond != nil {
		w.visit(loop.Cond)
	}
	if loop.Post != nil {
		w.visit(loop.Post)
	}
	if loop.Body != nil {
		w.depth++
		w.visit(loop.Body)
		w.depth--
	}
}

// report flags one call, spanning through its closing parenthesis.
func (w *timeAfterWalk) report(call *syntax.CallExpr) {
	w.pass.Report(Diagnostic{
		Pos:     syntax.StartPos(call),
		End:     w.pass.CallEnd(w.file, call),
		Message: "time.After in a loop body creates a timer on every iteration; create one time.Timer before the loop and Reset it",
	})
}

// isTimeAfter reports whether call invokes the function After of package
// time, not a method or a function of another package named After.
func isTimeAfter(pass *Pass, call *syntax.CallExpr) bool {
	fn := calleeFunc(pass, call)
	if fn == nil || fn.Name() != "After" || fn.Pkg() == nil || fn.Pkg().Path() != "time" {
		return false
	}
	sig, ok := fn.Type().(*types2.Signature)
	return ok && sig.Recv() == nil
}
