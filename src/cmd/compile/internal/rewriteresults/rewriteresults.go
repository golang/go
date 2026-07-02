// Copyright 2026 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

// Package rewriteresults rewrites local variables returned directly by a
// function to use the corresponding result parameter's storage.
package rewriteresults

import (
	"cmd/compile/internal/base"
	"cmd/compile/internal/ir"
	"cmd/compile/internal/types"
	"fmt"
	"os"
)

// Funcs applies the rewriteresults pass to fns.
func Funcs(fns []*ir.Func) {
	if base.Flag.N != 0 || base.Debug.RewriteResults == 0 {
		return
	}

	for _, fn := range fns {
		rewrite(fn)
	}
}

func rewrite(fn *ir.Func) {
	if fn == nil || len(fn.Body) == 0 {
		return
	}

	var returns []*ir.ReturnStmt
	hasDefer := false
	ir.VisitList(fn.Body, func(n ir.Node) {
		switch n := n.(type) {
		case *ir.ReturnStmt:
			returns = append(returns, n)
		case *ir.GoDeferStmt:
			if n.Op() == ir.ODEFER {
				hasDefer = true
			}
		}
	})
	if hasDefer || len(returns) == 0 {
		return
	}

	results := fn.Type().Results()
	for _, ret := range returns {
		if len(ret.Results) == 0 || len(ret.Results) != len(results) {
			return
		}
	}

	// candidates maps each local variable to the result slot whose storage
	// it can use.
	candidates := make(map[*ir.Name]*ir.Name)
	conflicts := make(map[*ir.Name]bool)
	for i, result := range results {
		// If the result is already named, source already has access to
		// its storage; leave those functions alone for now.
		if !isAnonymousResult(result) {
			continue
		}
		out := result.Nname.(*ir.Name)

		var local *ir.Name
		for _, ret := range returns {
			n, ok := ret.Results[i].(*ir.Name)
			if !ok {
				continue
			}
			if !isCandidateLocal(n, result) {
				continue
			}
			if local == nil {
				local = n
			} else if local != n {
				local = nil
				break
			}
		}
		if local == nil {
			continue
		}
		if prev, ok := candidates[local]; ok && prev != out {
			conflicts[local] = true
			continue
		}
		candidates[local] = out
	}
	if len(candidates) == 0 {
		return
	}
	for local := range conflicts {
		delete(candidates, local)
	}
	if len(candidates) == 0 {
		return
	}

	captured := make(map[*ir.Name]bool)
	ir.VisitList(fn.Body, func(n ir.Node) {
		if n, ok := n.(*ir.ClosureExpr); ok {
			for _, cv := range n.Func.ClosureVars {
				captured[cv.Canonical()] = true
			}
		}
	})
	for local := range candidates {
		if captured[local] {
			delete(candidates, local)
		}
	}
	if len(candidates) == 0 {
		return
	}
	for local, out := range candidates {
		if local.Addrtaken() {
			out.SetAddrtaken(true)
		}
		out.SetUsed(true)
		out.SetEsc(local.Esc())
	}

	if base.Debug.RewriteResults > 1 {
		for local, out := range candidates {
			fmt.Fprintf(os.Stderr, "rewriteresults: %v: %v => %v\n", ir.FuncName(fn), local, out)
		}
	}

	var edit func(ir.Node) ir.Node
	edit = func(n ir.Node) ir.Node {
		switch n := n.(type) {
		case nil:
			return nil
		case *ir.Name:
			if out, ok := candidates[n]; ok {
				return out
			}
			return n
		}

		ir.EditChildren(n, edit)
		return n
	}

	for i, n := range fn.Body {
		fn.Body[i] = edit(n)
	}
}

func isCandidateLocal(n *ir.Name, result *types.Field) bool {
	return n.Class == ir.PAUTO &&
		!n.AutoTemp() &&
		n.Esc() != ir.EscHeap &&
		isBareDecl(n) &&
		types.Identical(n.Type(), result.Type)
}

func isBareDecl(n *ir.Name) bool {
	// Locals declared without an explicit initializer have no defining
	// assignment; their nil-RHS assignment only supplies the zero value.
	return n.Defn == nil
}

func isAnonymousResult(result *types.Field) bool {
	return result.Sym == nil || result.Sym.IsBlank()
}
