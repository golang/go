// Copyright 2021 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

package noder

import (
	"strings"

	"cmd/compile/internal/base"
	"cmd/compile/internal/ir"
	"cmd/compile/internal/syntax"
	"cmd/compile/internal/types2"
)

// recordScopes populates fn.Parents and fn.Marks based on the scoping
// information provided by types2.
func (g *irgen) recordScopes(fn *ir.Func, sig *syntax.FuncType) {
	scope, ok := g.info.Scopes[sig]
	if !ok {
		base.FatalfAt(fn.Pos(), "missing scope for %v", fn)
	}

	for i, n := 0, scope.NumChildren(); i < n; i++ {
		g.walkScope(scope.Child(i))
	}

	g.marker.WriteTo(fn)
}

func (g *irgen) walkScope(scope *types2.Scope) bool {
	// types2 doesn't provide a proper API for determining the
	// lexical element a scope represents, so we have to resort to
	// string matching. Conveniently though, this allows us to
	// skip both function types and function literals, neither of
	// which are interesting to us here.
	if strings.HasPrefix(scope.String(), "function scope ") {
		return false
	}

	g.marker.Push(g.pos(scope))

	haveVars := false
	for _, name := range scope.Names() {
		if obj, ok := scope.Lookup(name).(*types2.Var); ok && obj.Name() != "_" {
			haveVars = true
			break
		}
	}

	for i, n := 0, scope.NumChildren(); i < n; i++ {
		if g.walkScope(scope.Child(i)) {
			haveVars = true
		}
	}

	if haveVars {
		g.marker.Pop(g.end(scope))
	} else {
		g.marker.Unpush()
	}

	return haveVars
}
