// Copyright 2024 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

//go:build go1.22
// +build go1.22

package aliases

import (
	"go/ast"
	"go/parser"
	"go/token"
	"go/types"
)

// Alias is an alias of types.Alias.
type Alias = types.Alias

// Rhs returns the type on the right-hand side of the alias declaration.
func Rhs(alias *Alias) types.Type {
	if alias, ok := any(alias).(interface{ Rhs() types.Type }); ok {
		return alias.Rhs() // go1.23+
	}

	// go1.22's Alias didn't have the Rhs method,
	// so Unalias is the best we can do.
	return Unalias(alias)
}

// Unalias is a wrapper of types.Unalias.
func Unalias(t types.Type) types.Type { return types.Unalias(t) }

// newAlias is an internal alias around types.NewAlias.
// Direct usage is discouraged as the moment.
// Try to use NewAlias instead.
func newAlias(tname *types.TypeName, rhs types.Type) *Alias {
	a := types.NewAlias(tname, rhs)
	// TODO(go.dev/issue/65455): Remove kludgy workaround to set a.actual as a side-effect.
	Unalias(a)
	return a
}

// Enabled reports whether [NewAlias] should create [types.Alias] types.
//
// This function is expensive! Call it sparingly.
func Enabled() bool {
	// The only reliable way to compute the answer is to invoke go/types.
	// We don't parse the GODEBUG environment variable, because
	// (a) it's tricky to do so in a manner that is consistent
	//     with the godebug package; in particular, a simple
	//     substring check is not good enough. The value is a
	//     rightmost-wins list of options. But more importantly:
	// (b) it is impossible to detect changes to the effective
	//     setting caused by os.Setenv("GODEBUG"), as happens in
	//     many tests. Therefore any attempt to cache the result
	//     is just incorrect.
	fset := token.NewFileSet()
	f, _ := parser.ParseFile(fset, "a.go", "package p; type A = int", 0)
	pkg, _ := new(types.Config).Check("p", fset, []*ast.File{f}, nil)
	_, enabled := pkg.Scope().Lookup("A").Type().(*types.Alias)
	return enabled
}
