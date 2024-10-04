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

// TypeParams returns the type parameter list of the alias.
func TypeParams(alias *Alias) *types.TypeParamList {
	if alias, ok := any(alias).(interface{ TypeParams() *types.TypeParamList }); ok {
		return alias.TypeParams() // go1.23+
	}
	return nil
}

// SetTypeParams sets the type parameters of the alias type.
func SetTypeParams(alias *Alias, tparams []*types.TypeParam) {
	if alias, ok := any(alias).(interface {
		SetTypeParams(tparams []*types.TypeParam)
	}); ok {
		alias.SetTypeParams(tparams) // go1.23+
	} else if len(tparams) > 0 {
		panic("cannot set type parameters of an Alias type in go1.22")
	}
}

// TypeArgs returns the type arguments used to instantiate the Alias type.
func TypeArgs(alias *Alias) *types.TypeList {
	if alias, ok := any(alias).(interface{ TypeArgs() *types.TypeList }); ok {
		return alias.TypeArgs() // go1.23+
	}
	return nil // empty (go1.22)
}

// Origin returns the generic Alias type of which alias is an instance.
// If alias is not an instance of a generic alias, Origin returns alias.
func Origin(alias *Alias) *Alias {
	if alias, ok := any(alias).(interface{ Origin() *types.Alias }); ok {
		return alias.Origin() // go1.23+
	}
	return alias // not an instance of a generic alias (go1.22)
}

// Unalias is a wrapper of types.Unalias.
func Unalias(t types.Type) types.Type { return types.Unalias(t) }

// newAlias is an internal alias around types.NewAlias.
// Direct usage is discouraged as the moment.
// Try to use NewAlias instead.
func newAlias(tname *types.TypeName, rhs types.Type, tparams []*types.TypeParam) *Alias {
	a := types.NewAlias(tname, rhs)
	SetTypeParams(a, tparams)
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
