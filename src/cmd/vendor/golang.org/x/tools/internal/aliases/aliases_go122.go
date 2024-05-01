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
	"os"
	"strings"
	"sync"
)

// Alias is an alias of types.Alias.
type Alias = types.Alias

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

// enabled returns true when types.Aliases are enabled.
func enabled() bool {
	// Use the gotypesalias value in GODEBUG if set.
	godebug := os.Getenv("GODEBUG")
	value := -1 // last set value.
	for _, f := range strings.Split(godebug, ",") {
		switch f {
		case "gotypesalias=1":
			value = 1
		case "gotypesalias=0":
			value = 0
		}
	}
	switch value {
	case 0:
		return false
	case 1:
		return true
	default:
		return aliasesDefault()
	}
}

// aliasesDefault reports if aliases are enabled by default.
func aliasesDefault() bool {
	// Dynamically check if Aliases will be produced from go/types.
	aliasesDefaultOnce.Do(func() {
		fset := token.NewFileSet()
		f, _ := parser.ParseFile(fset, "a.go", "package p; type A = int", 0)
		pkg, _ := new(types.Config).Check("p", fset, []*ast.File{f}, nil)
		_, gotypesaliasDefault = pkg.Scope().Lookup("A").Type().(*types.Alias)
	})
	return gotypesaliasDefault
}

var gotypesaliasDefault bool
var aliasesDefaultOnce sync.Once
