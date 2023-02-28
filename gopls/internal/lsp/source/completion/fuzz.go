// Copyright 2022 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

package completion

import (
	"fmt"
	"go/ast"
	"go/token"
	"go/types"
	"strings"

	"golang.org/x/tools/gopls/internal/lsp/protocol"
)

// golang/go#51089
// *testing.F deserves special treatment as member use is constrained:
// The arguments to f.Fuzz are determined by the arguments to a previous f.Add
// Inside f.Fuzz only f.Failed and f.Name are allowed.
// PJW: are there other packages where we can deduce usage constraints?

// if we find fuzz completions, then return true, as those are the only completions to offer
func (c *completer) fuzz(typ types.Type, mset *types.MethodSet, imp *importInfo, cb func(candidate), fset *token.FileSet) bool {
	// 1. inside f.Fuzz? (only f.Failed and f.Name)
	// 2. possible completing f.Fuzz?
	//    [Ident,SelectorExpr,Callexpr,ExprStmt,BlockiStmt,FuncDecl(Fuzz...)]
	// 3. before f.Fuzz, same (for 2., offer choice when looking at an F)

	// does the path contain FuncLit as arg to f.Fuzz CallExpr?
	inside := false
Loop:
	for i, n := range c.path {
		switch v := n.(type) {
		case *ast.CallExpr:
			if len(v.Args) != 1 {
				continue Loop
			}
			if _, ok := v.Args[0].(*ast.FuncLit); !ok {
				continue
			}
			if s, ok := v.Fun.(*ast.SelectorExpr); !ok || s.Sel.Name != "Fuzz" {
				continue
			}
			if i > 2 { // avoid t.Fuzz itself in tests
				inside = true
				break Loop
			}
		}
	}
	if inside {
		for i := 0; i < mset.Len(); i++ {
			o := mset.At(i).Obj()
			if o.Name() == "Failed" || o.Name() == "Name" {
				cb(candidate{
					obj:         o,
					score:       stdScore,
					imp:         imp,
					addressable: true,
				})
			}
		}
		return true
	}
	// if it could be t.Fuzz, look for the preceding t.Add
	id, ok := c.path[0].(*ast.Ident)
	if ok && strings.HasPrefix("Fuzz", id.Name) {
		var add *ast.CallExpr
		f := func(n ast.Node) bool {
			if n == nil {
				return true
			}
			call, ok := n.(*ast.CallExpr)
			if !ok {
				return true
			}
			s, ok := call.Fun.(*ast.SelectorExpr)
			if !ok {
				return true
			}
			if s.Sel.Name != "Add" {
				return true
			}
			// Sel.X should be of type *testing.F
			got := c.pkg.GetTypesInfo().Types[s.X]
			if got.Type.String() == "*testing.F" {
				add = call
			}
			return false // because we're done...
		}
		// look at the enclosing FuzzFoo functions
		if len(c.path) < 2 {
			return false
		}
		n := c.path[len(c.path)-2]
		if _, ok := n.(*ast.FuncDecl); !ok {
			// the path should start with ast.File, ast.FuncDecl, ...
			// but it didn't, so give up
			return false
		}
		ast.Inspect(n, f)
		if add == nil {
			// looks like f.Fuzz without a preceding f.Add.
			// let the regular completion handle it.
			return false
		}

		lbl := "Fuzz(func(t *testing.T"
		for i, a := range add.Args {
			info := c.pkg.GetTypesInfo().TypeOf(a)
			if info == nil {
				return false // How could this happen, but better safe than panic.
			}
			lbl += fmt.Sprintf(", %c %s", 'a'+i, info)
		}
		lbl += ")"
		xx := CompletionItem{
			Label:         lbl,
			InsertText:    lbl,
			Kind:          protocol.FunctionCompletion,
			Depth:         0,
			Score:         10, // pretty confident the user should see this
			Documentation: "argument types from f.Add",
			isSlice:       false,
		}
		c.items = append(c.items, xx)
		for i := 0; i < mset.Len(); i++ {
			o := mset.At(i).Obj()
			if o.Name() != "Fuzz" {
				cb(candidate{
					obj:         o,
					score:       stdScore,
					imp:         imp,
					addressable: true,
				})
			}
		}
		return true // done
	}
	// let the standard processing take care of it instead
	return false
}
