// Copyright 2015 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

package main

import (
	"go/ast"
	"strings"
	"unicode"
	"unicode/utf8"

	"golang.org/x/tools/go/types"
)

func init() {
	register("example",
		"check for common mistaken usages of documentation examples",
		checkExample,
		funcDecl)
}

func isExampleSuffix(s string) bool {
	r, size := utf8.DecodeRuneInString(s)
	return size > 0 && unicode.IsLower(r)
}

// checkExample walks the documentation example functions checking for common
// mistakes of misnamed functions, failure to map functions to existing
// identifiers, etc.
func checkExample(f *File, node ast.Node) {
	if !strings.HasSuffix(f.name, "_test.go") {
		return
	}
	var (
		pkg     = f.pkg
		pkgName = pkg.typesPkg.Name()
		scopes  = []*types.Scope{pkg.typesPkg.Scope()}
		lookup  = func(name string) types.Object {
			for _, scope := range scopes {
				if o := scope.Lookup(name); o != nil {
					return o
				}
			}
			return nil
		}
	)
	if strings.HasSuffix(pkgName, "_test") {
		// Treat 'package foo_test' as an alias for 'package foo'.
		var (
			basePkg = strings.TrimSuffix(pkgName, "_test")
			pkg     = f.pkg
		)
		for _, p := range pkg.typesPkg.Imports() {
			if p.Name() == basePkg {
				scopes = append(scopes, p.Scope())
				break
			}
		}
	}
	fn, ok := node.(*ast.FuncDecl)
	if !ok {
		// Ignore non-functions.
		return
	}
	var (
		fnName = fn.Name.Name
		report = func(format string, args ...interface{}) { f.Badf(node.Pos(), format, args...) }
	)
	if fn.Recv != nil || !strings.HasPrefix(fnName, "Example") {
		// Ignore methods and types not named "Example".
		return
	}
	if params := fn.Type.Params; len(params.List) != 0 {
		report("%s should be niladic", fnName)
	}
	if results := fn.Type.Results; results != nil && len(results.List) != 0 {
		report("%s should return nothing", fnName)
	}
	if fnName == "Example" {
		// Nothing more to do.
		return
	}
	if filesRun && !includesNonTest {
		// The coherence checks between a test and the package it tests
		// will report false positives if no non-test files have
		// been provided.
		return
	}
	var (
		exName = strings.TrimPrefix(fnName, "Example")
		elems  = strings.SplitN(exName, "_", 3)
		ident  = elems[0]
		obj    = lookup(ident)
	)
	if ident != "" && obj == nil {
		// Check ExampleFoo and ExampleBadFoo.
		report("%s refers to unknown identifier: %s", fnName, ident)
		// Abort since obj is absent and no subsequent checks can be performed.
		return
	}
	if elemCnt := strings.Count(exName, "_"); elemCnt == 0 {
		// Nothing more to do.
		return
	}
	mmbr := elems[1]
	if ident == "" {
		// Check Example_suffix and Example_BadSuffix.
		if residual := strings.TrimPrefix(exName, "_"); !isExampleSuffix(residual) {
			report("%s has malformed example suffix: %s", fnName, residual)
		}
		return
	}
	if !isExampleSuffix(mmbr) {
		// Check ExampleFoo_Method and ExampleFoo_BadMethod.
		if obj, _, _ := types.LookupFieldOrMethod(obj.Type(), true, obj.Pkg(), mmbr); obj == nil {
			report("%s refers to unknown field or method: %s.%s", fnName, ident, mmbr)
		}
	}
	if len(elems) == 3 && !isExampleSuffix(elems[2]) {
		// Check ExampleFoo_Method_suffix and ExampleFoo_Method_Badsuffix.
		report("%s has malformed example suffix: %s", fnName, elems[2])
	}
	return
}
