// Copyright 2015 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

// This file defines the check for unused results of calls to certain
// pure functions.

package main

import (
	"flag"
	"go/ast"
	"go/token"
	"strings"

	"golang.org/x/tools/go/types"
)

var unusedFuncsFlag = flag.String("unusedfuncs",
	"errors.New,fmt.Errorf,fmt.Sprintf,fmt.Sprint,sort.Reverse",
	"comma-separated list of functions whose results must be used")

var unusedStringMethodsFlag = flag.String("unusedstringmethods",
	"Error,String",
	"comma-separated list of names of methods of type func() string whose results must be used")

func init() {
	register("unusedresult",
		"check for unused result of calls to functions in -unusedfuncs list and methods in -unusedstringmethods list",
		checkUnusedResult,
		exprStmt)
}

// func() string
var sigNoArgsStringResult = types.NewSignature(nil, nil,
	types.NewTuple(types.NewVar(token.NoPos, nil, "", types.Typ[types.String])),
	false)

var unusedFuncs = make(map[string]bool)
var unusedStringMethods = make(map[string]bool)

func initUnusedFlags() {
	commaSplit := func(s string, m map[string]bool) {
		if s != "" {
			for _, name := range strings.Split(s, ",") {
				if len(name) == 0 {
					flag.Usage()
				}
				m[name] = true
			}
		}
	}
	commaSplit(*unusedFuncsFlag, unusedFuncs)
	commaSplit(*unusedStringMethodsFlag, unusedStringMethods)
}

func checkUnusedResult(f *File, n ast.Node) {
	call, ok := unparen(n.(*ast.ExprStmt).X).(*ast.CallExpr)
	if !ok {
		return // not a call statement
	}
	fun := unparen(call.Fun)

	if f.pkg.types[fun].IsType() {
		return // a conversion, not a call
	}

	selector, ok := fun.(*ast.SelectorExpr)
	if !ok {
		return // neither a method call nor a qualified ident
	}

	sel, ok := f.pkg.selectors[selector]
	if ok && sel.Kind() == types.MethodVal {
		// method (e.g. foo.String())
		obj := sel.Obj().(*types.Func)
		sig := sel.Type().(*types.Signature)
		if types.Identical(sig, sigNoArgsStringResult) {
			if unusedStringMethods[obj.Name()] {
				f.Badf(call.Lparen, "result of (%s).%s call not used",
					sig.Recv().Type(), obj.Name())
			}
		}
	} else if !ok {
		// package-qualified function (e.g. fmt.Errorf)
		obj, _ := f.pkg.uses[selector.Sel]
		if obj, ok := obj.(*types.Func); ok {
			qname := obj.Pkg().Path() + "." + obj.Name()
			if unusedFuncs[qname] {
				f.Badf(call.Lparen, "result of %v call not used", qname)
			}
		}
	}
}
