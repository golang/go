// +build ignore

// Copyright 2012 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

// This file contains the test for unkeyed struct literals.

package main

import (
	"cmd/vet/internal/whitelist"
	"flag"
	"go/ast"
	"go/types"
	"strings"
)

var compositeWhiteList = flag.Bool("compositewhitelist", true, "use composite white list; for testing only")

func init() {
	register("composites",
		"check that composite literals of types from imported packages use field-keyed elements",
		checkUnkeyedLiteral,
		compositeLit)
}

// checkUnkeyedLiteral checks if a composite literal is a struct literal with
// unkeyed fields.
func checkUnkeyedLiteral(f *File, node ast.Node) {
	cl := node.(*ast.CompositeLit)

	typ := f.pkg.types[cl].Type
	if typ == nil {
		// cannot determine composite literals' type, skip it
		return
	}
	typeName := typ.String()
	if *compositeWhiteList && whitelist.UnkeyedLiteral[typeName] {
		// skip whitelisted types
		return
	}
	under := typ.Underlying()
	for {
		ptr, ok := under.(*types.Pointer)
		if !ok {
			break
		}
		under = ptr.Elem().Underlying()
	}
	if _, ok := under.(*types.Struct); !ok {
		// skip non-struct composite literals
		return
	}
	if isLocalType(f, typ) {
		// allow unkeyed locally defined composite literal
		return
	}

	// check if the CompositeLit contains an unkeyed field
	allKeyValue := true
	for _, e := range cl.Elts {
		if _, ok := e.(*ast.KeyValueExpr); !ok {
			allKeyValue = false
			break
		}
	}
	if allKeyValue {
		// all the composite literal fields are keyed
		return
	}

	f.Badf(cl.Pos(), "%s composite literal uses unkeyed fields", typeName)
}

func isLocalType(f *File, typ types.Type) bool {
	switch x := typ.(type) {
	case *types.Struct:
		// struct literals are local types
		return true
	case *types.Pointer:
		return isLocalType(f, x.Elem())
	case *types.Named:
		// names in package foo are local to foo_test too
		return strings.TrimSuffix(x.Obj().Pkg().Path(), "_test") == strings.TrimSuffix(f.pkg.path, "_test")
	}
	return false
}
