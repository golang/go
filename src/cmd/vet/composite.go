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
		"check that composite literals used field-keyed elements",
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
	if _, ok := typ.Underlying().(*types.Struct); !ok {
		// skip non-struct composite literals
		return
	}
	if isLocalType(f, typeName) {
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

func isLocalType(f *File, typeName string) bool {
	if strings.HasPrefix(typeName, "struct{") {
		// struct literals are local types
		return true
	}

	pkgname := f.pkg.path
	if strings.HasPrefix(typeName, pkgname+".") {
		return true
	}

	// treat types as local inside test packages with _test name suffix
	if strings.HasSuffix(pkgname, "_test") {
		pkgname = pkgname[:len(pkgname)-len("_test")]
	}
	return strings.HasPrefix(typeName, pkgname+".")
}
