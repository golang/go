// Copyright 2018 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

// This file defines the check for passing non-pointer or non-interface
// types to unmarshal and decode functions.

package main

import (
	"go/ast"
	"go/types"
	"strings"
)

func init() {
	register("unmarshal",
		"check for passing non-pointer or non-interface types to unmarshal and decode functions",
		checkUnmarshalArg,
		callExpr)
}

var pointerArgFuncs = map[string]int{
	"encoding/json.Unmarshal":         1,
	"(*encoding/json.Decoder).Decode": 0,
	"(*encoding/gob.Decoder).Decode":  0,
	"encoding/xml.Unmarshal":          1,
	"(*encoding/xml.Decoder).Decode":  0,
}

func checkUnmarshalArg(f *File, n ast.Node) {
	call, ok := n.(*ast.CallExpr)
	if !ok {
		return // not a call statement
	}
	fun := unparen(call.Fun)

	if f.pkg.types[fun].IsType() {
		return // a conversion, not a call
	}

	info := &types.Info{Uses: f.pkg.uses, Selections: f.pkg.selectors}
	name := callName(info, call)

	arg, ok := pointerArgFuncs[name]
	if !ok {
		return // not a function we are interested in
	}

	if len(call.Args) < arg+1 {
		return // not enough arguments, e.g. called with return values of another function
	}

	typ := f.pkg.types[call.Args[arg]]

	if typ.Type == nil {
		return // type error prevents further analysis
	}

	switch typ.Type.Underlying().(type) {
	case *types.Pointer, *types.Interface:
		return
	}

	shortname := name[strings.LastIndexByte(name, '.')+1:]
	switch arg {
	case 0:
		f.Badf(call.Lparen, "call of %s passes non-pointer", shortname)
	case 1:
		f.Badf(call.Lparen, "call of %s passes non-pointer as second argument", shortname)
	}
}
