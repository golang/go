// Copyright 2018 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

// The unmarshal package defines an Analyzer that checks for passing
// non-pointer or non-interface types to unmarshal and decode functions.
package unmarshal

import (
	"go/ast"
	"go/types"

	"golang.org/x/tools/go/analysis"
	"golang.org/x/tools/go/analysis/passes/inspect"
	"golang.org/x/tools/go/ast/inspector"
	"golang.org/x/tools/go/types/typeutil"
)

const Doc = `report passing non-pointer or non-interface values to unmarshal

The unmarshal analysis reports calls to functions such as json.Unmarshal
in which the argument type is not a pointer or an interface.`

var Analyzer = &analysis.Analyzer{
	Name:     "unmarshal",
	Doc:      Doc,
	Requires: []*analysis.Analyzer{inspect.Analyzer},
	Run:      run,
}

func run(pass *analysis.Pass) (interface{}, error) {
	switch pass.Pkg.Path() {
	case "encoding/gob", "encoding/json", "encoding/xml":
		// These packages know how to use their own APIs.
		// Sometimes they are testing what happens to incorrect programs.
		return nil, nil
	}

	inspect := pass.ResultOf[inspect.Analyzer].(*inspector.Inspector)

	nodeFilter := []ast.Node{
		(*ast.CallExpr)(nil),
	}
	inspect.Preorder(nodeFilter, func(n ast.Node) {
		call := n.(*ast.CallExpr)
		fn := typeutil.StaticCallee(pass.TypesInfo, call)
		if fn == nil {
			return // not a static call
		}

		// Classify the callee (without allocating memory).
		argidx := -1
		recv := fn.Type().(*types.Signature).Recv()
		if fn.Name() == "Unmarshal" && recv == nil {
			// "encoding/json".Unmarshal
			//  "encoding/xml".Unmarshal
			switch fn.Pkg().Path() {
			case "encoding/json", "encoding/xml":
				argidx = 1 // func([]byte, interface{})
			}
		} else if fn.Name() == "Decode" && recv != nil {
			// (*"encoding/json".Decoder).Decode
			// (* "encoding/gob".Decoder).Decode
			// (* "encoding/xml".Decoder).Decode
			t := recv.Type()
			if ptr, ok := t.(*types.Pointer); ok {
				t = ptr.Elem()
			}
			tname := t.(*types.Named).Obj()
			if tname.Name() == "Decoder" {
				switch tname.Pkg().Path() {
				case "encoding/json", "encoding/xml", "encoding/gob":
					argidx = 0 // func(interface{})
				}
			}
		}
		if argidx < 0 {
			return // not a function we are interested in
		}

		if len(call.Args) < argidx+1 {
			return // not enough arguments, e.g. called with return values of another function
		}

		t := pass.TypesInfo.Types[call.Args[argidx]].Type
		switch t.Underlying().(type) {
		case *types.Pointer, *types.Interface:
			return
		}

		switch argidx {
		case 0:
			pass.Reportf(call.Lparen, "call of %s passes non-pointer", fn.Name())
		case 1:
			pass.Reportf(call.Lparen, "call of %s passes non-pointer as second argument", fn.Name())
		}
	})
	return nil, nil
}
