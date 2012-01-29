// Copyright 2011 The Go Authors.  All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

package main

import (
	"go/ast"
)

func init() {
	register(oserrorstringFix)
}

var oserrorstringFix = fix{
	"oserrorstring",
	"2011-06-22",
	oserrorstring,
	`Replace os.ErrorString() conversions with calls to os.NewError().

http://codereview.appspot.com/4607052
`,
}

func oserrorstring(f *ast.File) bool {
	if !imports(f, "os") {
		return false
	}

	fixed := false
	walk(f, func(n interface{}) {
		// The conversion os.ErrorString(x) looks like a call
		// of os.ErrorString with one argument.
		if call := callExpr(n, "os", "ErrorString"); call != nil {
			// os.ErrorString(args) -> os.NewError(args)
			call.Fun.(*ast.SelectorExpr).Sel.Name = "NewError"
			// os.ErrorString(args) -> os.NewError(args)
			call.Fun.(*ast.SelectorExpr).Sel.Name = "NewError"
			fixed = true
			return
		}

		// Remove os.Error type from variable declarations initialized
		// with an os.NewError.
		// (An *ast.ValueSpec may also be used in a const declaration
		// but those won't be initialized with a call to os.NewError.)
		if spec, ok := n.(*ast.ValueSpec); ok &&
			len(spec.Names) == 1 &&
			isPkgDot(spec.Type, "os", "Error") &&
			len(spec.Values) == 1 &&
			callExpr(spec.Values[0], "os", "NewError") != nil {
			// var name os.Error = os.NewError(x) ->
			// var name          = os.NewError(x)
			spec.Type = nil
			fixed = true
			return
		}

		// Other occurrences of os.ErrorString are not fixed
		// but they are rare.

	})
	return fixed
}

// callExpr returns the call expression if x is a call to pkg.name with one argument;
// otherwise it returns nil.
func callExpr(x interface{}, pkg, name string) *ast.CallExpr {
	if call, ok := x.(*ast.CallExpr); ok &&
		len(call.Args) == 1 &&
		isPkgDot(call.Fun, pkg, name) {
		return call
	}
	return nil
}
