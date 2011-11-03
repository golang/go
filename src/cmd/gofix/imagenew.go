// Copyright 2011 The Go Authors.  All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

package main

import (
	"go/ast"
)

func init() {
	register(imagenewFix)
}

var imagenewFix = fix{
	"imagenew",
	"2011-09-14",
	imagenew,
	`Adapt image.NewXxx calls to pass an image.Rectangle instead of (w, h int).

http://codereview.appspot.com/4964073
`,
}

var imagenewFuncs = map[string]bool{
	"NewRGBA":    true,
	"NewRGBA64":  true,
	"NewNRGBA":   true,
	"NewNRGBA64": true,
	"NewAlpha":   true,
	"NewAlpha16": true,
	"NewGray":    true,
	"NewGray16":  true,
}

func imagenew(f *ast.File) bool {
	if !imports(f, "image") {
		return false
	}

	fixed := false
	walk(f, func(n interface{}) {
		call, ok := n.(*ast.CallExpr)
		if !ok {
			return
		}
		isNewFunc := false
		for newFunc := range imagenewFuncs {
			if len(call.Args) == 2 && isPkgDot(call.Fun, "image", newFunc) {
				isNewFunc = true
				break
			}
		}
		if len(call.Args) == 3 && isPkgDot(call.Fun, "image", "NewPaletted") {
			isNewFunc = true
		}
		if !isNewFunc {
			return
		}
		// Replace image.NewXxx(w, h) with image.NewXxx(image.Rect(0, 0, w, h)).
		rectArgs := []ast.Expr{
			&ast.BasicLit{Value: "0"},
			&ast.BasicLit{Value: "0"},
		}
		rectArgs = append(rectArgs, call.Args[:2]...)
		rect := []ast.Expr{
			&ast.CallExpr{
				Fun: &ast.SelectorExpr{
					X: &ast.Ident{
						Name: "image",
					},
					Sel: &ast.Ident{
						Name: "Rect",
					},
				},
				Args: rectArgs,
			},
		}
		call.Args = append(rect, call.Args[2:]...)
		fixed = true
	})
	return fixed
}
