// Copyright 2011 The Go Authors.  All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

package main

import (
	"go/ast"
)

func init() {
	register(imageycbcrFix)
}

var imageycbcrFix = fix{
	"imageycbcr",
	"2011-12-20",
	imageycbcr,
	`Adapt code to types moved from image/ycbcr to image and image/color.

http://codereview.appspot.com/5493084
`,
}

func imageycbcr(f *ast.File) (fixed bool) {
	if !imports(f, "image/ycbcr") {
		return
	}

	walk(f, func(n interface{}) {
		s, ok := n.(*ast.SelectorExpr)

		if !ok || !isTopName(s.X, "ycbcr") {
			return
		}

		switch s.Sel.String() {
		case "RGBToYCbCr", "YCbCrToRGB":
			addImport(f, "image/color")
			s.X.(*ast.Ident).Name = "color"
		case "YCbCrColor":
			addImport(f, "image/color")
			s.X.(*ast.Ident).Name = "color"
			s.Sel.Name = "YCbCr"
		case "YCbCrColorModel":
			addImport(f, "image/color")
			s.X.(*ast.Ident).Name = "color"
			s.Sel.Name = "YCbCrModel"
		case "SubsampleRatio", "SubsampleRatio444", "SubsampleRatio422", "SubsampleRatio420":
			addImport(f, "image")
			s.X.(*ast.Ident).Name = "image"
			s.Sel.Name = "YCbCr" + s.Sel.Name
		case "YCbCr":
			addImport(f, "image")
			s.X.(*ast.Ident).Name = "image"
		default:
			return
		}
		fixed = true
	})

	deleteImport(f, "image/ycbcr")
	return
}
