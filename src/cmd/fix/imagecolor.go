// Copyright 2011 The Go Authors.  All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

package main

import (
	"go/ast"
)

func init() {
	register(imagecolorFix)
}

var imagecolorFix = fix{
	"imagecolor",
	"2011-10-04",
	imagecolor,
	`Adapt code to types moved from image to color.

http://codereview.appspot.com/5132048
`,
}

var colorRenames = []struct{ in, out string }{
	{"Color", "Color"},
	{"ColorModel", "Model"},
	{"ColorModelFunc", "ModelFunc"},
	{"PalettedColorModel", "Palette"},

	{"RGBAColor", "RGBA"},
	{"RGBA64Color", "RGBA64"},
	{"NRGBAColor", "NRGBA"},
	{"NRGBA64Color", "NRGBA64"},
	{"AlphaColor", "Alpha"},
	{"Alpha16Color", "Alpha16"},
	{"GrayColor", "Gray"},
	{"Gray16Color", "Gray16"},

	{"RGBAColorModel", "RGBAModel"},
	{"RGBA64ColorModel", "RGBA64Model"},
	{"NRGBAColorModel", "NRGBAModel"},
	{"NRGBA64ColorModel", "NRGBA64Model"},
	{"AlphaColorModel", "AlphaModel"},
	{"Alpha16ColorModel", "Alpha16Model"},
	{"GrayColorModel", "GrayModel"},
	{"Gray16ColorModel", "Gray16Model"},
}

func imagecolor(f *ast.File) (fixed bool) {
	if !imports(f, "image") {
		return
	}

	walk(f, func(n interface{}) {
		s, ok := n.(*ast.SelectorExpr)

		if !ok || !isTopName(s.X, "image") {
			return
		}

		switch sel := s.Sel.String(); {
		case sel == "ColorImage":
			s.Sel = &ast.Ident{Name: "Uniform"}
			fixed = true
		case sel == "NewColorImage":
			s.Sel = &ast.Ident{Name: "NewUniform"}
			fixed = true
		default:
			for _, rename := range colorRenames {
				if sel == rename.in {
					addImport(f, "image/color")
					s.X.(*ast.Ident).Name = "color"
					s.Sel.Name = rename.out
					fixed = true
				}
			}
		}
	})

	if fixed && !usesImport(f, "image") {
		deleteImport(f, "image")
	}
	return
}
