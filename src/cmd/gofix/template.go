// Copyright 2011 The Go Authors.  All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

package main

import (
	"go/ast"
)

func init() {
	register(templateFix)
}

var templateFix = fix{
	"template",
	"2011-11-22",
	template,
	`Rewrite calls to template.ParseFile to template.ParseFiles

http://codereview.appspot.com/5433048
`,
}

var templateSetGlobals = []string{
	"ParseSetFiles",
	"ParseSetGlob",
	"ParseTemplateFiles",
	"ParseTemplateGlob",
	"Set",
	"SetMust",
}

var templateSetMethods = []string{
	"ParseSetFiles",
	"ParseSetGlob",
	"ParseTemplateFiles",
	"ParseTemplateGlob",
}

var templateTypeConfig = &TypeConfig{
	Type: map[string]*Type{
		"template.Template": {
			Method: map[string]string{
				"Funcs":      "func() *template.Template",
				"Delims":     "func() *template.Template",
				"Parse":      "func() (*template.Template, error)",
				"ParseFile":  "func() (*template.Template, error)",
				"ParseInSet": "func() (*template.Template, error)",
			},
		},
		"template.Set": {
			Method: map[string]string{
				"ParseSetFiles":      "func() (*template.Set, error)",
				"ParseSetGlob":       "func() (*template.Set, error)",
				"ParseTemplateFiles": "func() (*template.Set, error)",
				"ParseTemplateGlob":  "func() (*template.Set, error)",
			},
		},
	},

	Func: map[string]string{
		"template.New":     "*template.Template",
		"template.Must":    "(*template.Template, error)",
		"template.SetMust": "(*template.Set, error)",
	},
}

func template(f *ast.File) bool {
	if !imports(f, "text/template") && !imports(f, "html/template") {
		return false
	}

	fixed := false

	typeof, _ := typecheck(templateTypeConfig, f)

	// Now update the names used by importers.
	walk(f, func(n interface{}) {
		if sel, ok := n.(*ast.SelectorExpr); ok {
			// Reference to top-level function ParseFile.
			if isPkgDot(sel, "template", "ParseFile") {
				sel.Sel.Name = "ParseFiles"
				fixed = true
				return
			}
			// Reference to ParseFiles method.
			if typeof[sel.X] == "*template.Template" && sel.Sel.Name == "ParseFile" {
				sel.Sel.Name = "ParseFiles"
				fixed = true
				return
			}
			// The Set type and its functions are now gone.
			for _, name := range templateSetGlobals {
				if isPkgDot(sel, "template", name) {
					warn(sel.Pos(), "reference to template.%s must be fixed manually", name)
					return
				}
			}
			// The methods of Set are now gone.
			for _, name := range templateSetMethods {
				if typeof[sel.X] == "*template.Set" && sel.Sel.Name == name {
					warn(sel.Pos(), "reference to template.*Set.%s must be fixed manually", name)
					return
				}
			}
		}
	})

	return fixed
}
