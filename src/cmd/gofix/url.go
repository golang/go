// Copyright 2011 The Go Authors.  All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

package main

import (
	"fmt"
	"os"
	"go/ast"
)

var _ fmt.Stringer
var _ os.Error

var urlFix = fix{
	"url",
	url,
	`Move the URL pieces of package http into a new package, url.

http://codereview.appspot.com/4893043
`,
}

func init() {
	register(urlFix)
}

var urlRenames = []struct{ in, out string }{
	{"URL", "URL"},
	{"ParseURL", "Parse"},
	{"ParseURLReference", "ParseWithReference"},
	{"ParseQuery", "ParseQuery"},
	{"Values", "Values"},
	{"URLEscape", "QueryEscape"},
	{"URLUnescape", "QueryUnescape"},
	{"URLError", "Error"},
	{"URLEscapeError", "EscapeError"},
}

func url(f *ast.File) bool {
	if imports(f, "url") || !imports(f, "http") {
		return false
	}

	fixed := false

	// Update URL code.
	var skip interface{}
	urlWalk := func(n interface{}) {
		if n == skip {
			skip = nil
			return
		}
		// Is it an identifier?
		if ident, ok := n.(*ast.Ident); ok && ident.Name == "url" {
			ident.Name = "url_"
			return
		}
		// Parameter and result names.
		if fn, ok := n.(*ast.FuncType); ok {
			fixed = urlDoFields(fn.Params) || fixed
			fixed = urlDoFields(fn.Results) || fixed
		}
		// U{url: ...} is likely a struct field.
		if kv, ok := n.(*ast.KeyValueExpr); ok {
			if ident, ok := kv.Key.(*ast.Ident); ok && ident.Name == "url" {
				skip = ident
			}
		}
	}

	// Fix up URL code and add import, at most once.
	fix := func() {
		if fixed {
			return
		}
		walkBeforeAfter(f, urlWalk, nop)
		addImport(f, "url")
		fixed = true
	}

	walk(f, func(n interface{}) {
		// Rename functions and methods.
		if expr, ok := n.(ast.Expr); ok {
			for _, s := range urlRenames {
				if isPkgDot(expr, "http", s.in) {
					fix()
					expr.(*ast.SelectorExpr).X.(*ast.Ident).Name = "url"
					expr.(*ast.SelectorExpr).Sel.Name = s.out
					return
				}
			}
		}
	})

	// Remove the http import if no longer needed.
	if fixed && !usesImport(f, "http") {
		deleteImport(f, "http")
	}

	return fixed
}

func urlDoFields(list *ast.FieldList) (fixed bool) {
	if list == nil {
		return
	}
	for _, field := range list.List {
		for _, ident := range field.Names {
			if ident.Name == "url" {
				fixed = true
				ident.Name = "url_"
			}
		}
	}
	return
}
