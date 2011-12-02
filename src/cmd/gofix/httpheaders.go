// Copyright 2011 The Go Authors.  All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

package main

import (
	"go/ast"
)

func init() {
	register(httpHeadersFix)
}

var httpHeadersFix = fix{
	"httpheaders",
	"2011-06-16",
	httpheaders,
	`Rename http Referer, UserAgent, Cookie, SetCookie, which are now methods.

http://codereview.appspot.com/4620049/
`,
}

func httpheaders(f *ast.File) bool {
	if !imports(f, "http") {
		return false
	}

	called := make(map[ast.Node]bool)
	walk(f, func(ni interface{}) {
		switch n := ni.(type) {
		case *ast.CallExpr:
			called[n.Fun] = true
		}
	})

	fixed := false
	typeof, _ := typecheck(headerTypeConfig, f)
	walk(f, func(ni interface{}) {
		switch n := ni.(type) {
		case *ast.SelectorExpr:
			if called[n] {
				break
			}
			if t := typeof[n.X]; t != "*http.Request" && t != "*http.Response" {
				break
			}
			switch n.Sel.Name {
			case "Referer", "UserAgent":
				n.Sel.Name += "()"
				fixed = true
			case "Cookie":
				n.Sel.Name = "Cookies()"
				fixed = true
			}
		}
	})
	return fixed
}

var headerTypeConfig = &TypeConfig{
	Type: map[string]*Type{
		"*http.Request":  {},
		"*http.Response": {},
	},
}
