// Copyright 2011 The Go Authors.  All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

package main

import "go/ast"

func init() {
	register(httputilFix)
}

var httputilFix = fix{
	"httputil",
	"2011-11-18",
	httputil,
	`Move some functions in http package into httputil package.

http://codereview.appspot.com/5336049
`,
}

var httputilFuncs = []string{
	"DumpRequest",
	"DumpRequestOut",
	"DumpResponse",
	"NewChunkedReader",
	"NewChunkedWriter",
	"NewClientConn",
	"NewProxyClientConn",
	"NewServerConn",
	"NewSingleHostReverseProxy",
}

func httputil(f *ast.File) bool {
	if imports(f, "net/http/httputil") {
		return false
	}

	fixed := false

	walk(f, func(n interface{}) {
		// Rename package name.
		if expr, ok := n.(ast.Expr); ok {
			for _, s := range httputilFuncs {
				if isPkgDot(expr, "http", s) {
					if !fixed {
						addImport(f, "net/http/httputil")
						fixed = true
					}
					expr.(*ast.SelectorExpr).X.(*ast.Ident).Name = "httputil"
				}
			}
		}
	})

	// Remove the net/http import if no longer needed.
	if fixed && !usesImport(f, "net/http") {
		deleteImport(f, "net/http")
	}

	return fixed
}
