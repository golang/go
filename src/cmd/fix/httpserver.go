// Copyright 2011 The Go Authors.  All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

package main

import (
	"go/ast"
	"go/token"
)

func init() {
	register(httpserverFix)
}

var httpserverFix = fix{
	"httpserver",
	"2011-03-15",
	httpserver,
	`Adapt http server methods and functions to changes
made to the http ResponseWriter interface.

http://codereview.appspot.com/4245064  Hijacker
http://codereview.appspot.com/4239076  Header
http://codereview.appspot.com/4239077  Flusher
http://codereview.appspot.com/4248075  RemoteAddr, UsingTLS
`,
}

func httpserver(f *ast.File) bool {
	if !imports(f, "http") {
		return false
	}

	fixed := false
	for _, decl := range f.Decls {
		fn, ok := decl.(*ast.FuncDecl)
		if !ok {
			continue
		}
		w, req, ok := isServeHTTP(fn)
		if !ok {
			continue
		}
		walk(fn.Body, func(n interface{}) {
			// Want to replace expression sometimes,
			// so record pointer to it for updating below.
			ptr, ok := n.(*ast.Expr)
			if ok {
				n = *ptr
			}

			// Look for w.UsingTLS() and w.Remoteaddr().
			call, ok := n.(*ast.CallExpr)
			if !ok || (len(call.Args) != 0 && len(call.Args) != 2) {
				return
			}
			sel, ok := call.Fun.(*ast.SelectorExpr)
			if !ok {
				return
			}
			if !refersTo(sel.X, w) {
				return
			}
			switch sel.Sel.String() {
			case "Hijack":
				// replace w with w.(http.Hijacker)
				sel.X = &ast.TypeAssertExpr{
					X:    sel.X,
					Type: ast.NewIdent("http.Hijacker"),
				}
				fixed = true
			case "Flush":
				// replace w with w.(http.Flusher)
				sel.X = &ast.TypeAssertExpr{
					X:    sel.X,
					Type: ast.NewIdent("http.Flusher"),
				}
				fixed = true
			case "UsingTLS":
				if ptr == nil {
					// can only replace expression if we have pointer to it
					break
				}
				// replace with req.TLS != nil
				*ptr = &ast.BinaryExpr{
					X: &ast.SelectorExpr{
						X:   ast.NewIdent(req.String()),
						Sel: ast.NewIdent("TLS"),
					},
					Op: token.NEQ,
					Y:  ast.NewIdent("nil"),
				}
				fixed = true
			case "RemoteAddr":
				if ptr == nil {
					// can only replace expression if we have pointer to it
					break
				}
				// replace with req.RemoteAddr
				*ptr = &ast.SelectorExpr{
					X:   ast.NewIdent(req.String()),
					Sel: ast.NewIdent("RemoteAddr"),
				}
				fixed = true
			case "SetHeader":
				// replace w.SetHeader with w.Header().Set
				// or w.Header().Del if second argument is ""
				sel.X = &ast.CallExpr{
					Fun: &ast.SelectorExpr{
						X:   ast.NewIdent(w.String()),
						Sel: ast.NewIdent("Header"),
					},
				}
				sel.Sel = ast.NewIdent("Set")
				if len(call.Args) == 2 && isEmptyString(call.Args[1]) {
					sel.Sel = ast.NewIdent("Del")
					call.Args = call.Args[:1]
				}
				fixed = true
			}
		})
	}
	return fixed
}

func isServeHTTP(fn *ast.FuncDecl) (w, req *ast.Ident, ok bool) {
	for _, field := range fn.Type.Params.List {
		if isPkgDot(field.Type, "http", "ResponseWriter") {
			w = field.Names[0]
			continue
		}
		if isPtrPkgDot(field.Type, "http", "Request") {
			req = field.Names[0]
			continue
		}
	}

	ok = w != nil && req != nil
	return
}
