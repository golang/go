// Copyright 2011 The Go Authors.  All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

package main

import (
	"go/ast"
)

func init() {
	register(go1pkgrenameFix)
}

var go1pkgrenameFix = fix{
	"go1rename",
	"2011-11-08",
	go1pkgrename,
	`Rewrite imports for packages moved during transition to Go 1.

http://codereview.appspot.com/5316078
`,
}

var go1PackageRenames = []struct{ old, new string }{
	{"asn1", "encoding/asn1"},
	{"big", "math/big"},
	{"cmath", "math/cmplx"},
	{"csv", "encoding/csv"},
	{"exec", "os/exec"},
	{"exp/template/html", "html/template"},
	{"gob", "encoding/gob"},
	{"http", "net/http"},
	{"http/cgi", "net/http/cgi"},
	{"http/fcgi", "net/http/fcgi"},
	{"http/httptest", "net/http/httptest"},
	{"http/pprof", "net/http/pprof"},
	{"json", "encoding/json"},
	{"mail", "net/mail"},
	{"rpc", "net/rpc"},
	{"rpc/jsonrpc", "net/rpc/jsonrpc"},
	{"scanner", "text/scanner"},
	{"smtp", "net/smtp"},
	{"syslog", "log/syslog"},
	{"tabwriter", "text/tabwriter"},
	{"template", "text/template"},
	{"template/parse", "text/template/parse"},
	{"rand", "math/rand"},
	{"url", "net/url"},
	{"utf16", "unicode/utf16"},
	{"utf8", "unicode/utf8"},
	{"xml", "encoding/xml"},
}

var go1PackageNameRenames = []struct{ newPath, old, new string }{
	{"html/template", "html", "template"},
	{"math/cmplx", "cmath", "cmplx"},
}

func go1pkgrename(f *ast.File) bool {
	fixed := false

	// First update the imports.
	for _, rename := range go1PackageRenames {
		if !imports(f, rename.old) {
			continue
		}
		if rewriteImport(f, rename.old, rename.new) {
			fixed = true
		}
	}
	if !fixed {
		return false
	}

	// Now update the package names used by importers.
	for _, rename := range go1PackageNameRenames {
		// These are rare packages, so do the import test before walking.
		if imports(f, rename.newPath) {
			walk(f, func(n interface{}) {
				if sel, ok := n.(*ast.SelectorExpr); ok {
					if isTopName(sel.X, rename.old) {
						// We know Sel.X is an Ident.
						sel.X.(*ast.Ident).Name = rename.new
						return
					}
				}
			})
		}
	}

	return fixed
}
