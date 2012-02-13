// Copyright 2011 The Go Authors.  All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

package main

import (
	"go/ast"
	"strings"
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

	// go.crypto sub-repository
	{"crypto/bcrypt", "code.google.com/p/go.crypto/bcrypt"},
	{"crypto/blowfish", "code.google.com/p/go.crypto/blowfish"},
	{"crypto/cast5", "code.google.com/p/go.crypto/cast5"},
	{"crypto/md4", "code.google.com/p/go.crypto/md4"},
	{"crypto/ocsp", "code.google.com/p/go.crypto/ocsp"},
	{"crypto/openpgp", "code.google.com/p/go.crypto/openpgp"},
	{"crypto/openpgp/armor", "code.google.com/p/go.crypto/openpgp/armor"},
	{"crypto/openpgp/elgamal", "code.google.com/p/go.crypto/openpgp/elgamal"},
	{"crypto/openpgp/errors", "code.google.com/p/go.crypto/openpgp/errors"},
	{"crypto/openpgp/packet", "code.google.com/p/go.crypto/openpgp/packet"},
	{"crypto/openpgp/s2k", "code.google.com/p/go.crypto/openpgp/s2k"},
	{"crypto/ripemd160", "code.google.com/p/go.crypto/ripemd160"},
	{"crypto/twofish", "code.google.com/p/go.crypto/twofish"},
	{"crypto/xtea", "code.google.com/p/go.crypto/xtea"},
	{"exp/ssh", "code.google.com/p/go.crypto/ssh"},

	// go.image sub-repository
	{"image/bmp", "code.google.com/p/go.image/bmp"},
	{"image/tiff", "code.google.com/p/go.image/tiff"},

	// go.net sub-repository
	{"net/dict", "code.google.com/p/go.net/dict"},
	{"net/websocket", "code.google.com/p/go.net/websocket"},
	{"exp/spdy", "code.google.com/p/go.net/spdy"},
	{"http/spdy", "code.google.com/p/go.net/spdy"},

	// go.codereview sub-repository
	{"encoding/git85", "code.google.com/p/go.codereview/git85"},
	{"patch", "code.google.com/p/go.codereview/patch"},

	// exp
	{"ebnf", "exp/ebnf"},
	{"go/types", "exp/types"},

	// deleted
	{"container/vector", ""},
	{"exp/datafmt", ""},
	{"go/typechecker", ""},
	{"old/netchan", ""},
	{"old/regexp", ""},
	{"old/template", ""},
	{"try", ""},
}

var go1PackageNameRenames = []struct{ newPath, old, new string }{
	{"html/template", "html", "template"},
	{"math/cmplx", "cmath", "cmplx"},
}

func go1pkgrename(f *ast.File) bool {
	fixed := false

	// First update the imports.
	for _, rename := range go1PackageRenames {
		spec := importSpec(f, rename.old)
		if spec == nil {
			continue
		}
		if rename.new == "" {
			warn(spec.Pos(), "package %q has been deleted in Go 1", rename.old)
			continue
		}
		if rewriteImport(f, rename.old, rename.new) {
			fixed = true
		}
		if strings.HasPrefix(rename.new, "exp/") {
			warn(spec.Pos(), "package %q is not part of Go 1", rename.new)
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
