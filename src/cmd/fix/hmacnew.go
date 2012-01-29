// Copyright 2011 The Go Authors.  All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

package main

import "go/ast"

func init() {
	register(hmacNewFix)
}

var hmacNewFix = fix{
	"hmacnew",
	"2012-01-19",
	hmacnew,
	`Deprecate hmac.NewMD5, hmac.NewSHA1 and hmac.NewSHA256.

This fix rewrites code using hmac.NewMD5, hmac.NewSHA1 and hmac.NewSHA256 to
use hmac.New:

	hmac.NewMD5(key) -> hmac.New(md5.New, key)
	hmac.NewSHA1(key) -> hmac.New(sha1.New, key)
	hmac.NewSHA256(key) -> hmac.New(sha256.New, key)

`,
}

func hmacnew(f *ast.File) (fixed bool) {
	if !imports(f, "crypto/hmac") {
		return
	}

	walk(f, func(n interface{}) {
		ce, ok := n.(*ast.CallExpr)
		if !ok {
			return
		}

		var pkg string
		switch {
		case isPkgDot(ce.Fun, "hmac", "NewMD5"):
			pkg = "md5"
		case isPkgDot(ce.Fun, "hmac", "NewSHA1"):
			pkg = "sha1"
		case isPkgDot(ce.Fun, "hmac", "NewSHA256"):
			pkg = "sha256"
		default:
			return
		}

		addImport(f, "crypto/"+pkg)

		ce.Fun = ast.NewIdent("hmac.New")
		ce.Args = append([]ast.Expr{ast.NewIdent(pkg + ".New")}, ce.Args...)

		fixed = true
	})

	return
}
