// Copyright 2011 The Go Authors.  All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

package main

import (
	"go/ast"
)

func init() {
	register(netdialFix)
	register(tlsdialFix)
	register(netlookupFix)
}

var netdialFix = fix{
	"netdial",
	"2011-03-28",
	netdial,
	`Adapt 3-argument calls of net.Dial to use 2-argument form.

http://codereview.appspot.com/4244055
`,
}

var tlsdialFix = fix{
	"tlsdial",
	"2011-03-28",
	tlsdial,
	`Adapt 4-argument calls of tls.Dial to use 3-argument form.

http://codereview.appspot.com/4244055
`,
}

var netlookupFix = fix{
	"netlookup",
	"2011-03-28",
	netlookup,
	`Adapt 3-result calls to net.LookupHost to use 2-result form.

http://codereview.appspot.com/4244055
`,
}

func netdial(f *ast.File) bool {
	if !imports(f, "net") {
		return false
	}

	fixed := false
	walk(f, func(n interface{}) {
		call, ok := n.(*ast.CallExpr)
		if !ok || !isPkgDot(call.Fun, "net", "Dial") || len(call.Args) != 3 {
			return
		}
		// net.Dial(a, "", b) -> net.Dial(a, b)
		if !isEmptyString(call.Args[1]) {
			warn(call.Pos(), "call to net.Dial with non-empty second argument")
			return
		}
		call.Args[1] = call.Args[2]
		call.Args = call.Args[:2]
		fixed = true
	})
	return fixed
}

func tlsdial(f *ast.File) bool {
	if !imports(f, "crypto/tls") {
		return false
	}

	fixed := false
	walk(f, func(n interface{}) {
		call, ok := n.(*ast.CallExpr)
		if !ok || !isPkgDot(call.Fun, "tls", "Dial") || len(call.Args) != 4 {
			return
		}
		// tls.Dial(a, "", b, c) -> tls.Dial(a, b, c)
		if !isEmptyString(call.Args[1]) {
			warn(call.Pos(), "call to tls.Dial with non-empty second argument")
			return
		}
		call.Args[1] = call.Args[2]
		call.Args[2] = call.Args[3]
		call.Args = call.Args[:3]
		fixed = true
	})
	return fixed
}

func netlookup(f *ast.File) bool {
	if !imports(f, "net") {
		return false
	}

	fixed := false
	walk(f, func(n interface{}) {
		as, ok := n.(*ast.AssignStmt)
		if !ok || len(as.Lhs) != 3 || len(as.Rhs) != 1 {
			return
		}
		call, ok := as.Rhs[0].(*ast.CallExpr)
		if !ok || !isPkgDot(call.Fun, "net", "LookupHost") {
			return
		}
		if !isBlank(as.Lhs[2]) {
			warn(as.Pos(), "call to net.LookupHost expecting cname; use net.LookupCNAME")
			return
		}
		as.Lhs = as.Lhs[:2]
		fixed = true
	})
	return fixed
}
