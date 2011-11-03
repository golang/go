// Copyright 2011 The Go Authors.  All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

package main

import (
	"go/ast"
)

func init() {
	register(httpFinalURLFix)
}

var httpFinalURLFix = fix{
	"httpfinalurl",
	"2011-05-13",
	httpfinalurl,
	`Adapt http Get calls to not have a finalURL result parameter.

http://codereview.appspot.com/4535056/
`,
}

func httpfinalurl(f *ast.File) bool {
	if !imports(f, "http") {
		return false
	}

	fixed := false
	walk(f, func(n interface{}) {
		// Fix up calls to http.Get.
		//
		// If they have blank identifiers, remove them:
		//    resp, _, err := http.Get(url)
		// -> resp, err := http.Get(url)
		//
		// But if they're using the finalURL parameter, warn:
		//    resp, finalURL, err := http.Get(url)
		as, ok := n.(*ast.AssignStmt)
		if !ok || len(as.Lhs) != 3 || len(as.Rhs) != 1 {
			return
		}

		if !isCall(as.Rhs[0], "http", "Get") {
			return
		}

		if isBlank(as.Lhs[1]) {
			as.Lhs = []ast.Expr{as.Lhs[0], as.Lhs[2]}
			fixed = true
		} else {
			warn(as.Pos(), "call to http.Get records final URL")
		}
	})
	return fixed
}
