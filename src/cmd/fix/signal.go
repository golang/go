// Copyright 2011 The Go Authors.  All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

package main

import (
	"go/ast"
	"strings"
)

func init() {
	register(signalFix)
}

var signalFix = fix{
	"signal",
	"2011-06-29",
	signal,
	`Adapt code to types moved from os/signal to signal.

http://codereview.appspot.com/4437091
`,
}

func signal(f *ast.File) (fixed bool) {
	if !imports(f, "os/signal") {
		return
	}

	walk(f, func(n interface{}) {
		s, ok := n.(*ast.SelectorExpr)

		if !ok || !isTopName(s.X, "signal") {
			return
		}

		sel := s.Sel.String()
		if sel == "Signal" || sel == "UnixSignal" || strings.HasPrefix(sel, "SIG") {
			addImport(f, "os")
			s.X = &ast.Ident{Name: "os"}
			fixed = true
		}
	})

	if fixed && !usesImport(f, "os/signal") {
		deleteImport(f, "os/signal")
	}
	return
}
