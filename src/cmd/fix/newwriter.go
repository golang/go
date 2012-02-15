// Copyright 2012 The Go Authors.  All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

package main

import (
	"go/ast"
)

func init() {
	register(newWriterFix)
}

var newWriterFix = fix{
	"newWriter",
	"2012-02-14",
	newWriter,
	`Adapt bufio, gzip and zlib NewWriterXxx calls for whether they return errors.

Also rename gzip.Compressor and gzip.Decompressor to gzip.Writer and gzip.Reader.

http://codereview.appspot.com/5639057 and
http://codereview.appspot.com/5642054
`,
}

func newWriter(f *ast.File) bool {
	if !imports(f, "bufio") && !imports(f, "compress/gzip") && !imports(f, "compress/zlib") {
		return false
	}

	fixed := false
	walk(f, func(n interface{}) {
		switch n := n.(type) {
		case *ast.SelectorExpr:
			if isTopName(n.X, "gzip") {
				switch n.Sel.String() {
				case "Compressor":
					n.Sel = &ast.Ident{Name: "Writer"}
					fixed = true
				case "Decompressor":
					n.Sel = &ast.Ident{Name: "Reader"}
					fixed = true
				}
			} else if isTopName(n.X, "zlib") {
				if n.Sel.String() == "NewWriterDict" {
					n.Sel = &ast.Ident{Name: "NewWriterLevelDict"}
					fixed = true
				}
			}

		case *ast.AssignStmt:
			// Drop the ", _" in assignments of the form:
			//	w0, _ = gzip.NewWriter(w1)
			if len(n.Lhs) != 2 || len(n.Rhs) != 1 {
				return
			}
			i, ok := n.Lhs[1].(*ast.Ident)
			if !ok {
				return
			}
			if i.String() != "_" {
				return
			}
			c, ok := n.Rhs[0].(*ast.CallExpr)
			if !ok {
				return
			}
			s, ok := c.Fun.(*ast.SelectorExpr)
			if !ok {
				return
			}
			sel := s.Sel.String()
			switch {
			case isTopName(s.X, "bufio") && (sel == "NewReaderSize" || sel == "NewWriterSize"):
				// No-op.
			case isTopName(s.X, "gzip") && sel == "NewWriter":
				// No-op.
			case isTopName(s.X, "zlib") && sel == "NewWriter":
				// No-op.
			default:
				return
			}
			n.Lhs = n.Lhs[:1]
			fixed = true
		}
	})
	return fixed
}
