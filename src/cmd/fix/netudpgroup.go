// Copyright 2011 The Go Authors.  All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

package main

import (
	"go/ast"
)

func init() {
	register(netudpgroupFix)
}

var netudpgroupFix = fix{
	"netudpgroup",
	"2011-08-18",
	netudpgroup,
	`Adapt 1-argument calls of net.(*UDPConn).JoinGroup, LeaveGroup to use 2-argument form.

http://codereview.appspot.com/4815074
`,
}

func netudpgroup(f *ast.File) bool {
	if !imports(f, "net") {
		return false
	}

	fixed := false
	for _, d := range f.Decls {
		fd, ok := d.(*ast.FuncDecl)
		if !ok || fd.Body == nil {
			continue
		}
		walk(fd.Body, func(n interface{}) {
			ce, ok := n.(*ast.CallExpr)
			if !ok {
				return
			}
			se, ok := ce.Fun.(*ast.SelectorExpr)
			if !ok || len(ce.Args) != 1 {
				return
			}
			switch se.Sel.String() {
			case "JoinGroup", "LeaveGroup":
				// c.JoinGroup(a) -> c.JoinGroup(nil, a)
				// c.LeaveGroup(a) -> c.LeaveGroup(nil, a)
				arg := ce.Args[0]
				ce.Args = make([]ast.Expr, 2)
				ce.Args[0] = ast.NewIdent("nil")
				ce.Args[1] = arg
				fixed = true
			}
		})
	}
	return fixed
}
