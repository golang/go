// Copyright 2012 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

package main

import (
	"go/ast"
	"slices"
)

func init() {
	register(netipv6zoneFix)
}

var netipv6zoneFix = fix{
	name: "netipv6zone",
	date: "2012-11-26",
	f:    netipv6zone,
	desc: `Adapt element key to IPAddr, UDPAddr or TCPAddr composite literals.

https://codereview.appspot.com/6849045/
`,
}

func netipv6zone(f *ast.File) bool {
	if !imports(f, "net") {
		return false
	}

	fixed := false
	walk(f, func(n any) {
		cl, ok := n.(*ast.CompositeLit)
		if !ok {
			return
		}
		se, ok := cl.Type.(*ast.SelectorExpr)
		if !ok {
			return
		}
		if !isTopName(se.X, "net") || se.Sel == nil {
			return
		}
		switch ss := se.Sel.String(); ss {
		case "IPAddr", "UDPAddr", "TCPAddr":
			for i, e := range cl.Elts {
				if _, ok := e.(*ast.KeyValueExpr); ok {
					break
				}
				switch i {
				case 0:
					cl.Elts[i] = &ast.KeyValueExpr{
						Key:   ast.NewIdent("IP"),
						Value: e,
					}
				case 1:
					if elit, ok := e.(*ast.BasicLit); ok && elit.Value == "0" {
						cl.Elts = slices.Delete(cl.Elts, i, i+1)
					} else {
						cl.Elts[i] = &ast.KeyValueExpr{
							Key:   ast.NewIdent("Port"),
							Value: e,
						}
					}
				}
				fixed = true
			}
		}
	})
	return fixed
}
