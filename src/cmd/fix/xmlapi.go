// Copyright 2012 The Go Authors.  All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

package main

import (
	"go/ast"
)

func init() {
	register(xmlapiFix)
}

var xmlapiFix = fix{
	"xmlapi",
	"2012-01-23",
	xmlapi,
	`
	Make encoding/xml's API look more like the rest of the encoding packages.

http://codereview.appspot.com/5574053
`,
}

var xmlapiTypeConfig = &TypeConfig{
	Func: map[string]string{
		"xml.NewParser":         "*xml.Parser",
		"os.Open":               "*os.File",
		"os.OpenFile":           "*os.File",
		"bytes.NewBuffer":       "*bytes.Buffer",
		"bytes.NewBufferString": "*bytes.Buffer",
		"bufio.NewReader":       "*bufio.Reader",
		"bufio.NewReadWriter":   "*bufio.ReadWriter",
	},
}

var isReader = map[string]bool{
	"*os.File":          true,
	"*bytes.Buffer":     true,
	"*bufio.Reader":     true,
	"*bufio.ReadWriter": true,
	"io.Reader":         true,
}

func xmlapi(f *ast.File) bool {
	if !imports(f, "encoding/xml") {
		return false
	}

	typeof, _ := typecheck(xmlapiTypeConfig, f)

	fixed := false
	walk(f, func(n interface{}) {
		s, ok := n.(*ast.SelectorExpr)
		if ok && typeof[s.X] == "*xml.Parser" && s.Sel.Name == "Unmarshal" {
			s.Sel.Name = "DecodeElement"
			fixed = true
			return
		}
		if ok && isPkgDot(s, "xml", "Parser") {
			s.Sel.Name = "Decoder"
			fixed = true
			return
		}

		call, ok := n.(*ast.CallExpr)
		if !ok {
			return
		}
		switch {
		case len(call.Args) == 2 && isPkgDot(call.Fun, "xml", "Marshal"):
			*call = xmlMarshal(call.Args)
			fixed = true
		case len(call.Args) == 2 && isPkgDot(call.Fun, "xml", "Unmarshal"):
			if isReader[typeof[call.Args[0]]] {
				*call = xmlUnmarshal(call.Args)
				fixed = true
			}
		case len(call.Args) == 1 && isPkgDot(call.Fun, "xml", "NewParser"):
			sel := call.Fun.(*ast.SelectorExpr).Sel
			sel.Name = "NewDecoder"
			fixed = true
		}
	})
	return fixed
}

func xmlMarshal(args []ast.Expr) ast.CallExpr {
	return xmlCallChain("NewEncoder", "Encode", args)
}

func xmlUnmarshal(args []ast.Expr) ast.CallExpr {
	return xmlCallChain("NewDecoder", "Decode", args)
}

func xmlCallChain(first, second string, args []ast.Expr) ast.CallExpr {
	return ast.CallExpr{
		Fun: &ast.SelectorExpr{
			X: &ast.CallExpr{
				Fun: &ast.SelectorExpr{
					X:   ast.NewIdent("xml"),
					Sel: ast.NewIdent(first),
				},
				Args: args[:1],
			},
			Sel: ast.NewIdent(second),
		},
		Args: args[1:2],
	}
}
