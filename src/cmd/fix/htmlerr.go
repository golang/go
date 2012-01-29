// Copyright 2011 The Go Authors.  All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

package main

import (
	"go/ast"
)

func init() {
	register(htmlerrFix)
}

var htmlerrFix = fix{
	"htmlerr",
	"2011-11-04",
	htmlerr,
	`Rename html's Tokenizer.Error method to Err.

http://codereview.appspot.com/5327064/
`,
}

var htmlerrTypeConfig = &TypeConfig{
	Func: map[string]string{
		"html.NewTokenizer": "html.Tokenizer",
	},
}

func htmlerr(f *ast.File) bool {
	if !imports(f, "html") {
		return false
	}

	typeof, _ := typecheck(htmlerrTypeConfig, f)

	fixed := false
	walk(f, func(n interface{}) {
		s, ok := n.(*ast.SelectorExpr)
		if ok && typeof[s.X] == "html.Tokenizer" && s.Sel.Name == "Error" {
			s.Sel.Name = "Err"
			fixed = true
		}
	})
	return fixed
}
