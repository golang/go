// Copyright 2011 The Go Authors.  All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

package main

import "go/ast"

func init() {
	register(strconvFix)
}

var strconvFix = fix{
	"strconv",
	"2011-12-01",
	strconvFn,
	`Convert to new strconv API.

http://codereview.appspot.com/5434095
http://codereview.appspot.com/5434069
`,
}

func strconvFn(f *ast.File) bool {
	if !imports(f, "strconv") {
		return false
	}

	fixed := false

	walk(f, func(n interface{}) {
		// Rename functions.
		call, ok := n.(*ast.CallExpr)
		if !ok || len(call.Args) < 1 {
			return
		}
		sel, ok := call.Fun.(*ast.SelectorExpr)
		if !ok || !isTopName(sel.X, "strconv") {
			return
		}
		change := func(name string) {
			fixed = true
			sel.Sel.Name = name
		}
		add := func(s string) {
			call.Args = append(call.Args, expr(s))
		}
		switch sel.Sel.Name {
		case "Atob":
			change("ParseBool")
		case "Atof32":
			change("ParseFloat")
			add("32") // bitSize
			warn(call.Pos(), "rewrote strconv.Atof32(_) to strconv.ParseFloat(_, 32) but return value must be converted to float32")
		case "Atof64":
			change("ParseFloat")
			add("64") // bitSize
		case "AtofN":
			change("ParseFloat")
		case "Atoi":
			// Atoi stayed as a convenience wrapper.
		case "Atoi64":
			change("ParseInt")
			add("10") // base
			add("64") // bitSize
		case "Atoui":
			change("ParseUint")
			add("10") // base
			add("0")  // bitSize
			warn(call.Pos(), "rewrote strconv.Atoui(_) to strconv.ParseUint(_, 10, 0) but return value must be converted to uint")
		case "Atoui64":
			change("ParseUint")
			add("10") // base
			add("64") // bitSize
		case "Btoa":
			change("FormatBool")
		case "Btoi64":
			change("ParseInt")
			add("64") // bitSize
		case "Btoui64":
			change("ParseUint")
			add("64") // bitSize
		case "Ftoa32":
			change("FormatFloat")
			call.Args[0] = strconvRewrite("float32", "float64", call.Args[0])
			add("32") // bitSize
		case "Ftoa64":
			change("FormatFloat")
			add("64") // bitSize
		case "FtoaN":
			change("FormatFloat")
		case "Itoa":
			// Itoa stayed as a convenience wrapper.
		case "Itoa64":
			change("FormatInt")
			add("10") // base
		case "Itob":
			change("FormatInt")
			call.Args[0] = strconvRewrite("int", "int64", call.Args[0])
		case "Itob64":
			change("FormatInt")
		case "Uitoa":
			change("FormatUint")
			call.Args[0] = strconvRewrite("uint", "uint64", call.Args[0])
			add("10") // base
		case "Uitoa64":
			change("FormatUint")
			add("10") // base
		case "Uitob":
			change("FormatUint")
			call.Args[0] = strconvRewrite("uint", "uint64", call.Args[0])
		case "Uitob64":
			change("FormatUint")
		}
	})
	return fixed
}

// rewrite from type t1 to type t2
// If the expression x is of the form t1(_), use t2(_).  Otherwise use t2(x).
func strconvRewrite(t1, t2 string, x ast.Expr) ast.Expr {
	if call, ok := x.(*ast.CallExpr); ok && isTopName(call.Fun, t1) {
		call.Fun.(*ast.Ident).Name = t2
		return x
	}
	return &ast.CallExpr{Fun: ast.NewIdent(t2), Args: []ast.Expr{x}}
}
