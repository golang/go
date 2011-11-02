// Copyright 2011 The Go Authors.  All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

package main

import (
	"go/ast"
	"regexp"
	"strings"
)

func init() {
	fixes = append(fixes, errorFix)
}

var errorFix = fix{
	"error",
	errorFn,
	`Use error instead of os.Error.

This fix rewrites code using os.Error to use error:

	os.Error -> error
	os.NewError -> errors.New
	os.EOF -> io.EOF

Seeing the old names above (os.Error and so on) triggers the following
heuristic rewrites.  The heuristics can be forced using the -force=error flag.

A top-level function, variable, or constant named error is renamed error_.

Error implementations—those types used as os.Error or named
XxxError—have their String methods renamed to Error.  Any existing
Error field or method is renamed to Err.

Error values—those with type os.Error or named e, err, error, err1,
and so on—have method calls and field references rewritten just
as the types do (String to Error, Error to Err).  Also, a type assertion
of the form err.(*os.Waitmsg) becomes err.(*exec.ExitError).

http://codereview.appspot.com/5305066
`,
}

// At minimum, this fix applies the following rewrites:
//
//	os.Error -> error
//	os.NewError -> errors.New
//	os.EOF -> io.EOF
//
// However, if can apply any of those rewrites, it assumes that the
// file predates the error type and tries to update the code to use
// the new definition for error - an Error method, not a String method.
// This more heuristic procedure may not be 100% accurate, so it is
// only run when the file needs updating anyway.  The heuristic can
// be forced to run using -force=error.
//
// First, we must identify the implementations of os.Error.
// These include the type of any value returned as or assigned to an os.Error.
// To that set we add any type whose name contains "Error" or "error".
// The heuristic helps for implementations that are not used as os.Error
// in the file in which they are defined.
//
// In any implementation of os.Error, we rename an existing struct field
// or method named Error to Err and rename the String method to Error.
//
// Second, we must identify the values of type os.Error.
// These include any value that obviously has type os.Error.
// To that set we add any variable whose name is e or err or error
// possibly followed by _ or a numeric or capitalized suffix.
// The heuristic helps for variables that are initialized using calls
// to functions in other packages.  The type checker does not have
// information about those packages available, and in general cannot
// (because the packages may themselves not compile).
//
// For any value of type os.Error, we replace a call to String with a call to Error.
// We also replace type assertion err.(*os.Waitmsg) with err.(*exec.ExitError).

// Variables matching this regexp are assumed to have type os.Error.
var errVar = regexp.MustCompile(`^(e|err|error)_?([A-Z0-9].*)?$`)

// Types matching this regexp are assumed to be implementations of os.Error.
var errType = regexp.MustCompile(`^\*?([Ee]rror|.*Error)$`)

// Type-checking configuration: tell the type-checker this basic
// information about types, functions, and variables in external packages.
var errorTypeConfig = &TypeConfig{
	Type: map[string]*Type{
		"os.Error": &Type{},
	},
	Func: map[string]string{
		"fmt.Errorf":  "os.Error",
		"os.NewError": "os.Error",
	},
	Var: map[string]string{
		"os.EPERM":        "os.Error",
		"os.ENOENT":       "os.Error",
		"os.ESRCH":        "os.Error",
		"os.EINTR":        "os.Error",
		"os.EIO":          "os.Error",
		"os.ENXIO":        "os.Error",
		"os.E2BIG":        "os.Error",
		"os.ENOEXEC":      "os.Error",
		"os.EBADF":        "os.Error",
		"os.ECHILD":       "os.Error",
		"os.EDEADLK":      "os.Error",
		"os.ENOMEM":       "os.Error",
		"os.EACCES":       "os.Error",
		"os.EFAULT":       "os.Error",
		"os.EBUSY":        "os.Error",
		"os.EEXIST":       "os.Error",
		"os.EXDEV":        "os.Error",
		"os.ENODEV":       "os.Error",
		"os.ENOTDIR":      "os.Error",
		"os.EISDIR":       "os.Error",
		"os.EINVAL":       "os.Error",
		"os.ENFILE":       "os.Error",
		"os.EMFILE":       "os.Error",
		"os.ENOTTY":       "os.Error",
		"os.EFBIG":        "os.Error",
		"os.ENOSPC":       "os.Error",
		"os.ESPIPE":       "os.Error",
		"os.EROFS":        "os.Error",
		"os.EMLINK":       "os.Error",
		"os.EPIPE":        "os.Error",
		"os.EAGAIN":       "os.Error",
		"os.EDOM":         "os.Error",
		"os.ERANGE":       "os.Error",
		"os.EADDRINUSE":   "os.Error",
		"os.ECONNREFUSED": "os.Error",
		"os.ENAMETOOLONG": "os.Error",
		"os.EAFNOSUPPORT": "os.Error",
		"os.ETIMEDOUT":    "os.Error",
		"os.ENOTCONN":     "os.Error",
	},
}

func errorFn(f *ast.File) bool {
	if !imports(f, "os") && !force["error"] {
		return false
	}

	// Fix gets called once to run the heuristics described above
	// when we notice that this file definitely needs fixing
	// (it mentions os.Error or something similar).
	var fixed bool
	var didHeuristic bool
	heuristic := func() {
		if didHeuristic {
			return
		}
		didHeuristic = true

		// We have identified a necessary fix (like os.Error -> error)
		// but have not applied it or any others yet.  Prepare the file
		// for fixing and apply heuristic fixes.

		// Rename error to error_ to make room for error.
		fixed = renameTop(f, "error", "error_") || fixed

		// Use type checker to build list of error implementations.
		typeof, assign := typecheck(errorTypeConfig, f)

		isError := map[string]bool{}
		for _, val := range assign["os.Error"] {
			t := typeof[val]
			if strings.HasPrefix(t, "*") {
				t = t[1:]
			}
			if t != "" && !strings.HasPrefix(t, "func(") {
				isError[t] = true
			}
		}

		// We use both the type check results and the "Error" name heuristic
		// to identify implementations of os.Error.
		isErrorImpl := func(typ string) bool {
			return isError[typ] || errType.MatchString(typ)
		}

		isErrorVar := func(x ast.Expr) bool {
			if typ := typeof[x]; typ != "" {
				return isErrorImpl(typ) || typ == "os.Error"
			}
			if sel, ok := x.(*ast.SelectorExpr); ok {
				return sel.Sel.Name == "Error" || sel.Sel.Name == "Err"
			}
			if id, ok := x.(*ast.Ident); ok {
				return errVar.MatchString(id.Name)
			}
			return false
		}

		walk(f, func(n interface{}) {
			// In method declaration on error implementation type,
			// rename String() to Error() and Error() to Err().
			fn, ok := n.(*ast.FuncDecl)
			if ok &&
				fn.Recv != nil &&
				len(fn.Recv.List) == 1 &&
				isErrorImpl(typeName(fn.Recv.List[0].Type)) {
				// Rename.
				switch fn.Name.Name {
				case "String":
					fn.Name.Name = "Error"
					fixed = true
				case "Error":
					fn.Name.Name = "Err"
					fixed = true
				}
				return
			}

			// In type definition of an error implementation type,
			// rename Error field to Err to make room for method.
			// Given type XxxError struct { ... Error T } rename field to Err.
			d, ok := n.(*ast.GenDecl)
			if ok {
				for _, s := range d.Specs {
					switch s := s.(type) {
					case *ast.TypeSpec:
						if isErrorImpl(typeName(s.Name)) {
							st, ok := s.Type.(*ast.StructType)
							if ok {
								for _, f := range st.Fields.List {
									for _, n := range f.Names {
										if n.Name == "Error" {
											n.Name = "Err"
											fixed = true
										}
									}
								}
							}
						}
					}
				}
			}

			// For values that are an error implementation type,
			// rename .Error to .Err and .String to .Error
			sel, selok := n.(*ast.SelectorExpr)
			if selok && isErrorImpl(typeof[sel.X]) {
				switch sel.Sel.Name {
				case "Error":
					sel.Sel.Name = "Err"
					fixed = true
				case "String":
					sel.Sel.Name = "Error"
					fixed = true
				}
			}

			// Assume x.Err is an error value and rename .String to .Error
			// Children have been processed so the rewrite from Error to Err
			// has already happened there.
			if selok {
				if subsel, ok := sel.X.(*ast.SelectorExpr); ok && subsel.Sel.Name == "Err" && sel.Sel.Name == "String" {
					sel.Sel.Name = "Error"
					fixed = true
				}
			}

			// For values that are an error variable, rename .String to .Error.
			if selok && isErrorVar(sel.X) && sel.Sel.Name == "String" {
				sel.Sel.Name = "Error"
				fixed = true
			}

			// Rewrite composite literal of error type to turn Error: into Err:.
			lit, ok := n.(*ast.CompositeLit)
			if ok && isErrorImpl(typeof[lit]) {
				for _, e := range lit.Elts {
					if kv, ok := e.(*ast.KeyValueExpr); ok && isName(kv.Key, "Error") {
						kv.Key.(*ast.Ident).Name = "Err"
						fixed = true
					}
				}
			}

			// Rename os.Waitmsg to exec.ExitError
			// when used in a type assertion on an error.
			ta, ok := n.(*ast.TypeAssertExpr)
			if ok && isErrorVar(ta.X) && isPtrPkgDot(ta.Type, "os", "Waitmsg") {
				addImport(f, "exec")
				sel := ta.Type.(*ast.StarExpr).X.(*ast.SelectorExpr)
				sel.X.(*ast.Ident).Name = "exec"
				sel.Sel.Name = "ExitError"
				fixed = true
			}

		})
	}

	fix := func() {
		if fixed {
			return
		}
		fixed = true
		heuristic()
	}

	if force["error"] {
		heuristic()
	}

	walk(f, func(n interface{}) {
		p, ok := n.(*ast.Expr)
		if !ok {
			return
		}
		sel, ok := (*p).(*ast.SelectorExpr)
		if !ok {
			return
		}
		switch {
		case isPkgDot(sel, "os", "Error"):
			fix()
			*p = &ast.Ident{NamePos: sel.Pos(), Name: "error"}
		case isPkgDot(sel, "os", "NewError"):
			fix()
			addImport(f, "errors")
			sel.X.(*ast.Ident).Name = "errors"
			sel.Sel.Name = "New"
		case isPkgDot(sel, "os", "EOF"):
			fix()
			addImport(f, "io")
			sel.X.(*ast.Ident).Name = "io"
		}
	})

	if fixed && !usesImport(f, "os") {
		deleteImport(f, "os")
	}

	return fixed
}

func typeName(typ ast.Expr) string {
	if p, ok := typ.(*ast.StarExpr); ok {
		typ = p.X
	}
	id, ok := typ.(*ast.Ident)
	if ok {
		return id.Name
	}
	sel, ok := typ.(*ast.SelectorExpr)
	if ok {
		return typeName(sel.X) + "." + sel.Sel.Name
	}
	return ""
}
