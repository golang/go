// Copyright 2013 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

package oracle

import (
	"fmt"
	"go/ast"
	"go/build"
	"go/token"
	"os"
	"path/filepath"
	"sort"
	"strings"

	"code.google.com/p/go.tools/astutil"
	"code.google.com/p/go.tools/oracle/serial"
)

// what reports all the information about the query selection that can be
// obtained from parsing only its containing source file.
// It is intended to be a very low-latency query callable from GUI
// tools, e.g. to populate a menu of options of slower queries about
// the selected location.
//
func what(posFlag string, buildContext *build.Context) (*Result, error) {
	qpos, err := fastQueryPos(posFlag)
	if err != nil {
		return nil, err
	}

	// (ignore errors)
	srcdir, importPath, _ := guessImportPath(qpos.fset.File(qpos.start).Name(), buildContext)

	// Determine which query modes are applicable to the selection.
	// TODO(adonovan): refactor: make each minfo have an 'enable'
	// predicate over qpos.
	enable := map[string]bool{
		"callgraph": true, // whole program; always enabled
		"describe":  true, // any syntax; always enabled
	}

	if qpos.end > qpos.start {
		enable["freevars"] = true // nonempty selection?
	}

	for _, n := range qpos.path {
		switch n := n.(type) {
		case *ast.Ident:
			enable["definition"] = true
			enable["referrers"] = true
			enable["implements"] = true
		case *ast.CallExpr:
			enable["callees"] = true
		case *ast.FuncDecl:
			enable["callers"] = true
			enable["callstack"] = true
		case *ast.SendStmt:
			enable["peers"] = true
		case *ast.UnaryExpr:
			if n.Op == token.ARROW {
				enable["peers"] = true
			}
		}

		// For implements, we approximate findInterestingNode.
		if _, ok := enable["implements"]; !ok {
			switch n.(type) {
			case *ast.ArrayType,
				*ast.StructType,
				*ast.FuncType,
				*ast.InterfaceType,
				*ast.MapType,
				*ast.ChanType:
				enable["implements"] = true
			}
		}

		// For pointsto, we approximate findInterestingNode.
		if _, ok := enable["pointsto"]; !ok {
			switch n.(type) {
			case ast.Stmt,
				*ast.ArrayType,
				*ast.StructType,
				*ast.FuncType,
				*ast.InterfaceType,
				*ast.MapType,
				*ast.ChanType:
				enable["pointsto"] = false // not an expr

			case ast.Expr, ast.Decl, *ast.ValueSpec:
				enable["pointsto"] = true // an expr, maybe

			default:
				// Comment, Field, KeyValueExpr, etc: ascend.
			}
		}
	}

	// If we don't have an exact selection, disable modes that need one.
	if !qpos.exact {
		for _, minfo := range modes {
			if minfo.needs&needExactPos != 0 {
				enable[minfo.name] = false
			}
		}
	}

	var modes []string
	for mode := range enable {
		modes = append(modes, mode)
	}
	sort.Strings(modes)

	return &Result{
		mode: "what",
		fset: qpos.fset,
		q: &whatResult{
			path:       qpos.path,
			srcdir:     srcdir,
			importPath: importPath,
			modes:      modes,
		},
	}, nil

}

// guessImportPath finds the package containing filename, and returns
// its source directory (an element of $GOPATH) and its import path
// relative to it.
//
// TODO(adonovan): what about _test.go files that are not part of the
// package?
//
func guessImportPath(filename string, buildContext *build.Context) (srcdir, importPath string, err error) {
	absFile, err := filepath.Abs(filename)
	if err != nil {
		err = fmt.Errorf("can't form absolute path of %s", filename)
		return
	}
	absFileDir := segments(filepath.Dir(absFile))

	// Find the innermost directory in $GOPATH that encloses filename.
	minD := 1024
	for _, gopathDir := range buildContext.SrcDirs() {
		absDir, err := filepath.Abs(gopathDir)
		if err != nil {
			continue // e.g. non-existent dir on $GOPATH
		}
		d := prefixLen(segments(absDir), absFileDir)
		// If there are multiple matches,
		// prefer the innermost enclosing directory
		// (smallest d).
		if d >= 0 && d < minD {
			minD = d
			srcdir = gopathDir
			importPath = strings.Join(absFileDir[len(absFileDir)-minD:], string(os.PathSeparator))
		}
	}
	if srcdir == "" {
		err = fmt.Errorf("can't find package for file %s", filename)
	}
	return
}

func segments(path string) []string {
	return strings.Split(path, string(os.PathSeparator))
}

// prefixLen returns the length of the remainder of y if x is a prefix
// of y, a negative number otherwise.
func prefixLen(x, y []string) int {
	d := len(y) - len(x)
	if d >= 0 {
		for i := range x {
			if y[i] != x[i] {
				return -1 // not a prefix
			}
		}
	}
	return d
}

type whatResult struct {
	path       []ast.Node
	modes      []string
	srcdir     string
	importPath string
}

func (r *whatResult) display(printf printfFunc) {
	for _, n := range r.path {
		printf(n, "%s", astutil.NodeDescription(n))
	}
	printf(nil, "modes: %s", r.modes)
	printf(nil, "srcdir: %s", r.srcdir)
	printf(nil, "import path: %s", r.importPath)
}

func (r *whatResult) toSerial(res *serial.Result, fset *token.FileSet) {
	var enclosing []serial.SyntaxNode
	for _, n := range r.path {
		enclosing = append(enclosing, serial.SyntaxNode{
			Description: astutil.NodeDescription(n),
			Start:       fset.Position(n.Pos()).Offset,
			End:         fset.Position(n.End()).Offset,
		})
	}
	res.What = &serial.What{
		Modes:      r.modes,
		SrcDir:     r.srcdir,
		ImportPath: r.importPath,
		Enclosing:  enclosing,
	}
}
