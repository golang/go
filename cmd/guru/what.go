// Copyright 2013 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

package main

import (
	"fmt"
	"go/ast"
	"go/build"
	"go/token"
	"os"
	"path"
	"path/filepath"
	"sort"
	"strings"

	"golang.org/x/tools/cmd/guru/serial"
	"golang.org/x/tools/go/ast/astutil"
)

// what reports all the information about the query selection that can be
// obtained from parsing only its containing source file.
// It is intended to be a very low-latency query callable from GUI
// tools, e.g. to populate a menu of options of slower queries about
// the selected location.
//
func what(q *Query) error {
	qpos, err := fastQueryPos(q.Build, q.Pos)
	if err != nil {
		return err
	}

	// (ignore errors)
	srcdir, importPath, _ := guessImportPath(qpos.fset.File(qpos.start).Name(), q.Build)

	// Determine which query modes are applicable to the selection.
	enable := map[string]bool{
		"describe": true, // any syntax; always enabled
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

		// For pointsto and whicherrs, we approximate findInterestingNode.
		if _, ok := enable["pointsto"]; !ok {
			switch n.(type) {
			case ast.Stmt,
				*ast.ArrayType,
				*ast.StructType,
				*ast.FuncType,
				*ast.InterfaceType,
				*ast.MapType,
				*ast.ChanType:
				// not an expression
				enable["pointsto"] = false
				enable["whicherrs"] = false

			case ast.Expr, ast.Decl, *ast.ValueSpec:
				// an expression, maybe
				enable["pointsto"] = true
				enable["whicherrs"] = true

			default:
				// Comment, Field, KeyValueExpr, etc: ascend.
			}
		}
	}

	// If we don't have an exact selection, disable modes that need one.
	if !qpos.exact {
		enable["callees"] = false
		enable["pointsto"] = false
		enable["whicherrs"] = false
		enable["describe"] = false
	}

	var modes []string
	for mode := range enable {
		modes = append(modes, mode)
	}
	sort.Strings(modes)

	// Find the object referred to by the selection (if it's an
	// identifier) and report the position of each identifier
	// that refers to the same object.
	//
	// This may return spurious matches (e.g. struct fields) because
	// it uses the best-effort name resolution done by go/parser.
	var sameids []token.Pos
	var object string
	if id, ok := qpos.path[0].(*ast.Ident); ok {
		if id.Obj == nil {
			// An unresolved identifier is potentially a package name.
			// Resolve them with a simple importer (adds ~100µs).
			importer := func(imports map[string]*ast.Object, path string) (*ast.Object, error) {
				pkg, ok := imports[path]
				if !ok {
					pkg = &ast.Object{
						Kind: ast.Pkg,
						Name: filepath.Base(path), // a guess
					}
					imports[path] = pkg
				}
				return pkg, nil
			}
			f := qpos.path[len(qpos.path)-1].(*ast.File)
			ast.NewPackage(qpos.fset, map[string]*ast.File{"": f}, importer, nil)
		}

		if id.Obj != nil {
			object = id.Obj.Name
			decl := qpos.path[len(qpos.path)-1]
			ast.Inspect(decl, func(n ast.Node) bool {
				if n, ok := n.(*ast.Ident); ok && n.Obj == id.Obj {
					sameids = append(sameids, n.Pos())
				}
				return true
			})
		}
	}

	q.Output(qpos.fset, &whatResult{
		path:       qpos.path,
		srcdir:     srcdir,
		importPath: importPath,
		modes:      modes,
		object:     object,
		sameids:    sameids,
	})
	return nil
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
		return "", "", fmt.Errorf("can't form absolute path of %s: %v", filename, err)
	}

	absFileDir := filepath.Dir(absFile)
	resolvedAbsFileDir, err := filepath.EvalSymlinks(absFileDir)
	if err != nil {
		return "", "", fmt.Errorf("can't evaluate symlinks of %s: %v", absFileDir, err)
	}

	segmentedAbsFileDir := segments(resolvedAbsFileDir)
	// Find the innermost directory in $GOPATH that encloses filename.
	minD := 1024
	for _, gopathDir := range buildContext.SrcDirs() {
		absDir, err := filepath.Abs(gopathDir)
		if err != nil {
			continue // e.g. non-existent dir on $GOPATH
		}
		resolvedAbsDir, err := filepath.EvalSymlinks(absDir)
		if err != nil {
			continue // e.g. non-existent dir on $GOPATH
		}

		d := prefixLen(segments(resolvedAbsDir), segmentedAbsFileDir)
		// If there are multiple matches,
		// prefer the innermost enclosing directory
		// (smallest d).
		if d >= 0 && d < minD {
			minD = d
			srcdir = gopathDir
			importPath = path.Join(segmentedAbsFileDir[len(segmentedAbsFileDir)-minD:]...)
		}
	}
	if srcdir == "" {
		return "", "", fmt.Errorf("directory %s is not beneath any of these GOROOT/GOPATH directories: %s",
			filepath.Dir(absFile), strings.Join(buildContext.SrcDirs(), ", "))
	}
	if importPath == "" {
		// This happens for e.g. $GOPATH/src/a.go, but
		// "" is not a valid path for (*go/build).Import.
		return "", "", fmt.Errorf("cannot load package in root of source directory %s", srcdir)
	}
	return srcdir, importPath, nil
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
	object     string
	sameids    []token.Pos
}

func (r *whatResult) PrintPlain(printf printfFunc) {
	for _, n := range r.path {
		printf(n, "%s", astutil.NodeDescription(n))
	}
	printf(nil, "modes: %s", r.modes)
	printf(nil, "srcdir: %s", r.srcdir)
	printf(nil, "import path: %s", r.importPath)
	for _, pos := range r.sameids {
		printf(pos, "%s", r.object)
	}
}

func (r *whatResult) JSON(fset *token.FileSet) []byte {
	var enclosing []serial.SyntaxNode
	for _, n := range r.path {
		enclosing = append(enclosing, serial.SyntaxNode{
			Description: astutil.NodeDescription(n),
			Start:       fset.Position(n.Pos()).Offset,
			End:         fset.Position(n.End()).Offset,
		})
	}

	var sameids []string
	for _, pos := range r.sameids {
		sameids = append(sameids, fset.Position(pos).String())
	}

	return toJSON(&serial.What{
		Modes:      r.modes,
		SrcDir:     r.srcdir,
		ImportPath: r.importPath,
		Enclosing:  enclosing,
		Object:     r.object,
		SameIDs:    sameids,
	})
}
