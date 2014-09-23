// Copyright 2014 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

package rename

import (
	"fmt"
	"go/build"
	"os"
	"path/filepath"
	"reflect"
	"strings"
	"unicode"

	"code.google.com/p/go.tools/go/types"
)

func objectKind(obj types.Object) string {
	switch obj := obj.(type) {
	case *types.PkgName:
		return "imported package name"
	case *types.TypeName:
		return "type"
	case *types.Var:
		if obj.IsField() {
			return "field"
		}
	case *types.Func:
		if obj.Type().(*types.Signature).Recv() != nil {
			return "method"
		}
	}
	// label, func, var, const
	return strings.ToLower(strings.TrimPrefix(reflect.TypeOf(obj).String(), "*types."))
}

func typeKind(T types.Type) string {
	return strings.ToLower(strings.TrimPrefix(reflect.TypeOf(T.Underlying()).String(), "*types."))
}

// NB: for renamings, blank is not considered valid.
func isValidIdentifier(id string) bool {
	if id == "" || id == "_" {
		return false
	}
	for i, r := range id {
		if !isLetter(r) && (i == 0 || !isDigit(r)) {
			return false
		}
	}
	return true
}

// isLocal reports whether obj is local to some function.
// Precondition: not a struct field or interface method.
func isLocal(obj types.Object) bool {
	// [... 5=stmt 4=func 3=file 2=pkg 1=universe]
	var depth int
	for scope := obj.Parent(); scope != nil; scope = scope.Parent() {
		depth++
	}
	return depth >= 4
}

func isPackageLevel(obj types.Object) bool {
	return obj.Pkg().Scope().Lookup(obj.Name()) == obj
}

// -- Plundered from go/scanner: ---------------------------------------

func isLetter(ch rune) bool {
	return 'a' <= ch && ch <= 'z' || 'A' <= ch && ch <= 'Z' || ch == '_' || ch >= 0x80 && unicode.IsLetter(ch)
}

func isDigit(ch rune) bool {
	return '0' <= ch && ch <= '9' || ch >= 0x80 && unicode.IsDigit(ch)
}

// -- Plundered from code.google.com/p/go.tools/oracle -----------------

// guessImportPath finds the package containing filename, and returns
// its import path relative to it.
func guessImportPath(filename string, ctxt *build.Context) (importPath string, err error) {
	// TODO(adonovan): move this to package "buildutil"; factor in common with oracle.
	// bp, err := buildutil.ContainingPackage(ctxt, wd, filename)
	// if err != nil {
	// 	return
	// }
	// return bp.ImportPath, nil

	absFile, err := filepath.Abs(filename)
	if err != nil {
		err = fmt.Errorf("can't form absolute path of %s", filename)
		return
	}
	absFileDir := segments(filepath.Dir(absFile))

	// Find the innermost directory in $GOPATH that encloses filename.
	minD := 1024
	for _, gopathDir := range ctxt.SrcDirs() {
		// We can assume $GOPATH and $GOROOT dirs are absolute,
		// thus gopathDir too, and that it exists.
		d := prefixLen(segments(gopathDir), absFileDir)
		// If there are multiple matches,
		// prefer the innermost enclosing directory
		// (smallest d).
		if d >= 0 && d < minD {
			minD = d
			importPath = strings.Join(absFileDir[len(absFileDir)-minD:], string(os.PathSeparator))
		}
	}
	if importPath == "" {
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

// sameFile returns true if x and y have the same basename and denote
// the same file.
//
func sameFile(x, y string) bool {
	if x == y {
		return true
	}
	if filepath.Base(x) == filepath.Base(y) { // (optimisation)
		if xi, err := os.Stat(x); err == nil {
			if yi, err := os.Stat(y); err == nil {
				return os.SameFile(xi, yi)
			}
		}
	}
	return false
}
