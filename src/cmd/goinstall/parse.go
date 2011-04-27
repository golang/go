// Copyright 2010 The Go Authors.  All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

// Wrappers for Go parser.

package main

import (
	"go/ast"
	"go/parser"
	"log"
	"os"
	"path/filepath"
	"strconv"
	"strings"
	"runtime"
)


type dirInfo struct {
	goFiles  []string // .go files within dir (including cgoFiles)
	cgoFiles []string // .go files that import "C"
	cFiles   []string // .c files within dir
	sFiles   []string // .s files within dir
	imports  []string // All packages imported by goFiles
	pkgName  string   // Name of package within dir
}

// scanDir returns a structure with details about the Go content found
// in the given directory. The list of files will NOT contain the
// following entries:
//
// - Files in package main (unless allowMain is true)
// - Files ending in _test.go
// - Files starting with _ (temporary)
// - Files containing .cgo in their names
//
// The imports map keys are package paths imported by listed Go files,
// and the values are the Go files importing the respective package paths.
func scanDir(dir string, allowMain bool) (info *dirInfo, err os.Error) {
	f, err := os.Open(dir)
	if err != nil {
		return nil, err
	}
	dirs, err := f.Readdir(-1)
	f.Close()
	if err != nil {
		return nil, err
	}

	goFiles := make([]string, 0, len(dirs))
	cgoFiles := make([]string, 0, len(dirs))
	cFiles := make([]string, 0, len(dirs))
	sFiles := make([]string, 0, len(dirs))
	importsm := make(map[string]bool)
	pkgName := ""
	for i := range dirs {
		d := &dirs[i]
		if strings.HasPrefix(d.Name, "_") || strings.Index(d.Name, ".cgo") != -1 {
			continue
		}
		if !goodOSArch(d.Name) {
			continue
		}

		switch filepath.Ext(d.Name) {
		case ".go":
			if strings.HasSuffix(d.Name, "_test.go") {
				continue
			}
		case ".c":
			cFiles = append(cFiles, d.Name)
			continue
		case ".s":
			sFiles = append(sFiles, d.Name)
			continue
		default:
			continue
		}

		filename := filepath.Join(dir, d.Name)
		pf, err := parser.ParseFile(fset, filename, nil, parser.ImportsOnly)
		if err != nil {
			return nil, err
		}
		s := string(pf.Name.Name)
		if s == "main" && !allowMain {
			continue
		}
		if s == "documentation" {
			continue
		}
		if pkgName == "" {
			pkgName = s
		} else if pkgName != s {
			// Only if all files in the directory are in package main
			// do we return pkgName=="main".
			// A mix of main and another package reverts
			// to the original (allowMain=false) behaviour.
			if s == "main" || pkgName == "main" {
				return scanDir(dir, false)
			}
			return nil, os.ErrorString("multiple package names in " + dir)
		}
		goFiles = append(goFiles, d.Name)
		for _, decl := range pf.Decls {
			for _, spec := range decl.(*ast.GenDecl).Specs {
				quoted := string(spec.(*ast.ImportSpec).Path.Value)
				unquoted, err := strconv.Unquote(quoted)
				if err != nil {
					log.Panicf("%s: parser returned invalid quoted string: <%s>", filename, quoted)
				}
				importsm[unquoted] = true
				if unquoted == "C" {
					cgoFiles = append(cgoFiles, d.Name)
				}
			}
		}
	}
	imports := make([]string, len(importsm))
	i := 0
	for p := range importsm {
		imports[i] = p
		i++
	}
	return &dirInfo{goFiles, cgoFiles, cFiles, sFiles, imports, pkgName}, nil
}

// goodOSArch returns false if the filename contains a $GOOS or $GOARCH
// suffix which does not match the current system.
// The recognized filename formats are:
//
//     name_$(GOOS).*
//     name_$(GOARCH).*
//     name_$(GOOS)_$(GOARCH).*
//
func goodOSArch(filename string) bool {
	if dot := strings.Index(filename, "."); dot != -1 {
		filename = filename[:dot]
	}
	l := strings.Split(filename, "_", -1)
	n := len(l)
	if n == 0 {
		return true
	}
	if good, known := goodOS[l[n-1]]; known {
		return good
	}
	if good, known := goodArch[l[n-1]]; known {
		if !good || n < 2 {
			return false
		}
		good, known = goodOS[l[n-2]]
		return good || !known
	}
	return true
}

var goodOS = make(map[string]bool)
var goodArch = make(map[string]bool)

func init() {
	goodOS = make(map[string]bool)
	goodArch = make(map[string]bool)
	for _, v := range strings.Fields(goosList) {
		goodOS[v] = v == runtime.GOOS
	}
	for _, v := range strings.Fields(goarchList) {
		goodArch[v] = v == runtime.GOARCH
	}
}
