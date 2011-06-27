// Copyright 2011 The Go Authors.  All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

package build

import (
	"go/parser"
	"go/token"
	"log"
	"os"
	"path/filepath"
	"strconv"
	"strings"
	"runtime"
)

type DirInfo struct {
	GoFiles  []string // .go files in dir (excluding CgoFiles)
	CgoFiles []string // .go files that import "C"
	CFiles   []string // .c files in dir
	SFiles   []string // .s files in dir
	Imports  []string // All packages imported by goFiles
	PkgName  string   // Name of package in dir
}

func (d *DirInfo) IsCommand() bool {
	return d.PkgName == "main"
}

// ScanDir returns a structure with details about the Go content found
// in the given directory. The file lists exclude:
//
//	- files in package main (unless allowMain is true)
//	- files in package documentation
//	- files ending in _test.go
// 	- files starting with _ or .
//
// Only files that satisfy the goodOSArch function are included.
func ScanDir(dir string, allowMain bool) (info *DirInfo, err os.Error) {
	f, err := os.Open(dir)
	if err != nil {
		return nil, err
	}
	dirs, err := f.Readdir(-1)
	f.Close()
	if err != nil {
		return nil, err
	}

	var di DirInfo
	imported := make(map[string]bool)
	fset := token.NewFileSet()
	for i := range dirs {
		d := &dirs[i]
		if strings.HasPrefix(d.Name, "_") ||
			strings.HasPrefix(d.Name, ".") {
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
			di.CFiles = append(di.CFiles, d.Name)
			continue
		case ".s":
			di.SFiles = append(di.SFiles, d.Name)
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
		if di.PkgName == "" {
			di.PkgName = s
		} else if di.PkgName != s {
			// Only if all files in the directory are in package main
			// do we return PkgName=="main".
			// A mix of main and another package reverts
			// to the original (allowMain=false) behaviour.
			if s == "main" || di.PkgName == "main" {
				return ScanDir(dir, false)
			}
			return nil, os.NewError("multiple package names in " + dir)
		}
		isCgo := false
		for _, spec := range pf.Imports {
			quoted := string(spec.Path.Value)
			path, err := strconv.Unquote(quoted)
			if err != nil {
				log.Panicf("%s: parser returned invalid quoted string: <%s>", filename, quoted)
			}
			imported[path] = true
			if path == "C" {
				isCgo = true
			}
		}
		if isCgo {
			di.CgoFiles = append(di.CgoFiles, d.Name)
		} else {
			di.GoFiles = append(di.GoFiles, d.Name)
		}
	}
	di.Imports = make([]string, len(imported))
	i := 0
	for p := range imported {
		di.Imports[i] = p
		i++
	}
	return &di, nil
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
	l := strings.Split(filename, "_")
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
