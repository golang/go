// Copyright 2011 The Go Authors.  All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

package build

import (
	"go/parser"
	"go/token"
	"io/ioutil"
	"log"
	"os"
	"path/filepath"
	"sort"
	"strconv"
	"strings"
	"runtime"
)

// A Context specifies the supporting context for a build.
type Context struct {
	GOARCH string // target architecture
	GOOS   string // target operating system
	// TODO(rsc,adg): GOPATH
}

// The DefaultContext is the default Context for builds.
// It uses the GOARCH and GOOS environment variables
// if set, or else the compiled code's GOARCH and GOOS.
var DefaultContext = Context{
	envOr("GOARCH", runtime.GOARCH),
	envOr("GOOS", runtime.GOOS),
}

func envOr(name, def string) string {
	s := os.Getenv(name)
	if s == "" {
		return def
	}
	return s
}

type DirInfo struct {
	GoFiles      []string // .go files in dir (excluding CgoFiles)
	CgoFiles     []string // .go files that import "C"
	CFiles       []string // .c files in dir
	SFiles       []string // .s files in dir
	Imports      []string // All packages imported by GoFiles
	TestImports  []string // All packages imported by (X)TestGoFiles
	PkgName      string   // Name of package in dir
	TestGoFiles  []string // _test.go files in package
	XTestGoFiles []string // _test.go files outside package
}

func (d *DirInfo) IsCommand() bool {
	return d.PkgName == "main"
}

// ScanDir calls DefaultContext.ScanDir.
func ScanDir(dir string, allowMain bool) (info *DirInfo, err os.Error) {
	return DefaultContext.ScanDir(dir, allowMain)
}

// ScanDir returns a structure with details about the Go content found
// in the given directory. The file lists exclude:
//
//	- files in package main (unless allowMain is true)
//	- files in package documentation
//	- files ending in _test.go
// 	- files starting with _ or .
//
func (ctxt *Context) ScanDir(dir string, allowMain bool) (info *DirInfo, err os.Error) {
	dirs, err := ioutil.ReadDir(dir)
	if err != nil {
		return nil, err
	}

	var di DirInfo
	imported := make(map[string]bool)
	testImported := make(map[string]bool)
	fset := token.NewFileSet()
	for _, d := range dirs {
		if strings.HasPrefix(d.Name, "_") ||
			strings.HasPrefix(d.Name, ".") {
			continue
		}
		if !ctxt.goodOSArch(d.Name) {
			continue
		}

		isTest := false
		switch filepath.Ext(d.Name) {
		case ".go":
			isTest = strings.HasSuffix(d.Name, "_test.go")
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
		pkg := string(pf.Name.Name)
		if pkg == "main" && !allowMain {
			continue
		}
		if pkg == "documentation" {
			continue
		}
		if isTest && strings.HasSuffix(pkg, "_test") {
			pkg = pkg[:len(pkg)-len("_test")]
		}
		if di.PkgName == "" {
			di.PkgName = pkg
		} else if di.PkgName != pkg {
			// Only if all files in the directory are in package main
			// do we return PkgName=="main".
			// A mix of main and another package reverts
			// to the original (allowMain=false) behaviour.
			if pkg == "main" || di.PkgName == "main" {
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
			if isTest {
				testImported[path] = true
			} else {
				imported[path] = true
			}
			if path == "C" {
				if isTest {
					return nil, os.NewError("use of cgo in test " + filename)
				}
				isCgo = true
			}
		}
		if isCgo {
			di.CgoFiles = append(di.CgoFiles, d.Name)
		} else if isTest {
			if pkg == string(pf.Name.Name) {
				di.TestGoFiles = append(di.TestGoFiles, d.Name)
			} else {
				di.XTestGoFiles = append(di.XTestGoFiles, d.Name)
			}
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
	di.TestImports = make([]string, len(testImported))
	i = 0
	for p := range testImported {
		di.TestImports[i] = p
		i++
	}
	// File name lists are sorted because ioutil.ReadDir sorts.
	sort.Strings(di.Imports)
	sort.Strings(di.TestImports)
	return &di, nil
}

// goodOSArch returns false if the name contains a $GOOS or $GOARCH
// suffix which does not match the current system.
// The recognized name formats are:
//
//     name_$(GOOS).*
//     name_$(GOARCH).*
//     name_$(GOOS)_$(GOARCH).*
//     name_$(GOOS)_test.*
//     name_$(GOARCH)_test.*
//     name_$(GOOS)_$(GOARCH)_test.*
//
func (ctxt *Context) goodOSArch(name string) bool {
	if dot := strings.Index(name, "."); dot != -1 {
		name = name[:dot]
	}
	l := strings.Split(name, "_")
	if n := len(l); n > 0 && l[n-1] == "test" {
		l = l[:n-1]
	}
	n := len(l)
	if n >= 2 && knownOS[l[n-2]] && knownArch[l[n-1]] {
		return l[n-2] == ctxt.GOOS && l[n-1] == ctxt.GOARCH
	}
	if n >= 1 && knownOS[l[n-1]] {
		return l[n-1] == ctxt.GOOS
	}
	if n >= 1 && knownArch[l[n-1]] {
		return l[n-1] == ctxt.GOARCH
	}
	return true
}

var knownOS = make(map[string]bool)
var knownArch = make(map[string]bool)

func init() {
	for _, v := range strings.Fields(goosList) {
		knownOS[v] = true
	}
	for _, v := range strings.Fields(goarchList) {
		knownArch[v] = true
	}
}
