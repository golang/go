// Copyright 2011 The Go Authors.  All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

package main

import (
	"fmt"
	"go/build"
	"go/doc"
	"os"
	"path/filepath"
	"sort"
	"strings"
	"time"
)

// A Package describes a single package found in a directory.
type Package struct {
	// Note: These fields are part of the go command's public API.
	// See list.go.  It is okay to add fields, but not to change or
	// remove existing ones.  Keep in sync with list.go
	Name       string // package name
	Doc        string `json:",omitempty"` // package documentation string
	ImportPath string // import path of package in dir
	Dir        string // directory containing package sources
	Version    string `json:",omitempty"` // version of installed package (TODO)
	Standard   bool   `json:",omitempty"` // is this package part of the standard Go library?
	Stale      bool   `json:",omitempty"` // would 'go install' do anything for this package?

	// Source files
	GoFiles      []string // .go source files (excluding CgoFiles, TestGoFiles and XTestGoFiles)
	TestGoFiles  []string `json:",omitempty"` // _test.go source files internal to the package they are testing
	XTestGoFiles []string `json:",omitempty"` //_test.go source files external to the package they are testing
	CFiles       []string `json:",omitempty"` // .c source files
	HFiles       []string `json:",omitempty"` // .h source files
	SFiles       []string `json:",omitempty"` // .s source files
	CgoFiles     []string `json:",omitempty"` // .go sources files that import "C"
	CgoCFLAGS    []string `json:",omitempty"` // cgo: flags for C compiler
	CgoLDFLAGS   []string `json:",omitempty"` // cgo: flags for linker

	// Dependency information
	Imports []string `json:",omitempty"` // import paths used by this package
	Deps    []string `json:",omitempty"` // all (recursively) imported dependencies

	// Unexported fields are not part of the public API.
	t       *build.Tree
	pkgdir  string
	info    *build.DirInfo
	imports []*Package
	gofiles []string // GoFiles+CgoFiles+TestGoFiles+XTestGoFiles files, absolute paths
	target  string   // installed file for this package (may be executable)
	fake    bool     // synthesized package
}

// packageCache is a lookup cache for loadPackage,
// so that if we look up a package multiple times
// we return the same pointer each time.
var packageCache = map[string]*Package{}

// loadPackage scans directory named by arg,
// which is either an import path or a file system path
// (if the latter, must be rooted or begin with . or ..),
// and returns a *Package describing the package
// found in that directory.
func loadPackage(arg string) (*Package, error) {
	// Check package cache.
	if p := packageCache[arg]; p != nil {
		// We use p.imports==nil to detect a package that
		// is in the midst of its own loadPackage call
		// (all the recursion below happens before p.imports gets set).
		if p.imports == nil {
			return nil, fmt.Errorf("import loop at %s", arg)
		}
		return p, nil
	}

	// Find basic information about package path.
	t, importPath, err := build.FindTree(arg)
	dir := ""
	// Maybe it is a standard command.
	if err != nil && !filepath.IsAbs(arg) && strings.HasPrefix(arg, "cmd/") {
		goroot := build.Path[0]
		p := filepath.Join(goroot.Path, "src", arg)
		if st, err1 := os.Stat(p); err1 == nil && st.IsDir() {
			t = goroot
			importPath = arg
			dir = p
			err = nil
		}
	}
	if err != nil {
		return nil, err
	}

	if dir == "" {
		dir = filepath.Join(t.SrcDir(), filepath.FromSlash(importPath))
	}

	// Maybe we know the package by its directory.
	if p := packageCache[dir]; p != nil {
		if p.imports == nil {
			return nil, fmt.Errorf("import loop at %s", arg)
		}
		return p, nil
	}

	return scanPackage(&build.DefaultContext, t, arg, importPath, dir)
}

func scanPackage(ctxt *build.Context, t *build.Tree, arg, importPath, dir string) (*Package, error) {
	// Read the files in the directory to learn the structure
	// of the package.
	info, err := ctxt.ScanDir(dir)
	if err != nil {
		return nil, err
	}

	var target string
	if info.Package == "main" {
		_, elem := filepath.Split(importPath)
		target = filepath.Join(t.BinDir(), elem)
		if ctxt.GOOS == "windows" {
			target += ".exe"
		}
	} else {
		target = filepath.Join(t.PkgDir(), filepath.FromSlash(importPath)+".a")
	}

	p := &Package{
		Name:         info.Package,
		Doc:          doc.CommentText(info.PackageComment),
		ImportPath:   importPath,
		Dir:          dir,
		Imports:      info.Imports,
		GoFiles:      info.GoFiles,
		TestGoFiles:  info.TestGoFiles,
		XTestGoFiles: info.XTestGoFiles,
		CFiles:       info.CFiles,
		HFiles:       info.HFiles,
		SFiles:       info.SFiles,
		CgoFiles:     info.CgoFiles,
		CgoCFLAGS:    info.CgoCFLAGS,
		CgoLDFLAGS:   info.CgoLDFLAGS,
		Standard:     t.Goroot && !strings.Contains(importPath, "."),
		target:       target,
		t:            t,
		info:         info,
	}

	var built time.Time
	if fi, err := os.Stat(target); err == nil {
		built = fi.ModTime()
	}

	// Build list of full paths to all Go files in the package,
	// for use by commands like go fmt.
	for _, f := range info.GoFiles {
		p.gofiles = append(p.gofiles, filepath.Join(dir, f))
	}
	for _, f := range info.CgoFiles {
		p.gofiles = append(p.gofiles, filepath.Join(dir, f))
	}
	for _, f := range info.TestGoFiles {
		p.gofiles = append(p.gofiles, filepath.Join(dir, f))
	}
	for _, f := range info.XTestGoFiles {
		p.gofiles = append(p.gofiles, filepath.Join(dir, f))
	}

	sort.Strings(p.gofiles)

	srcss := [][]string{
		p.GoFiles,
		p.CFiles,
		p.HFiles,
		p.SFiles,
		p.CgoFiles,
	}
Stale:
	for _, srcs := range srcss {
		for _, src := range srcs {
			if fi, err := os.Stat(filepath.Join(p.Dir, src)); err != nil || fi.ModTime().After(built) {
				//println("STALE", p.ImportPath, "needs", src, err)
				p.Stale = true
				break Stale
			}
		}
	}

	importPaths := p.Imports
	// Packages that use cgo import runtime/cgo implicitly,
	// except runtime/cgo itself.
	if len(info.CgoFiles) > 0 && (!p.Standard || p.ImportPath != "runtime/cgo") {
		importPaths = append(importPaths, "runtime/cgo")
	}
	// Everything depends on runtime, except runtime and unsafe.
	if !p.Standard || (p.ImportPath != "runtime" && p.ImportPath != "unsafe") {
		importPaths = append(importPaths, "runtime")
	}

	// Record package under both import path and full directory name.
	packageCache[dir] = p
	packageCache[importPath] = p

	// Build list of imported packages and full dependency list.
	imports := make([]*Package, 0, len(p.Imports))
	deps := make(map[string]bool)
	for _, path := range importPaths {
		deps[path] = true
		if path == "C" {
			continue
		}
		p1, err := loadPackage(path)
		if err != nil {
			delete(packageCache, dir)
			delete(packageCache, importPath)
			// Add extra error detail to show full import chain.
			// Always useful, but especially useful in import loops.
			return nil, fmt.Errorf("%s: import %s\n\t%v", arg, path, err)
		}
		imports = append(imports, p1)
		for _, dep := range p1.Deps {
			deps[dep] = true
		}
		if p1.Stale {
			p.Stale = true
		}
		// p1.target can be empty only if p1 is not a real package,
		// such as package unsafe or the temporary packages
		// created during go test.
		if !p.Stale && p1.target != "" {
			if fi, err := os.Stat(p1.target); err != nil || fi.ModTime().After(built) {
				//println("STALE", p.ImportPath, "needs", p1.target, err)
				//println("BUILT", built.String(), "VS", fi.ModTime().String())
				p.Stale = true
			}
		}
	}
	p.imports = imports

	p.Deps = make([]string, 0, len(deps))
	for dep := range deps {
		p.Deps = append(p.Deps, dep)
	}
	sort.Strings(p.Deps)

	// unsafe is a fake package and is never out-of-date.
	if p.Standard && p.ImportPath == "unsafe" {
		p.Stale = false
		p.target = ""
	}

	return p, nil
}

// packages returns the packages named by the
// command line arguments 'args'.
func packages(args []string) []*Package {
	args = importPaths(args)
	var pkgs []*Package
	for _, arg := range args {
		pkg, err := loadPackage(arg)
		if err != nil {
			errorf("%s", err)
			continue
		}
		pkgs = append(pkgs, pkg)
	}
	return pkgs
}
