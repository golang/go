// Copyright 2011 The Go Authors.  All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

package main

import (
	"bytes"
	"fmt"
	"go/build"
	"go/scanner"
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
	ImportPath string        // import path of package in dir
	Name       string        `json:",omitempty"` // package name
	Doc        string        `json:",omitempty"` // package documentation string
	Dir        string        `json:",omitempty"` // directory containing package sources
	Target     string        `json:",omitempty"` // install path
	Version    string        `json:",omitempty"` // version of installed package (TODO)
	Standard   bool          `json:",omitempty"` // is this package part of the standard Go library?
	Stale      bool          `json:",omitempty"` // would 'go install' do anything for this package?
	Incomplete bool          `json:",omitempty"` // was there an error loading this package or dependencies?
	Error      *PackageError `json:",omitempty"` // error loading this package (not dependencies)

	// Source files
	GoFiles      []string `json:",omitempty"` // .go source files (excluding CgoFiles, TestGoFiles and XTestGoFiles)
	TestGoFiles  []string `json:",omitempty"` // _test.go source files internal to the package they are testing
	XTestGoFiles []string `json:",omitempty"` //_test.go source files external to the package they are testing
	CFiles       []string `json:",omitempty"` // .c source files
	HFiles       []string `json:",omitempty"` // .h source files
	SFiles       []string `json:",omitempty"` // .s source files
	CgoFiles     []string `json:",omitempty"` // .go sources files that import "C"
	CgoCFLAGS    []string `json:",omitempty"` // cgo: flags for C compiler
	CgoLDFLAGS   []string `json:",omitempty"` // cgo: flags for linker

	// Dependency information
	Imports    []string        `json:",omitempty"` // import paths used by this package
	Deps       []string        `json:",omitempty"` // all (recursively) imported dependencies
	DepsErrors []*PackageError `json:",omitempty"` // errors loading dependencies

	// Unexported fields are not part of the public API.
	t       *build.Tree
	pkgdir  string
	info    *build.DirInfo
	imports []*Package
	deps    []*Package
	gofiles []string // GoFiles+CgoFiles+TestGoFiles+XTestGoFiles files, absolute paths
	target  string   // installed file for this package (may be executable)
	fake    bool     // synthesized package
}

// A PackageError describes an error loading information about a package.
type PackageError struct {
	ImportStack []string // shortest path from package named on command line to this one
	Pos         string   // position of error
	Err         string   // the error itself
}

func (p *PackageError) Error() string {
	if p.Pos != "" {
		return strings.Join(p.ImportStack, "\n\timports ") + ": " + p.Pos + ": " + p.Err
	}
	return strings.Join(p.ImportStack, "\n\timports ") + ": " + p.Err
}

// An importStack is a stack of import paths.
type importStack []string

func (s *importStack) push(p string) {
	*s = append(*s, p)
}

func (s *importStack) pop() {
	*s = (*s)[0 : len(*s)-1]
}

func (s *importStack) copy() []string {
	return append([]string{}, *s...)
}

// shorterThan returns true if sp is shorter than t.
// We use this to record the shortest import sequence
// that leads to a particular package.
func (sp *importStack) shorterThan(t []string) bool {
	s := *sp
	if len(s) != len(t) {
		return len(s) < len(t)
	}
	// If they are the same length, settle ties using string ordering.
	for i := range s {
		if s[i] != t[i] {
			return s[i] < t[i]
		}
	}
	return false // they are equal
}

// packageCache is a lookup cache for loadPackage,
// so that if we look up a package multiple times
// we return the same pointer each time.
var packageCache = map[string]*Package{}

// reloadPackage is like loadPackage but makes sure
// not to use the package cache.
func reloadPackage(arg string, stk *importStack) *Package {
	p := packageCache[arg]
	if p != nil {
		delete(packageCache, p.Dir)
		delete(packageCache, p.ImportPath)
	}
	return loadPackage(arg, stk)
}

// loadPackage scans directory named by arg,
// which is either an import path or a file system path
// (if the latter, must be rooted or begin with . or ..),
// and returns a *Package describing the package
// found in that directory.
func loadPackage(arg string, stk *importStack) *Package {
	stk.push(arg)
	defer stk.pop()

	// Check package cache.
	if p := packageCache[arg]; p != nil {
		return reusePackage(p, stk)
	}

	// Find basic information about package path.
	isCmd := false
	t, importPath, err := build.FindTree(arg)
	dir := ""
	// Maybe it is a standard command.
	if err != nil && strings.HasPrefix(arg, "cmd/") {
		goroot := build.Path[0]
		p := filepath.Join(goroot.Path, "src", arg)
		if st, err1 := os.Stat(p); err1 == nil && st.IsDir() {
			t = goroot
			importPath = arg
			dir = p
			err = nil
			isCmd = true
		}
	}
	// Maybe it is a path to a standard command.
	if err != nil && (filepath.IsAbs(arg) || isLocalPath(arg)) {
		arg, _ := filepath.Abs(arg)
		goroot := build.Path[0]
		cmd := filepath.Join(goroot.Path, "src", "cmd") + string(filepath.Separator)
		if st, err1 := os.Stat(arg); err1 == nil && st.IsDir() && strings.HasPrefix(arg, cmd) {
			t = goroot
			importPath = filepath.FromSlash(arg[len(cmd):])
			dir = arg
			err = nil
			isCmd = true
		}
	}
	if err != nil {
		p := &Package{
			ImportPath: arg,
			Error: &PackageError{
				ImportStack: stk.copy(),
				Err:         err.Error(),
			},
			Incomplete: true,
		}
		packageCache[arg] = p
		return p
	}

	if dir == "" {
		dir = filepath.Join(t.SrcDir(), filepath.FromSlash(importPath))
	}

	// Maybe we know the package by its directory.
	p := packageCache[dir]
	if p != nil {
		packageCache[importPath] = p
		p = reusePackage(p, stk)
	} else {
		p = scanPackage(&buildContext, t, arg, importPath, dir, stk, false)
	}

	// If we loaded the files from the Go root's cmd/ tree,
	// it must be a command (package main).
	if isCmd && p.Error == nil && p.Name != "main" {
		p.Error = &PackageError{
			ImportStack: stk.copy(),
			Err:         fmt.Sprintf("expected package main in %q; found package %s", dir, p.Name),
		}
	}
	return p
}

func reusePackage(p *Package, stk *importStack) *Package {
	// We use p.imports==nil to detect a package that
	// is in the midst of its own loadPackage call
	// (all the recursion below happens before p.imports gets set).
	if p.imports == nil {
		if p.Error == nil {
			p.Error = &PackageError{
				ImportStack: stk.copy(),
				Err:         "import loop",
			}
		}
		p.Incomplete = true
	}
	if p.Error != nil && stk.shorterThan(p.Error.ImportStack) {
		p.Error.ImportStack = stk.copy()
	}
	return p
}

// firstSentence returns the first sentence of the document text.
// The sentence ends after the first period followed by a space.
// The returned sentence will have no \n \r or \t characters and
// will use only single spaces between words.
func firstSentence(text string) string {
	var b []byte
	space := true
Loop:
	for i := 0; i < len(text); i++ {
		switch c := text[i]; c {
		case ' ', '\t', '\r', '\n':
			if !space {
				space = true
				if len(b) > 0 && b[len(b)-1] == '.' {
					break Loop
				}
				b = append(b, ' ')
			}
		default:
			space = false
			b = append(b, c)
		}
	}
	return string(b)
}

// isGoTool is the list of directories for Go programs that are installed in
// $GOROOT/bin/tool.
var isGoTool = map[string]bool{
	"cmd/api":      true,
	"cmd/cgo":      true,
	"cmd/fix":      true,
	"cmd/vet":      true,
	"cmd/yacc":     true,
	"exp/gotype":   true,
	"exp/ebnflint": true,
}

func scanPackage(ctxt *build.Context, t *build.Tree, arg, importPath, dir string, stk *importStack, useAllFiles bool) *Package {
	// Read the files in the directory to learn the structure
	// of the package.
	p := &Package{
		ImportPath: importPath,
		Dir:        dir,
		Standard:   t.Goroot && !strings.Contains(importPath, "."),
		t:          t,
	}
	packageCache[dir] = p
	packageCache[importPath] = p

	ctxt.UseAllFiles = useAllFiles
	info, err := ctxt.ScanDir(dir)
	useAllFiles = false // flag does not apply to dependencies
	if err != nil {
		p.Error = &PackageError{
			ImportStack: stk.copy(),
			Err:         err.Error(),
		}
		// Look for parser errors.
		if err, ok := err.(scanner.ErrorList); ok {
			// Prepare error with \n before each message.
			// When printed in something like context: %v
			// this will put the leading file positions each on
			// its own line.  It will also show all the errors
			// instead of just the first, as err.Error does.
			var buf bytes.Buffer
			for _, e := range err {
				buf.WriteString("\n")
				buf.WriteString(e.Error())
			}
			p.Error.Err = buf.String()
		}
		p.Incomplete = true
		return p
	}

	p.info = info
	p.Name = info.Package
	p.Doc = firstSentence(info.PackageComment.Text())
	p.Imports = info.Imports
	p.GoFiles = info.GoFiles
	p.TestGoFiles = info.TestGoFiles
	p.XTestGoFiles = info.XTestGoFiles
	p.CFiles = info.CFiles
	p.HFiles = info.HFiles
	p.SFiles = info.SFiles
	p.CgoFiles = info.CgoFiles
	p.CgoCFLAGS = info.CgoCFLAGS
	p.CgoLDFLAGS = info.CgoLDFLAGS

	if info.Package == "main" {
		_, elem := filepath.Split(importPath)
		full := ctxt.GOOS + "_" + ctxt.GOARCH + "/" + elem
		if t.Goroot && isGoTool[p.ImportPath] {
			p.target = filepath.Join(t.Path, "pkg/tool", full)
		} else {
			if ctxt.GOOS != toolGOOS || ctxt.GOARCH != toolGOARCH {
				// Install cross-compiled binaries to subdirectories of bin.
				elem = full
			}
			p.target = filepath.Join(t.BinDir(), elem)
		}
		if ctxt.GOOS == "windows" {
			p.target += ".exe"
		}
	} else {
		dir := t.PkgDir()
		// For gccgo, rewrite p.target with the expected library name.
		if _, ok := buildToolchain.(gccgoToolchain); ok {
			dir = filepath.Join(filepath.Dir(dir), "gccgo", filepath.Base(dir))
		}
		p.target = buildToolchain.pkgpath(dir, p)
	}

	var built time.Time
	if fi, err := os.Stat(p.target); err == nil {
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
		if path == "C" {
			continue
		}
		deps[path] = true
		p1 := loadPackage(path, stk)
		if p1.Error != nil {
			if info.ImportPos != nil && len(info.ImportPos[path]) > 0 {
				pos := info.ImportPos[path][0]
				p1.Error.Pos = pos.String()
			}
		}
		imports = append(imports, p1)
		for _, dep := range p1.Deps {
			deps[dep] = true
		}
		if p1.Stale {
			p.Stale = true
		}
		if p1.Incomplete {
			p.Incomplete = true
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
	for _, dep := range p.Deps {
		p1 := packageCache[dep]
		if p1 == nil {
			panic("impossible: missing entry in package cache for " + dep + " imported by " + p.ImportPath)
		}
		p.deps = append(p.deps, p1)
		if p1.Error != nil {
			p.DepsErrors = append(p.DepsErrors, p1.Error)
		}
	}

	// unsafe is a fake package and is never out-of-date.
	if p.Standard && p.ImportPath == "unsafe" {
		p.Stale = false
		p.target = ""
	}

	p.Target = p.target

	return p
}

// packages returns the packages named by the
// command line arguments 'args'.  If a named package
// cannot be loaded at all (for example, if the directory does not exist),
// then packages prints an error and does not include that
// package in the results.  However, if errors occur trying
// to load dependencies of a named package, the named
// package is still returned, with p.Incomplete = true
// and details in p.DepsErrors.
func packages(args []string) []*Package {
	args = importPaths(args)
	var pkgs []*Package
	var stk importStack
	for _, arg := range args {
		pkg := loadPackage(arg, &stk)
		if pkg.Error != nil {
			errorf("can't load package: %s", pkg.Error)
			continue
		}
		pkgs = append(pkgs, pkg)
	}
	return pkgs
}

// packagesAndErrors is like 'packages' but returns a 
// *Package for every argument, even the ones that
// cannot be loaded at all.
// The packages that fail to load will have p.Error != nil.
func packagesAndErrors(args []string) []*Package {
	args = importPaths(args)
	var pkgs []*Package
	var stk importStack
	for _, arg := range args {
		pkgs = append(pkgs, loadPackage(arg, &stk))
	}
	return pkgs
}

// packagesForBuild is like 'packages' but fails if any of
// the packages or their dependencies have errors
// (cannot be built).
func packagesForBuild(args []string) []*Package {
	pkgs := packagesAndErrors(args)
	printed := map[*PackageError]bool{}
	for _, pkg := range pkgs {
		if pkg.Error != nil {
			errorf("can't load package: %s", pkg.Error)
		}
		for _, err := range pkg.DepsErrors {
			// Since these are errors in dependencies,
			// the same error might show up multiple times,
			// once in each package that depends on it.
			// Only print each once.
			if !printed[err] {
				printed[err] = true
				errorf("%s", err)
			}
		}
	}
	exitIfErrors()
	return pkgs
}
