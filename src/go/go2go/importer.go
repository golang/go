// Copyright 2020 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

package go2go

import (
	"bytes"
	"fmt"
	"go/ast"
	"go/build"
	"go/importer"
	"go/parser"
	"go/token"
	"go/types"
	"internal/goroot"
	"io/ioutil"
	"log"
	"os"
	"os/exec"
	"path"
	"path/filepath"
	"runtime"
	"sort"
	"strings"
	"unicode"
)

// Importer implements the types.ImporterFrom interface.
// It looks for Go2 packages using GO2PATH.
// Imported Go2 packages are rewritten to normal Go packages.
// This type also tracks references across imported packages.
type Importer struct {
	// The default importer, for Go1 packages.
	defaultImporter types.ImporterFrom

	// Temporary directory used to rewrite packages.
	tmpdir string

	// Aggregated info from go/types.
	info *types.Info

	// Map from import path to directory holding rewritten files.
	translated map[string]string

	// Map from import path to package information.
	packages map[string]*types.Package

	// Map from import path to list of import paths that it imports.
	imports map[string][]string

	// Map from Object to AST function declaration for
	// parameterized functions.
	idToFunc map[types.Object]*ast.FuncDecl

	// Map from Object to AST type definition for parameterized types.
	idToTypeSpec map[types.Object]*ast.TypeSpec

	// Map from a Package to the instantiations we've created
	// for that package. This doesn't really belong here,
	// since it doesn't deal with import information,
	// but Importer is a useful common location to store the data.
	instantiations map[*types.Package]*instantiations

	// build tags
	tags map[string]bool
}

var _ types.ImporterFrom = &Importer{}

// NewImporter returns a new Importer.
// The tmpdir will become a GOPATH with translated files.
func NewImporter(tmpdir string) *Importer {
	info := &types.Info{
		Types: make(map[ast.Expr]types.TypeAndValue),
		Defs:  make(map[*ast.Ident]types.Object),
		Uses:  make(map[*ast.Ident]types.Object),
	}
	types.SetInferred(info, make(map[ast.Expr]types.Inferred))
	return &Importer{
		defaultImporter: importer.Default().(types.ImporterFrom),
		tmpdir:          tmpdir,
		info:            info,
		translated:      make(map[string]string),
		packages:        make(map[string]*types.Package),
		imports:         make(map[string][]string),
		idToFunc:        make(map[types.Object]*ast.FuncDecl),
		idToTypeSpec:    make(map[types.Object]*ast.TypeSpec),
		instantiations:  make(map[*types.Package]*instantiations),
		tags:            make(map[string]bool),
	}
}

func (imp *Importer) SetTags(tags []string) {
	for _, tag := range tags {
		imp.tags[tag] = true
	}
}

// Import should never be called. This is the old API; current code
// uses ImportFrom. This method still needs to be defined in order
// to implement the interface.
func (imp *Importer) Import(path string) (*types.Package, error) {
	log.Fatal("unexpected call to Import method")
	return nil, nil
}

// ImportFrom looks for a Go2 package, and if not found tries the
// default importer.
func (imp *Importer) ImportFrom(importPath, dir string, mode types.ImportMode) (*types.Package, error) {
	if build.IsLocalImport(importPath) {
		return imp.localImport(importPath, dir)
	}

	if imp.translated[importPath] != "" {
		tpkg, ok := imp.packages[importPath]
		if !ok {
			return nil, fmt.Errorf("circular import when processing %q", importPath)
		}
		return tpkg, nil
	}

	var pdir string
	if go2path := os.Getenv("GO2PATH"); go2path != "" {
		pdir = imp.findFromPath(go2path, importPath)
	}
	if pdir == "" {
		bpkg, err := build.Import(importPath, dir, build.FindOnly)
		if err != nil {
			return nil, err
		}
		pdir = bpkg.Dir
	}

	// If the directory holds .go2 files, we need to translate them.
	fdir, err := os.Open(pdir)
	if err != nil {
		return nil, err
	}
	defer fdir.Close()
	names, err := fdir.Readdirnames(-1)
	if err != nil {
		return nil, err
	}
	var gofiles, go2files []string
	for _, name := range names {
		switch filepath.Ext(name) {
		case ".go":
			gofiles = append(gofiles, name)
		case ".go2":
			go2files = append(go2files, name)
		}
	}

	if len(go2files) == 0 {
		return imp.importGo1Package(importPath, dir, mode, pdir, gofiles)
	}

	if len(gofiles) > 0 {
		for _, gofile := range gofiles {
			if err := checkGoFile(pdir, gofile); err != nil {
				return nil, err
			}
		}
	}

	tdir := filepath.Join(imp.tmpdir, "src", importPath)
	if err := os.MkdirAll(tdir, 0755); err != nil {
		return nil, err
	}
	for _, name := range go2files {
		data, err := ioutil.ReadFile(filepath.Join(pdir, name))
		if err != nil {
			return nil, err
		}
		if err := ioutil.WriteFile(filepath.Join(tdir, name), data, 0644); err != nil {
			return nil, err
		}
	}

	imp.translated[importPath] = tdir

	tpkgs, err := rewriteToPkgs(imp, importPath, tdir)
	if err != nil {
		return nil, err
	}

	switch len(tpkgs) {
	case 1:
		return tpkgs[0], nil
	case 2:
		if strings.HasSuffix(tpkgs[0].Name(), "_test") {
			return tpkgs[1], nil
		} else if strings.HasSuffix(tpkgs[1].Name(), "_test") {
			return tpkgs[0], nil
		}
	}

	return nil, fmt.Errorf("unexpected number of packages (%d) for %q (directory %q)", len(tpkgs), importPath, pdir)
}

// findFromPath looks for a directory under gopath.
func (imp *Importer) findFromPath(gopath, dir string) string {
	if filepath.IsAbs(dir) || build.IsLocalImport(dir) {
		return ""
	}
	for _, pd := range strings.Split(gopath, string(os.PathListSeparator)) {
		d := filepath.Join(pd, "src", dir)
		if fi, err := os.Stat(d); err == nil && fi.IsDir() {
			return d
		}
	}
	return ""
}

// importGo1Package handles importing a package with .go files rather
// than .go2 files. The default importer can do this if the package
// has been installed, but not otherwise. Installing the package using
// "go install" won't work if the Go 1 package depends on a Go 2 package.
// So use the default importer for a package in the standard library,
// and otherwise use go/types.
func (imp *Importer) importGo1Package(importPath, dir string, mode types.ImportMode, pdir string, gofiles []string) (*types.Package, error) {
	if goroot.IsStandardPackage(runtime.GOROOT(), "gc", importPath) {
		return imp.defaultImporter.ImportFrom(importPath, dir, mode)
	}

	if len(gofiles) == 0 {
		return nil, fmt.Errorf("importing %q: no Go files in %s", importPath, pdir)
	}

	fset := token.NewFileSet()
	filter := func(fi os.FileInfo) bool {
		name := fi.Name()
		if strings.HasSuffix(name, "_test.go") {
			return false
		}
		return imp.shouldInclude(path.Join(pdir, name))
	}
	pkgs, err := parser.ParseDir(fset, pdir, filter, 0)
	if err != nil {
		return nil, err
	}
	if len(pkgs) > 1 {
		return nil, fmt.Errorf("importing %q: multiple Go packages in %s", importPath, pdir)
	}

	var apkg *ast.Package
	for _, apkg = range pkgs {
		break
	}

	var asts []*ast.File
	for _, f := range apkg.Files {
		asts = append(asts, f)
	}
	sort.Slice(asts, func(i, j int) bool {
		return asts[i].Name.Name < asts[j].Name.Name
	})

	var merr multiErr
	conf := types.Config{
		Importer: imp,
		Error:    merr.add,
	}
	tpkg, err := conf.Check(importPath, fset, asts, imp.info)
	if err != nil {
		return nil, merr
	}

	return tpkg, nil
}

// installGo1Package runs "go install" to install a package.
// This is used for Go 1 packages, because the default
// importer looks at .a files, not sources.
// This is best effort; we don't report an error.
func (imp *Importer) installGo1Package(dir string) {
	gotool := filepath.Join(runtime.GOROOT(), "bin", "go")
	cmd := exec.Command(gotool, "install")
	cmd.Dir = dir
	cmd.Run()
}

// Register registers a package under an import path.
// This is for tests that use directives like //compiledir.
func (imp *Importer) Register(importPath string, tpkgs []*types.Package) error {
	switch len(tpkgs) {
	case 1:
		imp.packages[importPath] = tpkgs[0]
		return nil
	case 2:
		if strings.HasSuffix(tpkgs[0].Name(), "_test") {
			imp.packages[importPath] = tpkgs[1]
			return nil
		} else if strings.HasSuffix(tpkgs[1].Name(), "_test") {
			imp.packages[importPath] = tpkgs[0]
			return nil
		}
	}
	return fmt.Errorf("unexpected number of packages (%d) for %q", len(tpkgs), importPath)
}

// localImport handles a local import such as
//     import "./a"
// This is for tests that use directives like //compiledir.
func (imp *Importer) localImport(importPath, dir string) (*types.Package, error) {
	tpkg, ok := imp.packages[strings.TrimPrefix(importPath, "./")]
	if !ok {
		return nil, fmt.Errorf("cannot find local import %q", importPath)
	}
	return tpkg, nil
}

// record records information for a package, for use when working
// with packages that import this one.
func (imp *Importer) record(pkgName string, pkgfiles []namedAST, importPath string, tpkg *types.Package, asts []*ast.File) {
	if !strings.HasSuffix(pkgName, "_test") {
		if importPath != "" {
			imp.packages[importPath] = tpkg
		}
		imp.imports[importPath] = imp.collectImports(asts)
	}
	for _, nast := range pkgfiles {
		imp.addIDs(nast.ast)
	}
}

// collectImports returns all the imports paths imported by any of the ASTs.
func (imp *Importer) collectImports(asts []*ast.File) []string {
	m := make(map[string]bool)
	for _, a := range asts {
		for _, decl := range a.Decls {
			gen, ok := decl.(*ast.GenDecl)
			if !ok || gen.Tok != token.IMPORT {
				continue
			}
			for _, spec := range gen.Specs {
				imp := spec.(*ast.ImportSpec)
				if imp.Name != nil {
					// We don't try to handle import aliases.
					continue
				}
				path := strings.TrimPrefix(strings.TrimSuffix(imp.Path.Value, `"`), `"`)
				m[path] = true
			}
		}
	}
	s := make([]string, 0, len(m))
	for p := range m {
		s = append(s, p)
	}
	sort.Strings(s)
	return s
}

// addIDs finds IDs for generic functions and types and adds them to a map.
func (imp *Importer) addIDs(f *ast.File) {
	for _, decl := range f.Decls {
		switch decl := decl.(type) {
		case *ast.FuncDecl:
			if isParameterizedFuncDecl(decl, imp.info) {
				obj, ok := imp.info.Defs[decl.Name]
				if !ok {
					panic(fmt.Sprintf("no types.Object for %q", decl.Name.Name))
				}
				imp.idToFunc[obj] = decl
			}
		case *ast.GenDecl:
			if decl.Tok == token.TYPE {
				for _, s := range decl.Specs {
					ts := s.(*ast.TypeSpec)
					obj, ok := imp.info.Defs[ts.Name]
					if !ok {
						panic(fmt.Sprintf("no types.Object for %q", ts.Name.Name))
					}
					imp.idToTypeSpec[obj] = ts
				}
			}
		}
	}
}

// lookupPackage looks up a package by path.
func (imp *Importer) lookupPackage(path string) (*types.Package, bool) {
	pkg, ok := imp.packages[strings.TrimPrefix(path, "./")]
	return pkg, ok
}

// lookupFunc looks up a function by Object.
func (imp *Importer) lookupFunc(obj types.Object) (*ast.FuncDecl, bool) {
	decl, ok := imp.idToFunc[obj]
	return decl, ok
}

// lookupTypeSpec looks up a type by Object.
func (imp *Importer) lookupTypeSpec(obj types.Object) (*ast.TypeSpec, bool) {
	ts, ok := imp.idToTypeSpec[obj]
	return ts, ok
}

// transitiveImports returns all the transitive imports of an import path.
func (imp *Importer) transitiveImports(path string) []string {
	return imp.gatherTransitiveImports(path, make(map[string]bool))
}

// gatherTransitiveImports returns all the transitive imports of an import path,
// using a map to avoid duplicate work.
func (imp *Importer) gatherTransitiveImports(path string, m map[string]bool) []string {
	imports := imp.imports[path]
	if len(imports) == 0 {
		return nil
	}
	var r []string
	for _, im := range imports {
		r = append(r, im)
		if !m[im] {
			m[im] = true
			r = append(r, imp.gatherTransitiveImports(im, m)...)
		}
	}
	dup := make(map[string]bool)
	for _, p := range r {
		dup[p] = true
	}
	r = make([]string, 0, len(dup))
	for p := range dup {
		r = append(r, p)
	}
	sort.Strings(r)
	return r
}

var slashslash = []byte("//")

// shouldInclude reports whether it is okay to use this file,
// The rule is that in the file's leading run of // comments
// and blank lines, which must be followed by a blank line
// (to avoid including a Go package clause doc comment),
// lines beginning with '// +build' are taken as build directives.
//
// The file is accepted only if each such line lists something
// matching the file. For example:
//
//	// +build windows linux
//
// marks the file as applicable only on Windows and Linux.
//
// If tags["*"] is true, then ShouldBuild will consider every
// build tag except "ignore" to be both true and false for
// the purpose of satisfying build tags, in order to estimate
// (conservatively) whether a file could ever possibly be used
// in any build.
//
// This code was copied from the go command internals.
func (imp *Importer) shouldInclude(path string) bool {
	content, err := os.ReadFile(path)
	if err != nil {
		return false
	}

	// Pass 1. Identify leading run of // comments and blank lines,
	// which must be followed by a blank line.
	end := 0
	p := content
	for len(p) > 0 {
		line := p
		if i := bytes.IndexByte(line, '\n'); i >= 0 {
			line, p = line[:i], p[i+1:]
		} else {
			p = p[len(p):]
		}

		line = bytes.TrimSpace(line)
		if len(line) == 0 { // Blank line
			end = len(content) - len(p)
			continue
		}
		if !bytes.HasPrefix(line, slashslash) { // Not comment line
			break
		}
	}
	content = content[:end]

	// Pass 2.  Process each line in the run.
	p = content
	allok := true
	for len(p) > 0 {
		line := p
		if i := bytes.IndexByte(line, '\n'); i >= 0 {
			line, p = line[:i], p[i+1:]
		} else {
			p = p[len(p):]
		}
		line = bytes.TrimSpace(line)
		if !bytes.HasPrefix(line, slashslash) {
			continue
		}
		line = bytes.TrimSpace(line[len(slashslash):])
		if len(line) > 0 && line[0] == '+' {
			// Looks like a comment +line.
			f := strings.Fields(string(line))
			if f[0] == "+build" {
				ok := false
				for _, tok := range f[1:] {
					if matchTags(tok, imp.tags) {
						ok = true
					}
				}
				if !ok {
					allok = false
				}
			}
		}
	}

	return allok
}

// matchTags reports whether the name is one of:
//
//	tag (if tags[tag] is true)
//	!tag (if tags[tag] is false)
//	a comma-separated list of any of these
//
func matchTags(name string, tags map[string]bool) bool {
	if name == "" {
		return false
	}
	if i := strings.Index(name, ","); i >= 0 {
		// comma-separated list
		ok1 := matchTags(name[:i], tags)
		ok2 := matchTags(name[i+1:], tags)
		return ok1 && ok2
	}
	if strings.HasPrefix(name, "!!") { // bad syntax, reject always
		return false
	}
	if strings.HasPrefix(name, "!") { // negation
		return len(name) > 1 && matchTag(name[1:], tags, false)
	}
	return matchTag(name, tags, true)
}

// matchTag reports whether the tag name is valid and satisfied by tags[name]==want.
func matchTag(name string, tags map[string]bool, want bool) bool {
	// Tags must be letters, digits, underscores or dots.
	// Unlike in Go identifiers, all digits are fine (e.g., "386").
	for _, c := range name {
		if !unicode.IsLetter(c) && !unicode.IsDigit(c) && c != '_' && c != '.' {
			return false
		}
	}

	if tags["*"] && name != "" && name != "ignore" {
		// Special case for gathering all possible imports:
		// if we put * in the tags map then all tags
		// except "ignore" are considered both present and not
		// (so we return true no matter how 'want' is set).
		return true
	}

	have := tags[name]
	if name == "linux" {
		have = have || tags["android"]
	}
	if name == "solaris" {
		have = have || tags["illumos"]
	}
	if name == "darwin" {
		have = have || tags["ios"]
	}
	return have == want
}
