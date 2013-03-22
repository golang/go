// Copyright 2011 The Go Authors.  All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

package main

import (
	"bytes"
	"errors"
	"fmt"
	"go/build"
	"go/scanner"
	"go/token"
	"os"
	pathpkg "path"
	"path/filepath"
	"sort"
	"strings"
	"time"
	"unicode"
)

// A Package describes a single package found in a directory.
type Package struct {
	// Note: These fields are part of the go command's public API.
	// See list.go.  It is okay to add fields, but not to change or
	// remove existing ones.  Keep in sync with list.go
	Dir        string `json:",omitempty"` // directory containing package sources
	ImportPath string `json:",omitempty"` // import path of package in dir
	Name       string `json:",omitempty"` // package name
	Doc        string `json:",omitempty"` // package documentation string
	Target     string `json:",omitempty"` // install path
	Goroot     bool   `json:",omitempty"` // is this package found in the Go root?
	Standard   bool   `json:",omitempty"` // is this package part of the standard Go library?
	Stale      bool   `json:",omitempty"` // would 'go install' do anything for this package?
	Root       string `json:",omitempty"` // Go root or Go path dir containing this package

	// Source files
	GoFiles        []string `json:",omitempty"` // .go source files (excluding CgoFiles, TestGoFiles, XTestGoFiles)
	CgoFiles       []string `json:",omitempty"` // .go sources files that import "C"
	IgnoredGoFiles []string `json:",omitempty"` // .go sources ignored due to build constraints
	CFiles         []string `json:",omitempty"` // .c source files
	HFiles         []string `json:",omitempty"` // .h source files
	SFiles         []string `json:",omitempty"` // .s source files
	SysoFiles      []string `json:",omitempty"` // .syso system object files added to package
	SwigFiles      []string `json:",omitempty"` // .swig files
	SwigCXXFiles   []string `json:",omitempty"` // .swigcxx files

	// Cgo directives
	CgoCFLAGS    []string `json:",omitempty"` // cgo: flags for C compiler
	CgoLDFLAGS   []string `json:",omitempty"` // cgo: flags for linker
	CgoPkgConfig []string `json:",omitempty"` // cgo: pkg-config names

	// Dependency information
	Imports []string `json:",omitempty"` // import paths used by this package
	Deps    []string `json:",omitempty"` // all (recursively) imported dependencies

	// Error information
	Incomplete bool            `json:",omitempty"` // was there an error loading this package or dependencies?
	Error      *PackageError   `json:",omitempty"` // error loading this package (not dependencies)
	DepsErrors []*PackageError `json:",omitempty"` // errors loading dependencies

	// Test information
	TestGoFiles  []string `json:",omitempty"` // _test.go files in package
	TestImports  []string `json:",omitempty"` // imports from TestGoFiles
	XTestGoFiles []string `json:",omitempty"` // _test.go files outside package
	XTestImports []string `json:",omitempty"` // imports from XTestGoFiles

	// Unexported fields are not part of the public API.
	build        *build.Package
	pkgdir       string // overrides build.PkgDir
	imports      []*Package
	deps         []*Package
	gofiles      []string // GoFiles+CgoFiles+TestGoFiles+XTestGoFiles files, absolute paths
	sfiles       []string
	allgofiles   []string // gofiles + IgnoredGoFiles, absolute paths
	target       string   // installed file for this package (may be executable)
	fake         bool     // synthesized package
	forceBuild   bool     // this package must be rebuilt
	forceLibrary bool     // this package is a library (even if named "main")
	local        bool     // imported via local path (./ or ../)
	localPrefix  string   // interpret ./ and ../ imports relative to this prefix
	exeName      string   // desired name for temporary executable
}

func (p *Package) copyBuild(pp *build.Package) {
	p.build = pp

	p.Dir = pp.Dir
	p.ImportPath = pp.ImportPath
	p.Name = pp.Name
	p.Doc = pp.Doc
	p.Root = pp.Root
	// TODO? Target
	p.Goroot = pp.Goroot
	p.Standard = p.Goroot && p.ImportPath != "" && !strings.Contains(p.ImportPath, ".")
	p.GoFiles = pp.GoFiles
	p.CgoFiles = pp.CgoFiles
	p.IgnoredGoFiles = pp.IgnoredGoFiles
	p.CFiles = pp.CFiles
	p.HFiles = pp.HFiles
	p.SFiles = pp.SFiles
	p.SysoFiles = pp.SysoFiles
	p.SwigFiles = pp.SwigFiles
	p.SwigCXXFiles = pp.SwigCXXFiles
	p.CgoCFLAGS = pp.CgoCFLAGS
	p.CgoLDFLAGS = pp.CgoLDFLAGS
	p.CgoPkgConfig = pp.CgoPkgConfig
	p.Imports = pp.Imports
	p.TestGoFiles = pp.TestGoFiles
	p.TestImports = pp.TestImports
	p.XTestGoFiles = pp.XTestGoFiles
	p.XTestImports = pp.XTestImports
}

// A PackageError describes an error loading information about a package.
type PackageError struct {
	ImportStack []string // shortest path from package named on command line to this one
	Pos         string   // position of error
	Err         string   // the error itself
}

func (p *PackageError) Error() string {
	if p.Pos != "" {
		// Omit import stack.  The full path to the file where the error
		// is the most important thing.
		return p.Pos + ": " + p.Err
	}
	if len(p.ImportStack) == 0 {
		return p.Err
	}
	return "package " + strings.Join(p.ImportStack, "\n\timports ") + ": " + p.Err
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

// dirToImportPath returns the pseudo-import path we use for a package
// outside the Go path.  It begins with _/ and then contains the full path
// to the directory.  If the package lives in c:\home\gopher\my\pkg then
// the pseudo-import path is _/c_/home/gopher/my/pkg.
// Using a pseudo-import path like this makes the ./ imports no longer
// a special case, so that all the code to deal with ordinary imports works
// automatically.
func dirToImportPath(dir string) string {
	return pathpkg.Join("_", strings.Map(makeImportValid, filepath.ToSlash(dir)))
}

func makeImportValid(r rune) rune {
	// Should match Go spec, compilers, and ../../pkg/go/parser/parser.go:/isValidImport.
	const illegalChars = `!"#$%&'()*,:;<=>?[\]^{|}` + "`\uFFFD"
	if !unicode.IsGraphic(r) || unicode.IsSpace(r) || strings.ContainsRune(illegalChars, r) {
		return '_'
	}
	return r
}

// loadImport scans the directory named by path, which must be an import path,
// but possibly a local import path (an absolute file system path or one beginning
// with ./ or ../).  A local relative path is interpreted relative to srcDir.
// It returns a *Package describing the package found in that directory.
func loadImport(path string, srcDir string, stk *importStack, importPos []token.Position) *Package {
	stk.push(path)
	defer stk.pop()

	// Determine canonical identifier for this package.
	// For a local import the identifier is the pseudo-import path
	// we create from the full directory to the package.
	// Otherwise it is the usual import path.
	importPath := path
	isLocal := build.IsLocalImport(path)
	if isLocal {
		importPath = dirToImportPath(filepath.Join(srcDir, path))
	}
	if p := packageCache[importPath]; p != nil {
		return reusePackage(p, stk)
	}

	p := new(Package)
	p.local = isLocal
	p.ImportPath = importPath
	packageCache[importPath] = p

	// Load package.
	// Import always returns bp != nil, even if an error occurs,
	// in order to return partial information.
	//
	// TODO: After Go 1, decide when to pass build.AllowBinary here.
	// See issue 3268 for mistakes to avoid.
	bp, err := buildContext.Import(path, srcDir, 0)
	bp.ImportPath = importPath
	if gobin != "" {
		bp.BinDir = gobin
	}
	p.load(stk, bp, err)
	if p.Error != nil && len(importPos) > 0 {
		pos := importPos[0]
		pos.Filename = shortPath(pos.Filename)
		p.Error.Pos = pos.String()
	}

	return p
}

// reusePackage reuses package p to satisfy the import at the top
// of the import stack stk.  If this use causes an import loop,
// reusePackage updates p's error information to record the loop.
func reusePackage(p *Package, stk *importStack) *Package {
	// We use p.imports==nil to detect a package that
	// is in the midst of its own loadPackage call
	// (all the recursion below happens before p.imports gets set).
	if p.imports == nil {
		if p.Error == nil {
			p.Error = &PackageError{
				ImportStack: stk.copy(),
				Err:         "import cycle not allowed",
			}
		}
		p.Incomplete = true
	}
	if p.Error != nil && stk.shorterThan(p.Error.ImportStack) {
		p.Error.ImportStack = stk.copy()
	}
	return p
}

// isGoTool is the list of directories for Go programs that are installed in
// $GOROOT/pkg/tool.
var isGoTool = map[string]bool{
	"cmd/api":  true,
	"cmd/cgo":  true,
	"cmd/fix":  true,
	"cmd/vet":  true,
	"cmd/yacc": true,
}

// expandScanner expands a scanner.List error into all the errors in the list.
// The default Error method only shows the first error.
func expandScanner(err error) error {
	// Look for parser errors.
	if err, ok := err.(scanner.ErrorList); ok {
		// Prepare error with \n before each message.
		// When printed in something like context: %v
		// this will put the leading file positions each on
		// its own line.  It will also show all the errors
		// instead of just the first, as err.Error does.
		var buf bytes.Buffer
		for _, e := range err {
			e.Pos.Filename = shortPath(e.Pos.Filename)
			buf.WriteString("\n")
			buf.WriteString(e.Error())
		}
		return errors.New(buf.String())
	}
	return err
}

// load populates p using information from bp, err, which should
// be the result of calling build.Context.Import.
func (p *Package) load(stk *importStack, bp *build.Package, err error) *Package {
	p.copyBuild(bp)

	// The localPrefix is the path we interpret ./ imports relative to.
	// Synthesized main packages sometimes override this.
	p.localPrefix = dirToImportPath(p.Dir)

	if err != nil {
		p.Incomplete = true
		err = expandScanner(err)
		p.Error = &PackageError{
			ImportStack: stk.copy(),
			Err:         err.Error(),
		}
		return p
	}

	if p.Name == "main" {
		_, elem := filepath.Split(p.Dir)
		full := buildContext.GOOS + "_" + buildContext.GOARCH + "/" + elem
		if buildContext.GOOS != toolGOOS || buildContext.GOARCH != toolGOARCH {
			// Install cross-compiled binaries to subdirectories of bin.
			elem = full
		}
		if p.build.BinDir != "" {
			p.target = filepath.Join(p.build.BinDir, elem)
		}
		if p.Goroot && (isGoTool[p.ImportPath] || strings.HasPrefix(p.ImportPath, "exp/")) {
			p.target = filepath.Join(gorootPkg, "tool", full)
		}
		if p.target != "" && buildContext.GOOS == "windows" {
			p.target += ".exe"
		}
	} else if p.local {
		// Local import turned into absolute path.
		// No permanent install target.
		p.target = ""
	} else {
		p.target = p.build.PkgObj
	}

	importPaths := p.Imports
	// Packages that use cgo import runtime/cgo implicitly,
	// except runtime/cgo itself.
	if len(p.CgoFiles) > 0 && (!p.Standard || p.ImportPath != "runtime/cgo") {
		importPaths = append(importPaths, "runtime/cgo")
	}
	// Everything depends on runtime, except runtime and unsafe.
	if !p.Standard || (p.ImportPath != "runtime" && p.ImportPath != "unsafe") {
		importPaths = append(importPaths, "runtime")
		// When race detection enabled everything depends on runtime/race.
		// Exclude runtime/cgo and cmd/cgo to avoid circular dependencies.
		if buildRace && (!p.Standard || (p.ImportPath != "runtime/race" && p.ImportPath != "runtime/cgo" && p.ImportPath != "cmd/cgo")) {
			importPaths = append(importPaths, "runtime/race")
		}
	}

	// Build list of full paths to all Go files in the package,
	// for use by commands like go fmt.
	p.gofiles = stringList(p.GoFiles, p.CgoFiles, p.TestGoFiles, p.XTestGoFiles)
	for i := range p.gofiles {
		p.gofiles[i] = filepath.Join(p.Dir, p.gofiles[i])
	}
	sort.Strings(p.gofiles)

	p.sfiles = stringList(p.SFiles)
	for i := range p.sfiles {
		p.sfiles[i] = filepath.Join(p.Dir, p.sfiles[i])
	}
	sort.Strings(p.sfiles)

	p.allgofiles = stringList(p.IgnoredGoFiles)
	for i := range p.allgofiles {
		p.allgofiles[i] = filepath.Join(p.Dir, p.allgofiles[i])
	}
	p.allgofiles = append(p.allgofiles, p.gofiles...)
	sort.Strings(p.allgofiles)

	// Check for case-insensitive collision of input files.
	// To avoid problems on case-insensitive files, we reject any package
	// where two different input files have equal names under a case-insensitive
	// comparison.
	f1, f2 := foldDup(stringList(
		p.GoFiles,
		p.CgoFiles,
		p.IgnoredGoFiles,
		p.CFiles,
		p.HFiles,
		p.SFiles,
		p.SysoFiles,
		p.SwigFiles,
		p.SwigCXXFiles,
		p.TestGoFiles,
		p.XTestGoFiles,
	))
	if f1 != "" {
		p.Error = &PackageError{
			ImportStack: stk.copy(),
			Err:         fmt.Sprintf("case-insensitive file name collision: %q and %q", f1, f2),
		}
		return p
	}

	// Build list of imported packages and full dependency list.
	imports := make([]*Package, 0, len(p.Imports))
	deps := make(map[string]bool)
	for i, path := range importPaths {
		if path == "C" {
			continue
		}
		p1 := loadImport(path, p.Dir, stk, p.build.ImportPos[path])
		if p1.local {
			if !p.local && p.Error == nil {
				p.Error = &PackageError{
					ImportStack: stk.copy(),
					Err:         fmt.Sprintf("local import %q in non-local package", path),
				}
				pos := p.build.ImportPos[path]
				if len(pos) > 0 {
					p.Error.Pos = pos[0].String()
				}
			}
			path = p1.ImportPath
			importPaths[i] = path
		}
		deps[path] = true
		imports = append(imports, p1)
		for _, dep := range p1.Deps {
			deps[dep] = true
		}
		if p1.Incomplete {
			p.Incomplete = true
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

	// unsafe is a fake package.
	if p.Standard && (p.ImportPath == "unsafe" || buildContext.Compiler == "gccgo") {
		p.target = ""
	}
	p.Target = p.target

	// In the absence of errors lower in the dependency tree,
	// check for case-insensitive collisions of import paths.
	if len(p.DepsErrors) == 0 {
		dep1, dep2 := foldDup(p.Deps)
		if dep1 != "" {
			p.Error = &PackageError{
				ImportStack: stk.copy(),
				Err:         fmt.Sprintf("case-insensitive import collision: %q and %q", dep1, dep2),
			}
			return p
		}
	}

	return p
}

// usesSwig returns whether the package needs to run SWIG.
func (p *Package) usesSwig() bool {
	return len(p.SwigFiles) > 0 || len(p.SwigCXXFiles) > 0
}

// swigSoname returns the name of the shared library we create for a
// SWIG input file.
func (p *Package) swigSoname(file string) string {
	return strings.Replace(p.ImportPath, "/", "-", -1) + "-" + strings.Replace(file, ".", "-", -1) + ".so"
}

// swigDir returns the name of the shared SWIG directory for a
// package.
func (p *Package) swigDir(ctxt *build.Context) string {
	dir := p.build.PkgRoot
	if ctxt.Compiler == "gccgo" {
		dir = filepath.Join(dir, "gccgo_"+ctxt.GOOS+"_"+ctxt.GOARCH)
	} else {
		dir = filepath.Join(dir, ctxt.GOOS+"_"+ctxt.GOARCH)
	}
	return filepath.Join(dir, "swig")
}

// packageList returns the list of packages in the dag rooted at roots
// as visited in a depth-first post-order traversal.
func packageList(roots []*Package) []*Package {
	seen := map[*Package]bool{}
	all := []*Package{}
	var walk func(*Package)
	walk = func(p *Package) {
		if seen[p] {
			return
		}
		seen[p] = true
		for _, p1 := range p.imports {
			walk(p1)
		}
		all = append(all, p)
	}
	for _, root := range roots {
		walk(root)
	}
	return all
}

// computeStale computes the Stale flag in the package dag that starts
// at the named pkgs (command-line arguments).
func computeStale(pkgs ...*Package) {
	topRoot := map[string]bool{}
	for _, p := range pkgs {
		topRoot[p.Root] = true
	}

	for _, p := range packageList(pkgs) {
		p.Stale = isStale(p, topRoot)
	}
}

// isStale reports whether package p needs to be rebuilt.
func isStale(p *Package, topRoot map[string]bool) bool {
	if p.Standard && (p.ImportPath == "unsafe" || buildContext.Compiler == "gccgo") {
		// fake, builtin package
		return false
	}
	if p.Error != nil {
		return true
	}

	// A package without Go sources means we only found
	// the installed .a file.  Since we don't know how to rebuild
	// it, it can't be stale, even if -a is set.  This enables binary-only
	// distributions of Go packages, although such binaries are
	// only useful with the specific version of the toolchain that
	// created them.
	if len(p.gofiles) == 0 && !p.usesSwig() {
		return false
	}

	if buildA || p.target == "" || p.Stale {
		return true
	}

	// Package is stale if completely unbuilt.
	var built time.Time
	if fi, err := os.Stat(p.target); err == nil {
		built = fi.ModTime()
	}
	if built.IsZero() {
		return true
	}

	olderThan := func(file string) bool {
		fi, err := os.Stat(file)
		return err != nil || fi.ModTime().After(built)
	}

	// Package is stale if a dependency is, or if a dependency is newer.
	for _, p1 := range p.deps {
		if p1.Stale || p1.target != "" && olderThan(p1.target) {
			return true
		}
	}

	// As a courtesy to developers installing new versions of the compiler
	// frequently, define that packages are stale if they are
	// older than the compiler, and commands if they are older than
	// the linker.  This heuristic will not work if the binaries are
	// back-dated, as some binary distributions may do, but it does handle
	// a very common case.
	// See issue 3036.
	// Assume code in $GOROOT is up to date, since it may not be writeable.
	// See issue 4106.
	if p.Root != goroot {
		if olderThan(buildToolchain.compiler()) {
			return true
		}
		if p.build.IsCommand() && olderThan(buildToolchain.linker()) {
			return true
		}
	}

	// Have installed copy, probably built using current compilers,
	// and built after its imported packages.  The only reason now
	// that we'd have to rebuild it is if the sources were newer than
	// the package.   If a package p is not in the same tree as any
	// package named on the command-line, assume it is up-to-date
	// no matter what the modification times on the source files indicate.
	// This avoids rebuilding $GOROOT packages when people are
	// working outside the Go root, and it effectively makes each tree
	// listed in $GOPATH a separate compilation world.
	// See issue 3149.
	if p.Root != "" && !topRoot[p.Root] {
		return false
	}

	srcs := stringList(p.GoFiles, p.CFiles, p.HFiles, p.SFiles, p.CgoFiles, p.SysoFiles)
	for _, src := range srcs {
		if olderThan(filepath.Join(p.Dir, src)) {
			return true
		}
	}

	for _, src := range stringList(p.SwigFiles, p.SwigCXXFiles) {
		if olderThan(filepath.Join(p.Dir, src)) {
			return true
		}
		soname := p.swigSoname(src)
		fi, err := os.Stat(soname)
		if err != nil {
			return true
		}
		fiSrc, err := os.Stat(src)
		if err != nil || fiSrc.ModTime().After(fi.ModTime()) {
			return true
		}
	}

	return false
}

var cwd, _ = os.Getwd()

var cmdCache = map[string]*Package{}

// loadPackage is like loadImport but is used for command-line arguments,
// not for paths found in import statements.  In addition to ordinary import paths,
// loadPackage accepts pseudo-paths beginning with cmd/ to denote commands
// in the Go command directory, as well as paths to those directories.
func loadPackage(arg string, stk *importStack) *Package {
	if build.IsLocalImport(arg) {
		dir := arg
		if !filepath.IsAbs(dir) {
			if abs, err := filepath.Abs(dir); err == nil {
				// interpret relative to current directory
				dir = abs
			}
		}
		if sub, ok := hasSubdir(gorootSrc, dir); ok && strings.HasPrefix(sub, "cmd/") && !strings.Contains(sub[4:], "/") {
			arg = sub
		}
	}
	if strings.HasPrefix(arg, "cmd/") {
		if p := cmdCache[arg]; p != nil {
			return p
		}
		stk.push(arg)
		defer stk.pop()

		if strings.Contains(arg[4:], "/") {
			p := &Package{
				Error: &PackageError{
					ImportStack: stk.copy(),
					Err:         fmt.Sprintf("invalid import path: cmd/... is reserved for Go commands"),
				},
			}
			return p
		}

		bp, err := buildContext.ImportDir(filepath.Join(gorootSrc, arg), 0)
		bp.ImportPath = arg
		bp.Goroot = true
		bp.BinDir = gorootBin
		if gobin != "" {
			bp.BinDir = gobin
		}
		bp.Root = goroot
		bp.SrcRoot = gorootSrc
		p := new(Package)
		cmdCache[arg] = p
		p.load(stk, bp, err)
		if p.Error == nil && p.Name != "main" {
			p.Error = &PackageError{
				ImportStack: stk.copy(),
				Err:         fmt.Sprintf("expected package main but found package %s in %s", p.Name, p.Dir),
			}
		}
		return p
	}

	// Wasn't a command; must be a package.
	// If it is a local import path but names a standard package,
	// we treat it as if the user specified the standard package.
	// This lets you run go test ./ioutil in package io and be
	// referring to io/ioutil rather than a hypothetical import of
	// "./ioutil".
	if build.IsLocalImport(arg) {
		bp, _ := buildContext.ImportDir(filepath.Join(cwd, arg), build.FindOnly)
		if bp.ImportPath != "" && bp.ImportPath != "." {
			arg = bp.ImportPath
		}
	}

	return loadImport(arg, cwd, stk, nil)
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
	var pkgs []*Package
	for _, pkg := range packagesAndErrors(args) {
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
	if len(args) > 0 && strings.HasSuffix(args[0], ".go") {
		return []*Package{goFilesPackage(args)}
	}

	args = importPaths(args)
	var pkgs []*Package
	var stk importStack
	var set = make(map[string]bool)

	for _, arg := range args {
		if !set[arg] {
			pkgs = append(pkgs, loadPackage(arg, &stk))
			set[arg] = true
		}
	}
	computeStale(pkgs...)

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

// hasSubdir reports whether dir is a subdirectory of
// (possibly multiple levels below) root.
// If so, it sets rel to the path fragment that must be
// appended to root to reach dir.
func hasSubdir(root, dir string) (rel string, ok bool) {
	if p, err := filepath.EvalSymlinks(root); err == nil {
		root = p
	}
	if p, err := filepath.EvalSymlinks(dir); err == nil {
		dir = p
	}
	const sep = string(filepath.Separator)
	root = filepath.Clean(root)
	if !strings.HasSuffix(root, sep) {
		root += sep
	}
	dir = filepath.Clean(dir)
	if !strings.HasPrefix(dir, root) {
		return "", false
	}
	return filepath.ToSlash(dir[len(root):]), true
}
