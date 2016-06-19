// Copyright 2011 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

package main

import (
	"bytes"
	"crypto/sha1"
	"errors"
	"fmt"
	"go/build"
	"go/scanner"
	"go/token"
	"io"
	"io/ioutil"
	"os"
	pathpkg "path"
	"path/filepath"
	"runtime"
	"sort"
	"strconv"
	"strings"
	"unicode"
)

// A Package describes a single package found in a directory.
type Package struct {
	// Note: These fields are part of the go command's public API.
	// See list.go. It is okay to add fields, but not to change or
	// remove existing ones. Keep in sync with list.go
	Dir           string `json:",omitempty"` // directory containing package sources
	ImportPath    string `json:",omitempty"` // import path of package in dir
	ImportComment string `json:",omitempty"` // path in import comment on package statement
	Name          string `json:",omitempty"` // package name
	Doc           string `json:",omitempty"` // package documentation string
	Target        string `json:",omitempty"` // install path
	Shlib         string `json:",omitempty"` // the shared library that contains this package (only set when -linkshared)
	Goroot        bool   `json:",omitempty"` // is this package found in the Go root?
	Standard      bool   `json:",omitempty"` // is this package part of the standard Go library?
	Stale         bool   `json:",omitempty"` // would 'go install' do anything for this package?
	StaleReason   string `json:",omitempty"` // why is Stale true?
	Root          string `json:",omitempty"` // Go root or Go path dir containing this package
	ConflictDir   string `json:",omitempty"` // Dir is hidden by this other directory
	BinaryOnly    bool   `json:",omitempty"` // package cannot be recompiled

	// Source files
	GoFiles        []string `json:",omitempty"` // .go source files (excluding CgoFiles, TestGoFiles, XTestGoFiles)
	CgoFiles       []string `json:",omitempty"` // .go sources files that import "C"
	IgnoredGoFiles []string `json:",omitempty"` // .go sources ignored due to build constraints
	CFiles         []string `json:",omitempty"` // .c source files
	CXXFiles       []string `json:",omitempty"` // .cc, .cpp and .cxx source files
	MFiles         []string `json:",omitempty"` // .m source files
	HFiles         []string `json:",omitempty"` // .h, .hh, .hpp and .hxx source files
	FFiles         []string `json:",omitempty"` // .f, .F, .for and .f90 Fortran source files
	SFiles         []string `json:",omitempty"` // .s source files
	SwigFiles      []string `json:",omitempty"` // .swig files
	SwigCXXFiles   []string `json:",omitempty"` // .swigcxx files
	SysoFiles      []string `json:",omitempty"` // .syso system object files added to package

	// Cgo directives
	CgoCFLAGS    []string `json:",omitempty"` // cgo: flags for C compiler
	CgoCPPFLAGS  []string `json:",omitempty"` // cgo: flags for C preprocessor
	CgoCXXFLAGS  []string `json:",omitempty"` // cgo: flags for C++ compiler
	CgoFFLAGS    []string `json:",omitempty"` // cgo: flags for Fortran compiler
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
	allgofiles   []string             // gofiles + IgnoredGoFiles, absolute paths
	target       string               // installed file for this package (may be executable)
	fake         bool                 // synthesized package
	external     bool                 // synthesized external test package
	forceLibrary bool                 // this package is a library (even if named "main")
	cmdline      bool                 // defined by files listed on command line
	local        bool                 // imported via local path (./ or ../)
	localPrefix  string               // interpret ./ and ../ imports relative to this prefix
	exeName      string               // desired name for temporary executable
	coverMode    string               // preprocess Go source files with the coverage tool in this mode
	coverVars    map[string]*CoverVar // variables created by coverage analysis
	omitDWARF    bool                 // tell linker not to write DWARF information
	buildID      string               // expected build ID for generated package
	gobinSubdir  bool                 // install target would be subdir of GOBIN
}

// vendored returns the vendor-resolved version of imports,
// which should be p.TestImports or p.XTestImports, NOT p.Imports.
// The imports in p.TestImports and p.XTestImports are not recursively
// loaded during the initial load of p, so they list the imports found in
// the source file, but most processing should be over the vendor-resolved
// import paths. We do this resolution lazily both to avoid file system work
// and because the eventual real load of the test imports (during 'go test')
// can produce better error messages if it starts with the original paths.
// The initial load of p loads all the non-test imports and rewrites
// the vendored paths, so nothing should ever call p.vendored(p.Imports).
func (p *Package) vendored(imports []string) []string {
	if len(imports) > 0 && len(p.Imports) > 0 && &imports[0] == &p.Imports[0] {
		panic("internal error: p.vendored(p.Imports) called")
	}
	seen := make(map[string]bool)
	var all []string
	for _, path := range imports {
		path = vendoredImportPath(p, path)
		if !seen[path] {
			seen[path] = true
			all = append(all, path)
		}
	}
	sort.Strings(all)
	return all
}

// CoverVar holds the name of the generated coverage variables targeting the named file.
type CoverVar struct {
	File string // local file name
	Var  string // name of count struct
}

func (p *Package) copyBuild(pp *build.Package) {
	p.build = pp

	if pp.PkgTargetRoot != "" && buildPkgdir != "" {
		old := pp.PkgTargetRoot
		pp.PkgRoot = buildPkgdir
		pp.PkgTargetRoot = buildPkgdir
		pp.PkgObj = filepath.Join(buildPkgdir, strings.TrimPrefix(pp.PkgObj, old))
	}

	p.Dir = pp.Dir
	p.ImportPath = pp.ImportPath
	p.ImportComment = pp.ImportComment
	p.Name = pp.Name
	p.Doc = pp.Doc
	p.Root = pp.Root
	p.ConflictDir = pp.ConflictDir
	p.BinaryOnly = pp.BinaryOnly

	// TODO? Target
	p.Goroot = pp.Goroot
	p.Standard = p.Goroot && p.ImportPath != "" && isStandardImportPath(p.ImportPath)
	p.GoFiles = pp.GoFiles
	p.CgoFiles = pp.CgoFiles
	p.IgnoredGoFiles = pp.IgnoredGoFiles
	p.CFiles = pp.CFiles
	p.CXXFiles = pp.CXXFiles
	p.MFiles = pp.MFiles
	p.HFiles = pp.HFiles
	p.FFiles = pp.FFiles
	p.SFiles = pp.SFiles
	p.SwigFiles = pp.SwigFiles
	p.SwigCXXFiles = pp.SwigCXXFiles
	p.SysoFiles = pp.SysoFiles
	p.CgoCFLAGS = pp.CgoCFLAGS
	p.CgoCPPFLAGS = pp.CgoCPPFLAGS
	p.CgoCXXFLAGS = pp.CgoCXXFLAGS
	p.CgoLDFLAGS = pp.CgoLDFLAGS
	p.CgoPkgConfig = pp.CgoPkgConfig
	p.Imports = pp.Imports
	p.TestGoFiles = pp.TestGoFiles
	p.TestImports = pp.TestImports
	p.XTestGoFiles = pp.XTestGoFiles
	p.XTestImports = pp.XTestImports
}

// isStandardImportPath reports whether $GOROOT/src/path should be considered
// part of the standard distribution. For historical reasons we allow people to add
// their own code to $GOROOT instead of using $GOPATH, but we assume that
// code will start with a domain name (dot in the first element).
func isStandardImportPath(path string) bool {
	i := strings.Index(path, "/")
	if i < 0 {
		i = len(path)
	}
	elem := path[:i]
	return !strings.Contains(elem, ".")
}

// A PackageError describes an error loading information about a package.
type PackageError struct {
	ImportStack   []string // shortest path from package named on command line to this one
	Pos           string   // position of error
	Err           string   // the error itself
	isImportCycle bool     // the error is an import cycle
	hard          bool     // whether the error is soft or hard; soft errors are ignored in some places
}

func (p *PackageError) Error() string {
	// Import cycles deserve special treatment.
	if p.isImportCycle {
		return fmt.Sprintf("%s\npackage %s\n", p.Err, strings.Join(p.ImportStack, "\n\timports "))
	}
	if p.Pos != "" {
		// Omit import stack. The full path to the file where the error
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

// shorterThan reports whether sp is shorter than t.
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
// outside the Go path. It begins with _/ and then contains the full path
// to the directory. If the package lives in c:\home\gopher\my\pkg then
// the pseudo-import path is _/c_/home/gopher/my/pkg.
// Using a pseudo-import path like this makes the ./ imports no longer
// a special case, so that all the code to deal with ordinary imports works
// automatically.
func dirToImportPath(dir string) string {
	return pathpkg.Join("_", strings.Map(makeImportValid, filepath.ToSlash(dir)))
}

func makeImportValid(r rune) rune {
	// Should match Go spec, compilers, and ../../go/parser/parser.go:/isValidImport.
	const illegalChars = `!"#$%&'()*,:;<=>?[\]^{|}` + "`\uFFFD"
	if !unicode.IsGraphic(r) || unicode.IsSpace(r) || strings.ContainsRune(illegalChars, r) {
		return '_'
	}
	return r
}

// Mode flags for loadImport and download (in get.go).
const (
	// useVendor means that loadImport should do vendor expansion
	// (provided the vendoring experiment is enabled).
	// That is, useVendor means that the import path came from
	// a source file and has not been vendor-expanded yet.
	// Every import path should be loaded initially with useVendor,
	// and then the expanded version (with the /vendor/ in it) gets
	// recorded as the canonical import path. At that point, future loads
	// of that package must not pass useVendor, because
	// disallowVendor will reject direct use of paths containing /vendor/.
	useVendor = 1 << iota

	// getTestDeps is for download (part of "go get") and indicates
	// that test dependencies should be fetched too.
	getTestDeps
)

// loadImport scans the directory named by path, which must be an import path,
// but possibly a local import path (an absolute file system path or one beginning
// with ./ or ../).  A local relative path is interpreted relative to srcDir.
// It returns a *Package describing the package found in that directory.
func loadImport(path, srcDir string, parent *Package, stk *importStack, importPos []token.Position, mode int) *Package {
	stk.push(path)
	defer stk.pop()

	// Determine canonical identifier for this package.
	// For a local import the identifier is the pseudo-import path
	// we create from the full directory to the package.
	// Otherwise it is the usual import path.
	// For vendored imports, it is the expanded form.
	importPath := path
	origPath := path
	isLocal := build.IsLocalImport(path)
	if isLocal {
		importPath = dirToImportPath(filepath.Join(srcDir, path))
	} else if mode&useVendor != 0 {
		// We do our own vendor resolution, because we want to
		// find out the key to use in packageCache without the
		// overhead of repeated calls to buildContext.Import.
		// The code is also needed in a few other places anyway.
		path = vendoredImportPath(parent, path)
		importPath = path
	}

	if p := packageCache[importPath]; p != nil {
		if perr := disallowInternal(srcDir, p, stk); perr != p {
			return perr
		}
		if mode&useVendor != 0 {
			if perr := disallowVendor(srcDir, origPath, p, stk); perr != p {
				return perr
			}
		}
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
	buildMode := build.ImportComment
	if mode&useVendor == 0 || path != origPath {
		// Not vendoring, or we already found the vendored path.
		buildMode |= build.IgnoreVendor
	}
	bp, err := buildContext.Import(path, srcDir, buildMode)
	bp.ImportPath = importPath
	if gobin != "" {
		bp.BinDir = gobin
	}
	if err == nil && !isLocal && bp.ImportComment != "" && bp.ImportComment != path &&
		!strings.Contains(path, "/vendor/") && !strings.HasPrefix(path, "vendor/") {
		err = fmt.Errorf("code in directory %s expects import %q", bp.Dir, bp.ImportComment)
	}
	p.load(stk, bp, err)
	if p.Error != nil && p.Error.Pos == "" && len(importPos) > 0 {
		pos := importPos[0]
		pos.Filename = shortPath(pos.Filename)
		p.Error.Pos = pos.String()
	}

	if perr := disallowInternal(srcDir, p, stk); perr != p {
		return perr
	}
	if mode&useVendor != 0 {
		if perr := disallowVendor(srcDir, origPath, p, stk); perr != p {
			return perr
		}
	}

	return p
}

var isDirCache = map[string]bool{}

func isDir(path string) bool {
	result, ok := isDirCache[path]
	if ok {
		return result
	}

	fi, err := os.Stat(path)
	result = err == nil && fi.IsDir()
	isDirCache[path] = result
	return result
}

// vendoredImportPath returns the expansion of path when it appears in parent.
// If parent is x/y/z, then path might expand to x/y/z/vendor/path, x/y/vendor/path,
// x/vendor/path, vendor/path, or else stay path if none of those exist.
// vendoredImportPath returns the expanded path or, if no expansion is found, the original.
func vendoredImportPath(parent *Package, path string) (found string) {
	if parent == nil || parent.Root == "" {
		return path
	}

	dir := filepath.Clean(parent.Dir)
	root := filepath.Join(parent.Root, "src")
	if !hasFilePathPrefix(dir, root) {
		// Look for symlinks before reporting error.
		dir = expandPath(dir)
		root = expandPath(root)
	}
	if !hasFilePathPrefix(dir, root) || len(dir) <= len(root) || dir[len(root)] != filepath.Separator {
		fatalf("invalid vendoredImportPath: dir=%q root=%q separator=%q", dir, root, string(filepath.Separator))
	}

	vpath := "vendor/" + path
	for i := len(dir); i >= len(root); i-- {
		if i < len(dir) && dir[i] != filepath.Separator {
			continue
		}
		// Note: checking for the vendor directory before checking
		// for the vendor/path directory helps us hit the
		// isDir cache more often. It also helps us prepare a more useful
		// list of places we looked, to report when an import is not found.
		if !isDir(filepath.Join(dir[:i], "vendor")) {
			continue
		}
		targ := filepath.Join(dir[:i], vpath)
		if isDir(targ) && hasGoFiles(targ) {
			importPath := parent.ImportPath
			if importPath == "command-line-arguments" {
				// If parent.ImportPath is 'command-line-arguments'.
				// set to relative directory to root (also chopped root directory)
				importPath = dir[len(root)+1:]
			}
			// We started with parent's dir c:\gopath\src\foo\bar\baz\quux\xyzzy.
			// We know the import path for parent's dir.
			// We chopped off some number of path elements and
			// added vendor\path to produce c:\gopath\src\foo\bar\baz\vendor\path.
			// Now we want to know the import path for that directory.
			// Construct it by chopping the same number of path elements
			// (actually the same number of bytes) from parent's import path
			// and then append /vendor/path.
			chopped := len(dir) - i
			if chopped == len(importPath)+1 {
				// We walked up from c:\gopath\src\foo\bar
				// and found c:\gopath\src\vendor\path.
				// We chopped \foo\bar (length 8) but the import path is "foo/bar" (length 7).
				// Use "vendor/path" without any prefix.
				return vpath
			}
			return importPath[:len(importPath)-chopped] + "/" + vpath
		}
	}
	return path
}

// hasGoFiles reports whether dir contains any files with names ending in .go.
// For a vendor check we must exclude directories that contain no .go files.
// Otherwise it is not possible to vendor just a/b/c and still import the
// non-vendored a/b. See golang.org/issue/13832.
func hasGoFiles(dir string) bool {
	fis, _ := ioutil.ReadDir(dir)
	for _, fi := range fis {
		if !fi.IsDir() && strings.HasSuffix(fi.Name(), ".go") {
			return true
		}
	}
	return false
}

// reusePackage reuses package p to satisfy the import at the top
// of the import stack stk. If this use causes an import loop,
// reusePackage updates p's error information to record the loop.
func reusePackage(p *Package, stk *importStack) *Package {
	// We use p.imports==nil to detect a package that
	// is in the midst of its own loadPackage call
	// (all the recursion below happens before p.imports gets set).
	if p.imports == nil {
		if p.Error == nil {
			p.Error = &PackageError{
				ImportStack:   stk.copy(),
				Err:           "import cycle not allowed",
				isImportCycle: true,
			}
		}
		p.Incomplete = true
	}
	// Don't rewrite the import stack in the error if we have an import cycle.
	// If we do, we'll lose the path that describes the cycle.
	if p.Error != nil && !p.Error.isImportCycle && stk.shorterThan(p.Error.ImportStack) {
		p.Error.ImportStack = stk.copy()
	}
	return p
}

// disallowInternal checks that srcDir is allowed to import p.
// If the import is allowed, disallowInternal returns the original package p.
// If not, it returns a new package containing just an appropriate error.
func disallowInternal(srcDir string, p *Package, stk *importStack) *Package {
	// golang.org/s/go14internal:
	// An import of a path containing the element “internal”
	// is disallowed if the importing code is outside the tree
	// rooted at the parent of the “internal” directory.

	// There was an error loading the package; stop here.
	if p.Error != nil {
		return p
	}

	// The stack includes p.ImportPath.
	// If that's the only thing on the stack, we started
	// with a name given on the command line, not an
	// import. Anything listed on the command line is fine.
	if len(*stk) == 1 {
		return p
	}

	// Check for "internal" element: three cases depending on begin of string and/or end of string.
	i, ok := findInternal(p.ImportPath)
	if !ok {
		return p
	}

	// Internal is present.
	// Map import path back to directory corresponding to parent of internal.
	if i > 0 {
		i-- // rewind over slash in ".../internal"
	}
	parent := p.Dir[:i+len(p.Dir)-len(p.ImportPath)]
	if hasFilePathPrefix(filepath.Clean(srcDir), filepath.Clean(parent)) {
		return p
	}

	// Look for symlinks before reporting error.
	srcDir = expandPath(srcDir)
	parent = expandPath(parent)
	if hasFilePathPrefix(filepath.Clean(srcDir), filepath.Clean(parent)) {
		return p
	}

	// Internal is present, and srcDir is outside parent's tree. Not allowed.
	perr := *p
	perr.Error = &PackageError{
		ImportStack: stk.copy(),
		Err:         "use of internal package not allowed",
	}
	perr.Incomplete = true
	return &perr
}

// findInternal looks for the final "internal" path element in the given import path.
// If there isn't one, findInternal returns ok=false.
// Otherwise, findInternal returns ok=true and the index of the "internal".
func findInternal(path string) (index int, ok bool) {
	// Three cases, depending on internal at start/end of string or not.
	// The order matters: we must return the index of the final element,
	// because the final one produces the most restrictive requirement
	// on the importer.
	switch {
	case strings.HasSuffix(path, "/internal"):
		return len(path) - len("internal"), true
	case strings.Contains(path, "/internal/"):
		return strings.LastIndex(path, "/internal/") + 1, true
	case path == "internal", strings.HasPrefix(path, "internal/"):
		return 0, true
	}
	return 0, false
}

// disallowVendor checks that srcDir is allowed to import p as path.
// If the import is allowed, disallowVendor returns the original package p.
// If not, it returns a new package containing just an appropriate error.
func disallowVendor(srcDir, path string, p *Package, stk *importStack) *Package {
	// The stack includes p.ImportPath.
	// If that's the only thing on the stack, we started
	// with a name given on the command line, not an
	// import. Anything listed on the command line is fine.
	if len(*stk) == 1 {
		return p
	}

	if perr := disallowVendorVisibility(srcDir, p, stk); perr != p {
		return perr
	}

	// Paths like x/vendor/y must be imported as y, never as x/vendor/y.
	if i, ok := findVendor(path); ok {
		perr := *p
		perr.Error = &PackageError{
			ImportStack: stk.copy(),
			Err:         "must be imported as " + path[i+len("vendor/"):],
		}
		perr.Incomplete = true
		return &perr
	}

	return p
}

// disallowVendorVisibility checks that srcDir is allowed to import p.
// The rules are the same as for /internal/ except that a path ending in /vendor
// is not subject to the rules, only subdirectories of vendor.
// This allows people to have packages and commands named vendor,
// for maximal compatibility with existing source trees.
func disallowVendorVisibility(srcDir string, p *Package, stk *importStack) *Package {
	// The stack includes p.ImportPath.
	// If that's the only thing on the stack, we started
	// with a name given on the command line, not an
	// import. Anything listed on the command line is fine.
	if len(*stk) == 1 {
		return p
	}

	// Check for "vendor" element.
	i, ok := findVendor(p.ImportPath)
	if !ok {
		return p
	}

	// Vendor is present.
	// Map import path back to directory corresponding to parent of vendor.
	if i > 0 {
		i-- // rewind over slash in ".../vendor"
	}
	truncateTo := i + len(p.Dir) - len(p.ImportPath)
	if truncateTo < 0 || len(p.Dir) < truncateTo {
		return p
	}
	parent := p.Dir[:truncateTo]
	if hasFilePathPrefix(filepath.Clean(srcDir), filepath.Clean(parent)) {
		return p
	}

	// Look for symlinks before reporting error.
	srcDir = expandPath(srcDir)
	parent = expandPath(parent)
	if hasFilePathPrefix(filepath.Clean(srcDir), filepath.Clean(parent)) {
		return p
	}

	// Vendor is present, and srcDir is outside parent's tree. Not allowed.
	perr := *p
	perr.Error = &PackageError{
		ImportStack: stk.copy(),
		Err:         "use of vendored package not allowed",
	}
	perr.Incomplete = true
	return &perr
}

// findVendor looks for the last non-terminating "vendor" path element in the given import path.
// If there isn't one, findVendor returns ok=false.
// Otherwise, findVendor returns ok=true and the index of the "vendor".
//
// Note that terminating "vendor" elements don't count: "x/vendor" is its own package,
// not the vendored copy of an import "" (the empty import path).
// This will allow people to have packages or commands named vendor.
// This may help reduce breakage, or it may just be confusing. We'll see.
func findVendor(path string) (index int, ok bool) {
	// Two cases, depending on internal at start of string or not.
	// The order matters: we must return the index of the final element,
	// because the final one is where the effective import path starts.
	switch {
	case strings.Contains(path, "/vendor/"):
		return strings.LastIndex(path, "/vendor/") + 1, true
	case strings.HasPrefix(path, "vendor/"):
		return 0, true
	}
	return 0, false
}

type targetDir int

const (
	toRoot    targetDir = iota // to bin dir inside package root (default)
	toTool                     // GOROOT/pkg/tool
	stalePath                  // the old import path; fail to build
)

// goTools is a map of Go program import path to install target directory.
var goTools = map[string]targetDir{
	"cmd/addr2line":                        toTool,
	"cmd/api":                              toTool,
	"cmd/asm":                              toTool,
	"cmd/compile":                          toTool,
	"cmd/cgo":                              toTool,
	"cmd/cover":                            toTool,
	"cmd/dist":                             toTool,
	"cmd/doc":                              toTool,
	"cmd/fix":                              toTool,
	"cmd/link":                             toTool,
	"cmd/newlink":                          toTool,
	"cmd/nm":                               toTool,
	"cmd/objdump":                          toTool,
	"cmd/pack":                             toTool,
	"cmd/pprof":                            toTool,
	"cmd/trace":                            toTool,
	"cmd/vet":                              toTool,
	"cmd/yacc":                             toTool,
	"code.google.com/p/go.tools/cmd/cover": stalePath,
	"code.google.com/p/go.tools/cmd/godoc": stalePath,
	"code.google.com/p/go.tools/cmd/vet":   stalePath,
}

// expandScanner expands a scanner.List error into all the errors in the list.
// The default Error method only shows the first error.
func expandScanner(err error) error {
	// Look for parser errors.
	if err, ok := err.(scanner.ErrorList); ok {
		// Prepare error with \n before each message.
		// When printed in something like context: %v
		// this will put the leading file positions each on
		// its own line. It will also show all the errors
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

var raceExclude = map[string]bool{
	"runtime/race": true,
	"runtime/msan": true,
	"runtime/cgo":  true,
	"cmd/cgo":      true,
	"syscall":      true,
	"errors":       true,
}

var cgoExclude = map[string]bool{
	"runtime/cgo": true,
}

var cgoSyscallExclude = map[string]bool{
	"runtime/cgo":  true,
	"runtime/race": true,
	"runtime/msan": true,
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

	useBindir := p.Name == "main"
	if !p.Standard {
		switch buildBuildmode {
		case "c-archive", "c-shared":
			useBindir = false
		}
	}

	if useBindir {
		// Report an error when the old code.google.com/p/go.tools paths are used.
		if goTools[p.ImportPath] == stalePath {
			newPath := strings.Replace(p.ImportPath, "code.google.com/p/go.", "golang.org/x/", 1)
			e := fmt.Sprintf("the %v command has moved; use %v instead.", p.ImportPath, newPath)
			p.Error = &PackageError{Err: e}
			return p
		}
		_, elem := filepath.Split(p.Dir)
		full := buildContext.GOOS + "_" + buildContext.GOARCH + "/" + elem
		if buildContext.GOOS != toolGOOS || buildContext.GOARCH != toolGOARCH {
			// Install cross-compiled binaries to subdirectories of bin.
			elem = full
		}
		if p.build.BinDir != "" {
			// Install to GOBIN or bin of GOPATH entry.
			p.target = filepath.Join(p.build.BinDir, elem)
			if !p.Goroot && strings.Contains(elem, "/") && gobin != "" {
				// Do not create $GOBIN/goos_goarch/elem.
				p.target = ""
				p.gobinSubdir = true
			}
		}
		if goTools[p.ImportPath] == toTool {
			// This is for 'go tool'.
			// Override all the usual logic and force it into the tool directory.
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
		if buildLinkshared {
			shlibnamefile := p.target[:len(p.target)-2] + ".shlibname"
			shlib, err := ioutil.ReadFile(shlibnamefile)
			if err == nil {
				libname := strings.TrimSpace(string(shlib))
				if buildContext.Compiler == "gccgo" {
					p.Shlib = filepath.Join(p.build.PkgTargetRoot, "shlibs", libname)
				} else {
					p.Shlib = filepath.Join(p.build.PkgTargetRoot, libname)

				}
			} else if !os.IsNotExist(err) {
				fatalf("unexpected error reading %s: %v", shlibnamefile, err)
			}
		}
	}

	importPaths := p.Imports
	// Packages that use cgo import runtime/cgo implicitly.
	// Packages that use cgo also import syscall implicitly,
	// to wrap errno.
	// Exclude certain packages to avoid circular dependencies.
	if len(p.CgoFiles) > 0 && (!p.Standard || !cgoExclude[p.ImportPath]) {
		importPaths = append(importPaths, "runtime/cgo")
	}
	if len(p.CgoFiles) > 0 && (!p.Standard || !cgoSyscallExclude[p.ImportPath]) {
		importPaths = append(importPaths, "syscall")
	}

	// Currently build modes c-shared, pie, and -linkshared force
	// external linking mode, and external linking mode forces an
	// import of runtime/cgo.
	if p.Name == "main" && !p.Goroot && (buildBuildmode == "c-shared" || buildBuildmode == "pie" || buildLinkshared) {
		importPaths = append(importPaths, "runtime/cgo")
	}

	// Everything depends on runtime, except runtime, its internal
	// subpackages, and unsafe.
	if !p.Standard || (p.ImportPath != "runtime" && !strings.HasPrefix(p.ImportPath, "runtime/internal/") && p.ImportPath != "unsafe") {
		importPaths = append(importPaths, "runtime")
		// When race detection enabled everything depends on runtime/race.
		// Exclude certain packages to avoid circular dependencies.
		if buildRace && (!p.Standard || !raceExclude[p.ImportPath]) {
			importPaths = append(importPaths, "runtime/race")
		}
		// MSan uses runtime/msan.
		if buildMSan && (!p.Standard || !raceExclude[p.ImportPath]) {
			importPaths = append(importPaths, "runtime/msan")
		}
		// On ARM with GOARM=5, everything depends on math for the link.
		if p.Name == "main" && goarch == "arm" {
			importPaths = append(importPaths, "math")
		}
	}

	// Runtime and its internal packages depend on runtime/internal/sys,
	// so that they pick up the generated zversion.go file.
	// This can be an issue particularly for runtime/internal/atomic;
	// see issue 13655.
	if p.Standard && (p.ImportPath == "runtime" || strings.HasPrefix(p.ImportPath, "runtime/internal/")) && p.ImportPath != "runtime/internal/sys" {
		importPaths = append(importPaths, "runtime/internal/sys")
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
		p.CXXFiles,
		p.MFiles,
		p.HFiles,
		p.FFiles,
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
	deps := make(map[string]*Package)
	for i, path := range importPaths {
		if path == "C" {
			continue
		}
		p1 := loadImport(path, p.Dir, p, stk, p.build.ImportPos[path], useVendor)
		if p1.Name == "main" {
			p.Error = &PackageError{
				ImportStack: stk.copy(),
				Err:         fmt.Sprintf("import %q is a program, not an importable package", path),
			}
			pos := p.build.ImportPos[path]
			if len(pos) > 0 {
				p.Error.Pos = pos[0].String()
			}
		}
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
		}
		if p.Standard && p.Error == nil && !p1.Standard && p1.Error == nil {
			p.Error = &PackageError{
				ImportStack: stk.copy(),
				Err:         fmt.Sprintf("non-standard import %q in standard package %q", path, p.ImportPath),
			}
			pos := p.build.ImportPos[path]
			if len(pos) > 0 {
				p.Error.Pos = pos[0].String()
			}
		}

		path = p1.ImportPath
		importPaths[i] = path
		if i < len(p.Imports) {
			p.Imports[i] = path
		}
		deps[path] = p1
		imports = append(imports, p1)
		for _, dep := range p1.deps {
			// The same import path could produce an error or not,
			// depending on what tries to import it.
			// Prefer to record entries with errors, so we can report them.
			if deps[dep.ImportPath] == nil || dep.Error != nil {
				deps[dep.ImportPath] = dep
			}
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
		p1 := deps[dep]
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

	// If cgo is not enabled, ignore cgo supporting sources
	// just as we ignore go files containing import "C".
	if !buildContext.CgoEnabled {
		p.CFiles = nil
		p.CXXFiles = nil
		p.MFiles = nil
		p.SwigFiles = nil
		p.SwigCXXFiles = nil
		// Note that SFiles are okay (they go to the Go assembler)
		// and HFiles are okay (they might be used by the SFiles).
		// Also Sysofiles are okay (they might not contain object
		// code; see issue #16050).
	}

	// The gc toolchain only permits C source files with cgo.
	if len(p.CFiles) > 0 && !p.usesCgo() && !p.usesSwig() && buildContext.Compiler == "gc" {
		p.Error = &PackageError{
			ImportStack: stk.copy(),
			Err:         fmt.Sprintf("C source files not allowed when not using cgo or SWIG: %s", strings.Join(p.CFiles, " ")),
		}
		return p
	}

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

	if p.BinaryOnly {
		// For binary-only package, use build ID from supplied package binary.
		buildID, err := readBuildID(p)
		if err == nil {
			p.buildID = buildID
		}
	} else {
		computeBuildID(p)
	}
	return p
}

// usesSwig reports whether the package needs to run SWIG.
func (p *Package) usesSwig() bool {
	return len(p.SwigFiles) > 0 || len(p.SwigCXXFiles) > 0
}

// usesCgo reports whether the package needs to run cgo
func (p *Package) usesCgo() bool {
	return len(p.CgoFiles) > 0
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
	for _, p := range packageList(pkgs) {
		p.Stale, p.StaleReason = isStale(p)
	}
}

// The runtime version string takes one of two forms:
// "go1.X[.Y]" for Go releases, and "devel +hash" at tip.
// Determine whether we are in a released copy by
// inspecting the version.
var isGoRelease = strings.HasPrefix(runtime.Version(), "go1")

// isStale and computeBuildID
//
// Theory of Operation
//
// There is an installed copy of the package (or binary).
// Can we reuse the installed copy, or do we need to build a new one?
//
// We can use the installed copy if it matches what we'd get
// by building a new one. The hard part is predicting that without
// actually running a build.
//
// To start, we must know the set of inputs to the build process that can
// affect the generated output. At a minimum, that includes the source
// files for the package and also any compiled packages imported by those
// source files. The *Package has these, and we use them. One might also
// argue for including in the input set: the build tags, whether the race
// detector is in use, the target operating system and architecture, the
// compiler and linker binaries being used, the additional flags being
// passed to those, the cgo binary being used, the additional flags cgo
// passes to the host C compiler, the host C compiler being used, the set
// of host C include files and installed C libraries, and so on.
// We include some but not all of this information.
//
// Once we have decided on a set of inputs, we must next decide how to
// tell whether the content of that set has changed since the last build
// of p. If there have been no changes, then we assume a new build would
// produce the same result and reuse the installed package or binary.
// But if there have been changes, then we assume a new build might not
// produce the same result, so we rebuild.
//
// There are two common ways to decide whether the content of the set has
// changed: modification times and content hashes. We use a mixture of both.
//
// The use of modification times (mtimes) was pioneered by make:
// assuming that a file's mtime is an accurate record of when that file was last written,
// and assuming that the modification time of an installed package or
// binary is the time that it was built, if the mtimes of the inputs
// predate the mtime of the installed object, then the build of that
// object saw those versions of the files, and therefore a rebuild using
// those same versions would produce the same object. In contrast, if any
// mtime of an input is newer than the mtime of the installed object, a
// change has occurred since the build, and the build should be redone.
//
// Modification times are attractive because the logic is easy to
// understand and the file system maintains the mtimes automatically
// (less work for us). Unfortunately, there are a variety of ways in
// which the mtime approach fails to detect a change and reuses a stale
// object file incorrectly. (Making the opposite mistake, rebuilding
// unnecessarily, is only a performance problem and not a correctness
// problem, so we ignore that one.)
//
// As a warmup, one problem is that to be perfectly precise, we need to
// compare the input mtimes against the time at the beginning of the
// build, but the object file time is the time at the end of the build.
// If an input file changes after being read but before the object is
// written, the next build will see an object newer than the input and
// will incorrectly decide that the object is up to date. We make no
// attempt to detect or solve this problem.
//
// Another problem is that due to file system imprecision, an input and
// output that are actually ordered in time have the same mtime.
// This typically happens on file systems with 1-second (or, worse,
// 2-second) mtime granularity and with automated scripts that write an
// input and then immediately run a build, or vice versa. If an input and
// an output have the same mtime, the conservative behavior is to treat
// the output as out-of-date and rebuild. This can cause one or more
// spurious rebuilds, but only for 1 second, until the object finally has
// an mtime later than the input.
//
// Another problem is that binary distributions often set the mtime on
// all files to the same time. If the distribution includes both inputs
// and cached build outputs, the conservative solution to the previous
// problem will cause unnecessary rebuilds. Worse, in such a binary
// distribution, those rebuilds might not even have permission to update
// the cached build output. To avoid these write errors, if an input and
// output have the same mtime, we assume the output is up-to-date.
// This is the opposite of what the previous problem would have us do,
// but binary distributions are more common than instances of the
// previous problem.
//
// A variant of the last problem is that some binary distributions do not
// set the mtime on all files to the same time. Instead they let the file
// system record mtimes as the distribution is unpacked. If the outputs
// are unpacked before the inputs, they'll be older and a build will try
// to rebuild them. That rebuild might hit the same write errors as in
// the last scenario. We don't make any attempt to solve this, and we
// haven't had many reports of it. Perhaps the only time this happens is
// when people manually unpack the distribution, and most of the time
// that's done as the same user who will be using it, so an initial
// rebuild on first use succeeds quietly.
//
// More generally, people and programs change mtimes on files. The last
// few problems were specific examples of this, but it's a general problem.
// For example, instead of a binary distribution, copying a home
// directory from one directory or machine to another might copy files
// but not preserve mtimes. If the inputs are new than the outputs on the
// first machine but copied first, they end up older than the outputs on
// the second machine.
//
// Because many other build systems have the same sensitivity to mtimes,
// most programs manipulating source code take pains not to break the
// mtime assumptions. For example, Git does not set the mtime of files
// during a checkout operation, even when checking out an old version of
// the code. This decision was made specifically to work well with
// mtime-based build systems.
//
// The killer problem, though, for mtime-based build systems is that the
// build only has access to the mtimes of the inputs that still exist.
// If it is possible to remove an input without changing any other inputs,
// a later build will think the object is up-to-date when it is not.
// This happens for Go because a package is made up of all source
// files in a directory. If a source file is removed, there is no newer
// mtime available recording that fact. The mtime on the directory could
// be used, but it also changes when unrelated files are added to or
// removed from the directory, so including the directory mtime would
// cause unnecessary rebuilds, possibly many. It would also exacerbate
// the problems mentioned earlier, since even programs that are careful
// to maintain mtimes on files rarely maintain mtimes on directories.
//
// A variant of the last problem is when the inputs change for other
// reasons. For example, Go 1.4 and Go 1.5 both install $GOPATH/src/mypkg
// into the same target, $GOPATH/pkg/$GOOS_$GOARCH/mypkg.a.
// If Go 1.4 has built mypkg into mypkg.a, a build using Go 1.5 must
// rebuild mypkg.a, but from mtimes alone mypkg.a looks up-to-date.
// If Go 1.5 has just been installed, perhaps the compiler will have a
// newer mtime; since the compiler is considered an input, that would
// trigger a rebuild. But only once, and only the last Go 1.4 build of
// mypkg.a happened before Go 1.5 was installed. If a user has the two
// versions installed in different locations and flips back and forth,
// mtimes alone cannot tell what to do. Changing the toolchain is
// changing the set of inputs, without affecting any mtimes.
//
// To detect the set of inputs changing, we turn away from mtimes and to
// an explicit data comparison. Specifically, we build a list of the
// inputs to the build, compute its SHA1 hash, and record that as the
// ``build ID'' in the generated object. At the next build, we can
// recompute the build ID and compare it to the one in the generated
// object. If they differ, the list of inputs has changed, so the object
// is out of date and must be rebuilt.
//
// Because this build ID is computed before the build begins, the
// comparison does not have the race that mtime comparison does.
//
// Making the build sensitive to changes in other state is
// straightforward: include the state in the build ID hash, and if it
// changes, so does the build ID, triggering a rebuild.
//
// To detect changes in toolchain, we include the toolchain version in
// the build ID hash for package runtime, and then we include the build
// IDs of all imported packages in the build ID for p.
//
// It is natural to think about including build tags in the build ID, but
// the naive approach of just dumping the tags into the hash would cause
// spurious rebuilds. For example, 'go install' and 'go install -tags neverusedtag'
// produce the same binaries (assuming neverusedtag is never used).
// A more precise approach would be to include only tags that have an
// effect on the build. But the effect of a tag on the build is to
// include or exclude a file from the compilation, and that file list is
// already in the build ID hash. So the build ID is already tag-sensitive
// in a perfectly precise way. So we do NOT explicitly add build tags to
// the build ID hash.
//
// We do not include as part of the build ID the operating system,
// architecture, or whether the race detector is enabled, even though all
// three have an effect on the output, because that information is used
// to decide the install location. Binaries for linux and binaries for
// darwin are written to different directory trees; including that
// information in the build ID is unnecessary (although it would be
// harmless).
//
// TODO(rsc): Investigate the cost of putting source file content into
// the build ID hash as a replacement for the use of mtimes. Using the
// file content would avoid all the mtime problems, but it does require
// reading all the source files, something we avoid today (we read the
// beginning to find the build tags and the imports, but we stop as soon
// as we see the import block is over). If the package is stale, the compiler
// is going to read the files anyway. But if the package is up-to-date, the
// read is overhead.
//
// TODO(rsc): Investigate the complexity of making the build more
// precise about when individual results are needed. To be fully precise,
// there are two results of a compilation: the entire .a file used by the link
// and the subpiece used by later compilations (__.PKGDEF only).
// If a rebuild is needed but produces the previous __.PKGDEF, then
// no more recompilation due to the rebuilt package is needed, only
// relinking. To date, there is nothing in the Go command to express this.
//
// Special Cases
//
// When the go command makes the wrong build decision and does not
// rebuild something it should, users fall back to adding the -a flag.
// Any common use of the -a flag should be considered prima facie evidence
// that isStale is returning an incorrect false result in some important case.
// Bugs reported in the behavior of -a itself should prompt the question
// ``Why is -a being used at all? What bug does that indicate?''
//
// There is a long history of changes to isStale to try to make -a into a
// suitable workaround for bugs in the mtime-based decisions.
// It is worth recording that history to inform (and, as much as possible, deter) future changes.
//
// (1) Before the build IDs were introduced, building with alternate tags
// would happily reuse installed objects built without those tags.
// For example, "go build -tags netgo myprog.go" would use the installed
// copy of package net, even if that copy had been built without netgo.
// (The netgo tag controls whether package net uses cgo or pure Go for
// functionality such as name resolution.)
// Using the installed non-netgo package defeats the purpose.
//
// Users worked around this with "go build -tags netgo -a myprog.go".
//
// Build IDs have made that workaround unnecessary:
// "go build -tags netgo myprog.go"
// cannot use a non-netgo copy of package net.
//
// (2) Before the build IDs were introduced, building with different toolchains,
// especially changing between toolchains, tried to reuse objects stored in
// $GOPATH/pkg, resulting in link-time errors about object file mismatches.
//
// Users worked around this with "go install -a ./...".
//
// Build IDs have made that workaround unnecessary:
// "go install ./..." will rebuild any objects it finds that were built against
// a different toolchain.
//
// (3) The common use of "go install -a ./..." led to reports of problems
// when the -a forced the rebuild of the standard library, which for some
// users was not writable. Because we didn't understand that the real
// problem was the bug -a was working around, we changed -a not to
// apply to the standard library.
//
// (4) The common use of "go build -tags netgo -a myprog.go" broke
// when we changed -a not to apply to the standard library, because
// if go build doesn't rebuild package net, it uses the non-netgo version.
//
// Users worked around this with "go build -tags netgo -installsuffix barf myprog.go".
// The -installsuffix here is making the go command look for packages
// in pkg/$GOOS_$GOARCH_barf instead of pkg/$GOOS_$GOARCH.
// Since the former presumably doesn't exist, go build decides to rebuild
// everything, including the standard library. Since go build doesn't
// install anything it builds, nothing is ever written to pkg/$GOOS_$GOARCH_barf,
// so repeated invocations continue to work.
//
// If the use of -a wasn't a red flag, the use of -installsuffix to point to
// a non-existent directory in a command that installs nothing should
// have been.
//
// (5) Now that (1) and (2) no longer need -a, we have removed the kludge
// introduced in (3): once again, -a means ``rebuild everything,'' not
// ``rebuild everything except the standard library.'' Only Go 1.4 had
// the restricted meaning.
//
// In addition to these cases trying to trigger rebuilds, there are
// special cases trying NOT to trigger rebuilds. The main one is that for
// a variety of reasons (see above), the install process for a Go release
// cannot be relied upon to set the mtimes such that the go command will
// think the standard library is up to date. So the mtime evidence is
// ignored for the standard library if we find ourselves in a release
// version of Go. Build ID-based staleness checks still apply to the
// standard library, even in release versions. This makes
// 'go build -tags netgo' work, among other things.

// isStale reports whether package p needs to be rebuilt,
// along with the reason why.
func isStale(p *Package) (bool, string) {
	if p.Standard && (p.ImportPath == "unsafe" || buildContext.Compiler == "gccgo") {
		// fake, builtin package
		return false, "builtin package"
	}
	if p.Error != nil {
		return true, "errors loading package"
	}
	if p.Stale {
		return true, p.StaleReason
	}

	// If this is a package with no source code, it cannot be rebuilt.
	// If the binary is missing, we mark the package stale so that
	// if a rebuild is needed, that rebuild attempt will produce a useful error.
	// (Some commands, such as 'go list', do not attempt to rebuild.)
	if p.BinaryOnly {
		if p.target == "" {
			// Fail if a build is attempted.
			return true, "no source code for package, but no install target"
		}
		if _, err := os.Stat(p.target); err != nil {
			// Fail if a build is attempted.
			return true, "no source code for package, but cannot access install target: " + err.Error()
		}
		return false, "no source code for package"
	}

	// If the -a flag is given, rebuild everything.
	if buildA {
		return true, "build -a flag in use"
	}

	// If there's no install target, we have to rebuild.
	if p.target == "" {
		return true, "no install target"
	}

	// Package is stale if completely unbuilt.
	fi, err := os.Stat(p.target)
	if err != nil {
		return true, "cannot stat install target"
	}

	// Package is stale if the expected build ID differs from the
	// recorded build ID. This catches changes like a source file
	// being removed from a package directory. See issue 3895.
	// It also catches changes in build tags that affect the set of
	// files being compiled. See issue 9369.
	// It also catches changes in toolchain, like when flipping between
	// two versions of Go compiling a single GOPATH.
	// See issue 8290 and issue 10702.
	targetBuildID, err := readBuildID(p)
	if err == nil && targetBuildID != p.buildID {
		return true, "build ID mismatch"
	}

	// Package is stale if a dependency is.
	for _, p1 := range p.deps {
		if p1.Stale {
			return true, "stale dependency"
		}
	}

	// The checks above are content-based staleness.
	// We assume they are always accurate.
	//
	// The checks below are mtime-based staleness.
	// We hope they are accurate, but we know that they fail in the case of
	// prebuilt Go installations that don't preserve the build mtimes
	// (for example, if the pkg/ mtimes are before the src/ mtimes).
	// See the large comment above isStale for details.

	// If we are running a release copy of Go and didn't find a content-based
	// reason to rebuild the standard packages, do not rebuild them.
	// They may not be writable anyway, but they are certainly not changing.
	// This makes 'go build' skip the standard packages when
	// using an official release, even when the mtimes have been changed.
	// See issue 3036, issue 3149, issue 4106, issue 8290.
	// (If a change to a release tree must be made by hand, the way to force the
	// install is to run make.bash, which will remove the old package archives
	// before rebuilding.)
	if p.Standard && isGoRelease {
		return false, "standard package in Go release distribution"
	}

	// Time-based staleness.

	built := fi.ModTime()

	olderThan := func(file string) bool {
		fi, err := os.Stat(file)
		return err != nil || fi.ModTime().After(built)
	}

	// Package is stale if a dependency is, or if a dependency is newer.
	for _, p1 := range p.deps {
		if p1.target != "" && olderThan(p1.target) {
			return true, "newer dependency"
		}
	}

	// As a courtesy to developers installing new versions of the compiler
	// frequently, define that packages are stale if they are
	// older than the compiler, and commands if they are older than
	// the linker. This heuristic will not work if the binaries are
	// back-dated, as some binary distributions may do, but it does handle
	// a very common case.
	// See issue 3036.
	// Exclude $GOROOT, under the assumption that people working on
	// the compiler may want to control when everything gets rebuilt,
	// and people updating the Go repository will run make.bash or all.bash
	// and get a full rebuild anyway.
	// Excluding $GOROOT used to also fix issue 4106, but that's now
	// taken care of above (at least when the installed Go is a released version).
	if p.Root != goroot {
		if olderThan(buildToolchain.compiler()) {
			return true, "newer compiler"
		}
		if p.build.IsCommand() && olderThan(buildToolchain.linker()) {
			return true, "newer linker"
		}
	}

	// Note: Until Go 1.5, we had an additional shortcut here.
	// We built a list of the workspace roots ($GOROOT, each $GOPATH)
	// containing targets directly named on the command line,
	// and if p were not in any of those, it would be treated as up-to-date
	// as long as it is built. The goal was to avoid rebuilding a system-installed
	// $GOROOT, unless something from $GOROOT were explicitly named
	// on the command line (like go install math).
	// That's now handled by the isGoRelease clause above.
	// The other effect of the shortcut was to isolate different entries in
	// $GOPATH from each other. This had the unfortunate effect that
	// if you had (say), GOPATH listing two entries, one for commands
	// and one for libraries, and you did a 'git pull' in the library one
	// and then tried 'go install commands/...', it would build the new libraries
	// during the first build (because they wouldn't have been installed at all)
	// but then subsequent builds would not rebuild the libraries, even if the
	// mtimes indicate they are stale, because the different GOPATH entries
	// were treated differently. This behavior was confusing when using
	// non-trivial GOPATHs, which were particularly common with some
	// code management conventions, like the original godep.
	// Since the $GOROOT case (the original motivation) is handled separately,
	// we no longer put a barrier between the different $GOPATH entries.
	//
	// One implication of this is that if there is a system directory for
	// non-standard Go packages that is included in $GOPATH, the mtimes
	// on those compiled packages must be no earlier than the mtimes
	// on the source files. Since most distributions use the same mtime
	// for all files in a tree, they will be unaffected. People using plain
	// tar x to extract system-installed packages will need to adjust mtimes,
	// but it's better to force them to get the mtimes right than to ignore
	// the mtimes and thereby do the wrong thing in common use cases.
	//
	// So there is no GOPATH vs GOPATH shortcut here anymore.
	//
	// If something needs to come back here, we could try writing a dummy
	// file with a random name to the $GOPATH/pkg directory (and removing it)
	// to test for write access, and then skip GOPATH roots we don't have write
	// access to. But hopefully we can just use the mtimes always.

	srcs := stringList(p.GoFiles, p.CFiles, p.CXXFiles, p.MFiles, p.HFiles, p.FFiles, p.SFiles, p.CgoFiles, p.SysoFiles, p.SwigFiles, p.SwigCXXFiles)
	for _, src := range srcs {
		if olderThan(filepath.Join(p.Dir, src)) {
			return true, "newer source file"
		}
	}

	return false, ""
}

// computeBuildID computes the build ID for p, leaving it in p.buildID.
// Build ID is a hash of the information we want to detect changes in.
// See the long comment in isStale for details.
func computeBuildID(p *Package) {
	h := sha1.New()

	// Include the list of files compiled as part of the package.
	// This lets us detect removed files. See issue 3895.
	inputFiles := stringList(
		p.GoFiles,
		p.CgoFiles,
		p.CFiles,
		p.CXXFiles,
		p.MFiles,
		p.HFiles,
		p.SFiles,
		p.SysoFiles,
		p.SwigFiles,
		p.SwigCXXFiles,
	)
	for _, file := range inputFiles {
		fmt.Fprintf(h, "file %s\n", file)
	}

	// Include the content of runtime/internal/sys/zversion.go in the hash
	// for package runtime. This will give package runtime a
	// different build ID in each Go release.
	if p.Standard && p.ImportPath == "runtime/internal/sys" {
		data, err := ioutil.ReadFile(filepath.Join(p.Dir, "zversion.go"))
		if err != nil {
			fatalf("go: %s", err)
		}
		fmt.Fprintf(h, "zversion %q\n", string(data))
	}

	// Include the build IDs of any dependencies in the hash.
	// This, combined with the runtime/zversion content,
	// will cause packages to have different build IDs when
	// compiled with different Go releases.
	// This helps the go command know to recompile when
	// people use the same GOPATH but switch between
	// different Go releases. See issue 10702.
	// This is also a better fix for issue 8290.
	for _, p1 := range p.deps {
		fmt.Fprintf(h, "dep %s %s\n", p1.ImportPath, p1.buildID)
	}

	p.buildID = fmt.Sprintf("%x", h.Sum(nil))
}

var cwd, _ = os.Getwd()

var cmdCache = map[string]*Package{}

// loadPackage is like loadImport but is used for command-line arguments,
// not for paths found in import statements. In addition to ordinary import paths,
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
	if strings.HasPrefix(arg, "cmd/") && !strings.Contains(arg[4:], "/") {
		if p := cmdCache[arg]; p != nil {
			return p
		}
		stk.push(arg)
		defer stk.pop()

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

	return loadImport(arg, cwd, nil, stk, nil, 0)
}

// packages returns the packages named by the
// command line arguments 'args'.  If a named package
// cannot be loaded at all (for example, if the directory does not exist),
// then packages prints an error and does not include that
// package in the results. However, if errors occur trying
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
	var (
		pkgs    []*Package
		stk     importStack
		seenArg = make(map[string]bool)
		seenPkg = make(map[*Package]bool)
	)

	for _, arg := range args {
		if seenArg[arg] {
			continue
		}
		seenArg[arg] = true
		pkg := loadPackage(arg, &stk)
		if seenPkg[pkg] {
			continue
		}
		seenPkg[pkg] = true
		pkgs = append(pkgs, pkg)
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

	// Check for duplicate loads of the same package.
	// That should be impossible, but if it does happen then
	// we end up trying to build the same package twice,
	// usually in parallel overwriting the same files,
	// which doesn't work very well.
	seen := map[string]bool{}
	reported := map[string]bool{}
	for _, pkg := range packageList(pkgs) {
		if seen[pkg.ImportPath] && !reported[pkg.ImportPath] {
			reported[pkg.ImportPath] = true
			errorf("internal error: duplicate loads of %s", pkg.ImportPath)
		}
		seen[pkg.ImportPath] = true
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

var (
	errBuildIDToolchain = fmt.Errorf("build ID only supported in gc toolchain")
	errBuildIDMalformed = fmt.Errorf("malformed object file")
	errBuildIDUnknown   = fmt.Errorf("lost build ID")
)

var (
	bangArch = []byte("!<arch>")
	pkgdef   = []byte("__.PKGDEF")
	goobject = []byte("go object ")
	buildid  = []byte("build id ")
)

// readBuildID reads the build ID from an archive or binary.
// It only supports the gc toolchain.
// Other toolchain maintainers should adjust this function.
func readBuildID(p *Package) (id string, err error) {
	if buildToolchain != (gcToolchain{}) {
		return "", errBuildIDToolchain
	}

	// For commands, read build ID directly from binary.
	if p.Name == "main" {
		return ReadBuildIDFromBinary(p.Target)
	}

	// Otherwise, we expect to have an archive (.a) file,
	// and we can read the build ID from the Go export data.
	if !strings.HasSuffix(p.Target, ".a") {
		return "", &os.PathError{Op: "parse", Path: p.Target, Err: errBuildIDUnknown}
	}

	// Read just enough of the target to fetch the build ID.
	// The archive is expected to look like:
	//
	//	!<arch>
	//	__.PKGDEF       0           0     0     644     7955      `
	//	go object darwin amd64 devel X:none
	//	build id "b41e5c45250e25c9fd5e9f9a1de7857ea0d41224"
	//
	// The variable-sized strings are GOOS, GOARCH, and the experiment list (X:none).
	// Reading the first 1024 bytes should be plenty.
	f, err := os.Open(p.Target)
	if err != nil {
		return "", err
	}
	data := make([]byte, 1024)
	n, err := io.ReadFull(f, data)
	f.Close()

	if err != nil && n == 0 {
		return "", err
	}

	bad := func() (string, error) {
		return "", &os.PathError{Op: "parse", Path: p.Target, Err: errBuildIDMalformed}
	}

	// Archive header.
	for i := 0; ; i++ { // returns during i==3
		j := bytes.IndexByte(data, '\n')
		if j < 0 {
			return bad()
		}
		line := data[:j]
		data = data[j+1:]
		switch i {
		case 0:
			if !bytes.Equal(line, bangArch) {
				return bad()
			}
		case 1:
			if !bytes.HasPrefix(line, pkgdef) {
				return bad()
			}
		case 2:
			if !bytes.HasPrefix(line, goobject) {
				return bad()
			}
		case 3:
			if !bytes.HasPrefix(line, buildid) {
				// Found the object header, just doesn't have a build id line.
				// Treat as successful, with empty build id.
				return "", nil
			}
			id, err := strconv.Unquote(string(line[len(buildid):]))
			if err != nil {
				return bad()
			}
			return id, nil
		}
	}
}

var (
	goBuildPrefix = []byte("\xff Go build ID: \"")
	goBuildEnd    = []byte("\"\n \xff")

	elfPrefix = []byte("\x7fELF")

	machoPrefixes = [][]byte{
		{0xfe, 0xed, 0xfa, 0xce},
		{0xfe, 0xed, 0xfa, 0xcf},
		{0xce, 0xfa, 0xed, 0xfe},
		{0xcf, 0xfa, 0xed, 0xfe},
	}
)

var BuildIDReadSize = 32 * 1024 // changed for testing

// ReadBuildIDFromBinary reads the build ID from a binary.
//
// ELF binaries store the build ID in a proper PT_NOTE section.
//
// Other binary formats are not so flexible. For those, the linker
// stores the build ID as non-instruction bytes at the very beginning
// of the text segment, which should appear near the beginning
// of the file. This is clumsy but fairly portable. Custom locations
// can be added for other binary types as needed, like we did for ELF.
func ReadBuildIDFromBinary(filename string) (id string, err error) {
	if filename == "" {
		return "", &os.PathError{Op: "parse", Path: filename, Err: errBuildIDUnknown}
	}

	// Read the first 32 kB of the binary file.
	// That should be enough to find the build ID.
	// In ELF files, the build ID is in the leading headers,
	// which are typically less than 4 kB, not to mention 32 kB.
	// In Mach-O files, there's no limit, so we have to parse the file.
	// On other systems, we're trying to read enough that
	// we get the beginning of the text segment in the read.
	// The offset where the text segment begins in a hello
	// world compiled for each different object format today:
	//
	//	Plan 9: 0x20
	//	Windows: 0x600
	//
	f, err := os.Open(filename)
	if err != nil {
		return "", err
	}
	defer f.Close()

	data := make([]byte, BuildIDReadSize)
	_, err = io.ReadFull(f, data)
	if err == io.ErrUnexpectedEOF {
		err = nil
	}
	if err != nil {
		return "", err
	}

	if bytes.HasPrefix(data, elfPrefix) {
		return readELFGoBuildID(filename, f, data)
	}
	for _, m := range machoPrefixes {
		if bytes.HasPrefix(data, m) {
			return readMachoGoBuildID(filename, f, data)
		}
	}

	return readRawGoBuildID(filename, data)
}

// readRawGoBuildID finds the raw build ID stored in text segment data.
func readRawGoBuildID(filename string, data []byte) (id string, err error) {
	i := bytes.Index(data, goBuildPrefix)
	if i < 0 {
		// Missing. Treat as successful but build ID empty.
		return "", nil
	}

	j := bytes.Index(data[i+len(goBuildPrefix):], goBuildEnd)
	if j < 0 {
		return "", &os.PathError{Op: "parse", Path: filename, Err: errBuildIDMalformed}
	}

	quoted := data[i+len(goBuildPrefix)-1 : i+len(goBuildPrefix)+j+1]
	id, err = strconv.Unquote(string(quoted))
	if err != nil {
		return "", &os.PathError{Op: "parse", Path: filename, Err: errBuildIDMalformed}
	}

	return id, nil
}
