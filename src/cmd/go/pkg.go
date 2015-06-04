// Copyright 2011 The Go Authors.  All rights reserved.
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
	"time"
	"unicode"
)

// A Package describes a single package found in a directory.
type Package struct {
	// Note: These fields are part of the go command's public API.
	// See list.go.  It is okay to add fields, but not to change or
	// remove existing ones.  Keep in sync with list.go
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
	Root          string `json:",omitempty"` // Go root or Go path dir containing this package
	ConflictDir   string `json:",omitempty"` // Dir is hidden by this other directory

	// Source files
	GoFiles        []string `json:",omitempty"` // .go source files (excluding CgoFiles, TestGoFiles, XTestGoFiles)
	CgoFiles       []string `json:",omitempty"` // .go sources files that import "C"
	IgnoredGoFiles []string `json:",omitempty"` // .go sources ignored due to build constraints
	CFiles         []string `json:",omitempty"` // .c source files
	CXXFiles       []string `json:",omitempty"` // .cc, .cpp and .cxx source files
	MFiles         []string `json:",omitempty"` // .m source files
	HFiles         []string `json:",omitempty"` // .h, .hh, .hpp and .hxx source files
	SFiles         []string `json:",omitempty"` // .s source files
	SwigFiles      []string `json:",omitempty"` // .swig files
	SwigCXXFiles   []string `json:",omitempty"` // .swigcxx files
	SysoFiles      []string `json:",omitempty"` // .syso system object files added to package

	// Cgo directives
	CgoCFLAGS    []string `json:",omitempty"` // cgo: flags for C compiler
	CgoCPPFLAGS  []string `json:",omitempty"` // cgo: flags for C preprocessor
	CgoCXXFLAGS  []string `json:",omitempty"` // cgo: flags for C++ compiler
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
	forceBuild   bool                 // this package must be rebuilt
	forceLibrary bool                 // this package is a library (even if named "main")
	cmdline      bool                 // defined by files listed on command line
	local        bool                 // imported via local path (./ or ../)
	localPrefix  string               // interpret ./ and ../ imports relative to this prefix
	exeName      string               // desired name for temporary executable
	coverMode    string               // preprocess Go source files with the coverage tool in this mode
	coverVars    map[string]*CoverVar // variables created by coverage analysis
	omitDWARF    bool                 // tell linker not to write DWARF information
	buildID      string               // expected build ID for generated package
}

// CoverVar holds the name of the generated coverage variables targeting the named file.
type CoverVar struct {
	File string // local file name
	Var  string // name of count struct
}

func (p *Package) copyBuild(pp *build.Package) {
	p.build = pp

	p.Dir = pp.Dir
	p.ImportPath = pp.ImportPath
	p.ImportComment = pp.ImportComment
	p.Name = pp.Name
	p.Doc = pp.Doc
	p.Root = pp.Root
	p.ConflictDir = pp.ConflictDir
	// TODO? Target
	p.Goroot = pp.Goroot
	p.Standard = p.Goroot && p.ImportPath != "" && !strings.Contains(p.ImportPath, ".")
	p.GoFiles = pp.GoFiles
	p.CgoFiles = pp.CgoFiles
	p.IgnoredGoFiles = pp.IgnoredGoFiles
	p.CFiles = pp.CFiles
	p.CXXFiles = pp.CXXFiles
	p.MFiles = pp.MFiles
	p.HFiles = pp.HFiles
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
	// Should match Go spec, compilers, and ../../go/parser/parser.go:/isValidImport.
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
		if perr := disallowInternal(srcDir, p, stk); perr != p {
			return perr
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
	bp, err := buildContext.Import(path, srcDir, build.ImportComment)
	bp.ImportPath = importPath
	if gobin != "" {
		bp.BinDir = gobin
	}
	if err == nil && !isLocal && bp.ImportComment != "" && bp.ImportComment != path {
		err = fmt.Errorf("code in directory %s expects import %q", bp.Dir, bp.ImportComment)
	}
	p.load(stk, bp, err)
	if p.Error != nil && len(importPos) > 0 {
		pos := importPos[0]
		pos.Filename = shortPath(pos.Filename)
		p.Error.Pos = pos.String()
	}

	if perr := disallowInternal(srcDir, p, stk); perr != p {
		return perr
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

	// The stack includes p.ImportPath.
	// If that's the only thing on the stack, we started
	// with a name given on the command line, not an
	// import. Anything listed on the command line is fine.
	if len(*stk) == 1 {
		return p
	}

	// Check for "internal" element: four cases depending on begin of string and/or end of string.
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
	if hasPathPrefix(filepath.ToSlash(srcDir), filepath.ToSlash(parent)) {
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
	// Four cases, depending on internal at start/end of string or not.
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

type targetDir int

const (
	toRoot    targetDir = iota // to bin dir inside package root (default)
	toTool                     // GOROOT/pkg/tool
	toBin                      // GOROOT/bin
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
	"cmd/old5a":                            toTool,
	"cmd/old6a":                            toTool,
	"cmd/old8a":                            toTool,
	"cmd/old9a":                            toTool,
	"cmd/pack":                             toTool,
	"cmd/pprof":                            toTool,
	"cmd/trace":                            toTool,
	"cmd/yacc":                             toTool,
	"golang.org/x/tools/cmd/godoc":         toBin,
	"golang.org/x/tools/cmd/vet":           toTool,
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

var raceExclude = map[string]bool{
	"runtime/race": true,
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
		if p.build.BinDir != gobin && goTools[p.ImportPath] == toBin {
			// Override BinDir.
			// This is from a subrepo but installs to $GOROOT/bin
			// by default anyway (like godoc).
			p.target = filepath.Join(gorootBin, elem)
		} else if p.build.BinDir != "" {
			// Install to GOBIN or bin of GOPATH entry.
			p.target = filepath.Join(p.build.BinDir, elem)
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
				p.Shlib = filepath.Join(p.build.PkgTargetRoot, libname)
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

	// Currently build mode c-shared, or -linkshared, forces
	// external linking mode, and external linking mode forces an
	// import of runtime/cgo.
	if p.Name == "main" && !p.Goroot && (buildBuildmode == "c-shared" || buildLinkshared) {
		importPaths = append(importPaths, "runtime/cgo")
	}

	// Everything depends on runtime, except runtime and unsafe.
	if !p.Standard || (p.ImportPath != "runtime" && p.ImportPath != "unsafe") {
		importPaths = append(importPaths, "runtime")
		// When race detection enabled everything depends on runtime/race.
		// Exclude certain packages to avoid circular dependencies.
		if buildRace && (!p.Standard || !raceExclude[p.ImportPath]) {
			importPaths = append(importPaths, "runtime/race")
		}
		// On ARM with GOARM=5, everything depends on math for the link.
		if p.ImportPath == "main" && goarch == "arm" {
			importPaths = append(importPaths, "math")
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
		p.CXXFiles,
		p.MFiles,
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
	deps := make(map[string]*Package)
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
		deps[path] = p1
		imports = append(imports, p1)
		for _, dep := range p1.deps {
			deps[dep.ImportPath] = dep
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

	// Compute build ID for this package.
	// Build ID is hash of information we want to detect changes in.
	// The mtime-based checks in computeStale take care of most
	// of that information, but they cannot detect the removal of a
	// source file from a directory (with no changes to files that remain
	// and no new files in that directory). We hash the list of source
	// files (without full path, to allow moving the entire tree)
	// so that if one is removed, we detect it via the build IDs.
	// In the future we might include other relevant information,
	// like build tags or whether we're using the race detector or
	// (if it becomes cheap enough) file contents.
	h := sha1.New()
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
	fmt.Fprintf(h, "%d files\n", len(inputFiles))
	for _, file := range inputFiles {
		fmt.Fprintf(h, "%s\n", file)
	}
	p.buildID = fmt.Sprintf("%x", h.Sum(nil))

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
		p.Stale = isStale(p)
	}
}

// The runtime version string takes one of two forms:
// "go1.X[.Y]" for Go releases, and "devel +hash" at tip.
// Determine whether we are in a released copy by
// inspecting the version.
var isGoRelease = strings.HasPrefix(runtime.Version(), "go1")

// isStale reports whether package p needs to be rebuilt.
func isStale(p *Package) bool {
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

	// If there's no install target or it's already marked stale, we have to rebuild.
	if p.target == "" || p.Stale {
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

	// If we are running a release copy of Go, do not rebuild the standard packages.
	// They may not be writable anyway, but they are certainly not changing.
	// This makes 'go build' and 'go build -a' skip the standard packages when
	// using an official release. See issue 3036, issue 3149, issue 4106, issue 8290.
	// (If a change to a release tree must be made by hand, the way to force the
	// install is to run make.bash, which will remove the old package archives
	// before rebuilding.)
	if p.Standard && isGoRelease {
		return false
	}

	// If the -a flag is given, rebuild everything (except standard packages; see above).
	if buildA {
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

	// Package is stale if the expected build ID differs from the
	// recorded build ID. This catches changes like a source file
	// being removed from a package directory. See issue 3895.
	targetBuildID, err := readBuildID(p)
	if err == nil && targetBuildID != p.buildID {
		return true
	}

	// As a courtesy to developers installing new versions of the compiler
	// frequently, define that packages are stale if they are
	// older than the compiler, and commands if they are older than
	// the linker.  This heuristic will not work if the binaries are
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
			return true
		}
		if p.build.IsCommand() && olderThan(buildToolchain.linker()) {
			return true
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

	srcs := stringList(p.GoFiles, p.CFiles, p.CXXFiles, p.MFiles, p.HFiles, p.SFiles, p.CgoFiles, p.SysoFiles, p.SwigFiles, p.SwigCXXFiles)
	for _, src := range srcs {
		if olderThan(filepath.Join(p.Dir, src)) {
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
		return readBuildIDFromBinary(p.Target)
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

	elfPrefix = []byte("ELF\x7F")
)

// readBuildIDFromBinary reads the build ID from a binary.
//
// ELF binaries store the build ID in a proper PT_NOTE section.
//
// Other binary formats are not so flexible. For those, the linker
// stores the build ID as non-instruction bytes at the very beginning
// of the text segment, which should appear near the beginning
// of the file. This is clumsy but fairly portable. Custom locations
// can be added for other binary types as needed, like we did for ELF.
func readBuildIDFromBinary(filename string) (id string, err error) {
	if filename == "" {
		return "", &os.PathError{Op: "parse", Path: filename, Err: errBuildIDUnknown}
	}

	// Read the first 8 kB of the binary file.
	// That should be enough to find the build ID.
	// In ELF files, the build ID is in the leading headers,
	// which are typically less than 4 kB, not to mention 8 kB.
	// On other systems, we're trying to read enough that
	// we get the beginning of the text segment in the read.
	// The offset where the text segment begins in a hello
	// world compiled for each different object format today:
	//
	//	Plan 9: 0x20
	//	Windows: 0x600
	//	Mach-O: 0x1000
	//
	f, err := os.Open(filename)
	if err != nil {
		return "", err
	}
	defer f.Close()

	data := make([]byte, 8192)
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
