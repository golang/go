// Copyright 2011 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

// Package load loads packages.
package load

import (
	"bytes"
	"fmt"
	"go/build"
	"go/token"
	"io/ioutil"
	"os"
	pathpkg "path"
	"path/filepath"
	"sort"
	"strconv"
	"strings"
	"unicode"
	"unicode/utf8"

	"cmd/go/internal/base"
	"cmd/go/internal/cfg"
	"cmd/go/internal/modinfo"
	"cmd/go/internal/search"
	"cmd/go/internal/str"
)

var (
	// module initialization hook; never nil, no-op if module use is disabled
	ModInit func()

	// module hooks; nil if module use is disabled
	ModBinDir            func() string                                       // return effective bin directory
	ModLookup            func(path string) (dir, realPath string, err error) // lookup effective meaning of import
	ModPackageModuleInfo func(path string) *modinfo.ModulePublic             // return module info for Package struct
	ModImportPaths       func(args []string) []*search.Match                 // expand import paths
	ModPackageBuildInfo  func(main string, deps []string) string             // return module info to embed in binary
	ModInfoProg          func(info string) []byte                            // wrap module info in .go code for binary
	ModImportFromFiles   func([]string)                                      // update go.mod to add modules for imports in these files
	ModDirImportPath     func(string) string                                 // return effective import path for directory
)

var IgnoreImports bool // control whether we ignore imports in packages

// A Package describes a single package found in a directory.
type Package struct {
	PackagePublic                 // visible in 'go list'
	Internal      PackageInternal // for use inside go command only
}

type PackagePublic struct {
	// Note: These fields are part of the go command's public API.
	// See list.go. It is okay to add fields, but not to change or
	// remove existing ones. Keep in sync with list.go
	Dir           string                `json:",omitempty"` // directory containing package sources
	ImportPath    string                `json:",omitempty"` // import path of package in dir
	ImportComment string                `json:",omitempty"` // path in import comment on package statement
	Name          string                `json:",omitempty"` // package name
	Doc           string                `json:",omitempty"` // package documentation string
	Target        string                `json:",omitempty"` // installed target for this package (may be executable)
	Shlib         string                `json:",omitempty"` // the shared library that contains this package (only set when -linkshared)
	Root          string                `json:",omitempty"` // Go root or Go path dir containing this package
	ConflictDir   string                `json:",omitempty"` // Dir is hidden by this other directory
	ForTest       string                `json:",omitempty"` // package is only for use in named test
	Export        string                `json:",omitempty"` // file containing export data (set by go list -export)
	Module        *modinfo.ModulePublic `json:",omitempty"` // info about package's module, if any
	Match         []string              `json:",omitempty"` // command-line patterns matching this package
	Goroot        bool                  `json:",omitempty"` // is this package found in the Go root?
	Standard      bool                  `json:",omitempty"` // is this package part of the standard Go library?
	DepOnly       bool                  `json:",omitempty"` // package is only as a dependency, not explicitly listed
	BinaryOnly    bool                  `json:",omitempty"` // package cannot be recompiled
	Incomplete    bool                  `json:",omitempty"` // was there an error loading this package or dependencies?

	// Stale and StaleReason remain here *only* for the list command.
	// They are only initialized in preparation for list execution.
	// The regular build determines staleness on the fly during action execution.
	Stale       bool   `json:",omitempty"` // would 'go install' do anything for this package?
	StaleReason string `json:",omitempty"` // why is Stale true?

	// Source files
	// If you add to this list you MUST add to p.AllFiles (below) too.
	// Otherwise file name security lists will not apply to any new additions.
	GoFiles         []string `json:",omitempty"` // .go source files (excluding CgoFiles, TestGoFiles, XTestGoFiles)
	CgoFiles        []string `json:",omitempty"` // .go source files that import "C"
	CompiledGoFiles []string `json:",omitempty"` // .go output from running cgo on CgoFiles
	IgnoredGoFiles  []string `json:",omitempty"` // .go source files ignored due to build constraints
	CFiles          []string `json:",omitempty"` // .c source files
	CXXFiles        []string `json:",omitempty"` // .cc, .cpp and .cxx source files
	MFiles          []string `json:",omitempty"` // .m source files
	HFiles          []string `json:",omitempty"` // .h, .hh, .hpp and .hxx source files
	FFiles          []string `json:",omitempty"` // .f, .F, .for and .f90 Fortran source files
	SFiles          []string `json:",omitempty"` // .s source files
	SwigFiles       []string `json:",omitempty"` // .swig files
	SwigCXXFiles    []string `json:",omitempty"` // .swigcxx files
	SysoFiles       []string `json:",omitempty"` // .syso system object files added to package

	// Cgo directives
	CgoCFLAGS    []string `json:",omitempty"` // cgo: flags for C compiler
	CgoCPPFLAGS  []string `json:",omitempty"` // cgo: flags for C preprocessor
	CgoCXXFLAGS  []string `json:",omitempty"` // cgo: flags for C++ compiler
	CgoFFLAGS    []string `json:",omitempty"` // cgo: flags for Fortran compiler
	CgoLDFLAGS   []string `json:",omitempty"` // cgo: flags for linker
	CgoPkgConfig []string `json:",omitempty"` // cgo: pkg-config names

	// Dependency information
	Imports   []string          `json:",omitempty"` // import paths used by this package
	ImportMap map[string]string `json:",omitempty"` // map from source import to ImportPath (identity entries omitted)
	Deps      []string          `json:",omitempty"` // all (recursively) imported dependencies

	// Error information
	// Incomplete is above, packed into the other bools
	Error      *PackageError   `json:",omitempty"` // error loading this package (not dependencies)
	DepsErrors []*PackageError `json:",omitempty"` // errors loading dependencies

	// Test information
	// If you add to this list you MUST add to p.AllFiles (below) too.
	// Otherwise file name security lists will not apply to any new additions.
	TestGoFiles  []string `json:",omitempty"` // _test.go files in package
	TestImports  []string `json:",omitempty"` // imports from TestGoFiles
	XTestGoFiles []string `json:",omitempty"` // _test.go files outside package
	XTestImports []string `json:",omitempty"` // imports from XTestGoFiles
}

// AllFiles returns the names of all the files considered for the package.
// This is used for sanity and security checks, so we include all files,
// even IgnoredGoFiles, because some subcommands consider them.
// The go/build package filtered others out (like foo_wrongGOARCH.s)
// and that's OK.
func (p *Package) AllFiles() []string {
	return str.StringList(
		p.GoFiles,
		p.CgoFiles,
		// no p.CompiledGoFiles, because they are from GoFiles or generated by us
		p.IgnoredGoFiles,
		p.CFiles,
		p.CXXFiles,
		p.MFiles,
		p.HFiles,
		p.FFiles,
		p.SFiles,
		p.SwigFiles,
		p.SwigCXXFiles,
		p.SysoFiles,
		p.TestGoFiles,
		p.XTestGoFiles,
	)
}

// Desc returns the package "description", for use in b.showOutput.
func (p *Package) Desc() string {
	if p.ForTest != "" {
		return p.ImportPath + " [" + p.ForTest + ".test]"
	}
	return p.ImportPath
}

type PackageInternal struct {
	// Unexported fields are not part of the public API.
	Build             *build.Package
	Imports           []*Package           // this package's direct imports
	CompiledImports   []string             // additional Imports necessary when using CompiledGoFiles (all from standard library)
	RawImports        []string             // this package's original imports as they appear in the text of the program
	ForceLibrary      bool                 // this package is a library (even if named "main")
	CmdlineFiles      bool                 // package built from files listed on command line
	CmdlinePkg        bool                 // package listed on command line
	CmdlinePkgLiteral bool                 // package listed as literal on command line (not via wildcard)
	Local             bool                 // imported via local path (./ or ../)
	LocalPrefix       string               // interpret ./ and ../ imports relative to this prefix
	ExeName           string               // desired name for temporary executable
	CoverMode         string               // preprocess Go source files with the coverage tool in this mode
	CoverVars         map[string]*CoverVar // variables created by coverage analysis
	OmitDebug         bool                 // tell linker not to write debug information
	GobinSubdir       bool                 // install target would be subdir of GOBIN
	BuildInfo         string               // add this info to package main
	TestmainGo        *[]byte              // content for _testmain.go

	Asmflags   []string // -asmflags for this package
	Gcflags    []string // -gcflags for this package
	Ldflags    []string // -ldflags for this package
	Gccgoflags []string // -gccgoflags for this package
}

type NoGoError struct {
	Package *Package
}

func (e *NoGoError) Error() string {
	// Count files beginning with _ and ., which we will pretend don't exist at all.
	dummy := 0
	for _, name := range e.Package.IgnoredGoFiles {
		if strings.HasPrefix(name, "_") || strings.HasPrefix(name, ".") {
			dummy++
		}
	}

	if len(e.Package.IgnoredGoFiles) > dummy {
		// Go files exist, but they were ignored due to build constraints.
		return "build constraints exclude all Go files in " + e.Package.Dir
	}
	if len(e.Package.TestGoFiles)+len(e.Package.XTestGoFiles) > 0 {
		// Test Go files exist, but we're not interested in them.
		// The double-negative is unfortunate but we want e.Package.Dir
		// to appear at the end of error message.
		return "no non-test Go files in " + e.Package.Dir
	}
	return "no Go files in " + e.Package.Dir
}

// Resolve returns the resolved version of imports,
// which should be p.TestImports or p.XTestImports, NOT p.Imports.
// The imports in p.TestImports and p.XTestImports are not recursively
// loaded during the initial load of p, so they list the imports found in
// the source file, but most processing should be over the vendor-resolved
// import paths. We do this resolution lazily both to avoid file system work
// and because the eventual real load of the test imports (during 'go test')
// can produce better error messages if it starts with the original paths.
// The initial load of p loads all the non-test imports and rewrites
// the vendored paths, so nothing should ever call p.vendored(p.Imports).
func (p *Package) Resolve(imports []string) []string {
	if len(imports) > 0 && len(p.Imports) > 0 && &imports[0] == &p.Imports[0] {
		panic("internal error: p.Resolve(p.Imports) called")
	}
	seen := make(map[string]bool)
	var all []string
	for _, path := range imports {
		path = ResolveImportPath(p, path)
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
	p.Internal.Build = pp

	if pp.PkgTargetRoot != "" && cfg.BuildPkgdir != "" {
		old := pp.PkgTargetRoot
		pp.PkgRoot = cfg.BuildPkgdir
		pp.PkgTargetRoot = cfg.BuildPkgdir
		pp.PkgObj = filepath.Join(cfg.BuildPkgdir, strings.TrimPrefix(pp.PkgObj, old))
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
	p.Standard = p.Goroot && p.ImportPath != "" && search.IsStandardImportPath(p.ImportPath)
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
	p.CgoFFLAGS = pp.CgoFFLAGS
	p.CgoLDFLAGS = pp.CgoLDFLAGS
	p.CgoPkgConfig = pp.CgoPkgConfig
	// We modify p.Imports in place, so make copy now.
	p.Imports = make([]string, len(pp.Imports))
	copy(p.Imports, pp.Imports)
	p.Internal.RawImports = pp.Imports
	p.TestGoFiles = pp.TestGoFiles
	p.TestImports = pp.TestImports
	p.XTestGoFiles = pp.XTestGoFiles
	p.XTestImports = pp.XTestImports
	if IgnoreImports {
		p.Imports = nil
		p.Internal.RawImports = nil
		p.TestImports = nil
		p.XTestImports = nil
	}
}

// A PackageError describes an error loading information about a package.
type PackageError struct {
	ImportStack   []string // shortest path from package named on command line to this one
	Pos           string   // position of error
	Err           string   // the error itself
	IsImportCycle bool     `json:"-"` // the error is an import cycle
	Hard          bool     `json:"-"` // whether the error is soft or hard; soft errors are ignored in some places
}

func (p *PackageError) Error() string {
	// Import cycles deserve special treatment.
	if p.IsImportCycle {
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

// An ImportStack is a stack of import paths, possibly with the suffix " (test)" appended.
// The import path of a test package is the import path of the corresponding
// non-test package with the suffix "_test" added.
type ImportStack []string

func (s *ImportStack) Push(p string) {
	*s = append(*s, p)
}

func (s *ImportStack) Pop() {
	*s = (*s)[0 : len(*s)-1]
}

func (s *ImportStack) Copy() []string {
	return append([]string{}, *s...)
}

// shorterThan reports whether sp is shorter than t.
// We use this to record the shortest import sequence
// that leads to a particular package.
func (sp *ImportStack) shorterThan(t []string) bool {
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

func ClearPackageCache() {
	for name := range packageCache {
		delete(packageCache, name)
	}
}

func ClearPackageCachePartial(args []string) {
	for _, arg := range args {
		p := packageCache[arg]
		if p != nil {
			delete(packageCache, p.Dir)
			delete(packageCache, p.ImportPath)
		}
	}
}

// ReloadPackageNoFlags is like LoadPackageNoFlags but makes sure
// not to use the package cache.
// It is only for use by GOPATH-based "go get".
// TODO(rsc): When GOPATH-based "go get" is removed, delete this function.
func ReloadPackageNoFlags(arg string, stk *ImportStack) *Package {
	p := packageCache[arg]
	if p != nil {
		delete(packageCache, p.Dir)
		delete(packageCache, p.ImportPath)
	}
	return LoadPackageNoFlags(arg, stk)
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
	// ResolveImport means that loadImport should do import path expansion.
	// That is, ResolveImport means that the import path came from
	// a source file and has not been expanded yet to account for
	// vendoring or possible module adjustment.
	// Every import path should be loaded initially with ResolveImport,
	// and then the expanded version (for example with the /vendor/ in it)
	// gets recorded as the canonical import path. At that point, future loads
	// of that package must not pass ResolveImport, because
	// disallowVendor will reject direct use of paths containing /vendor/.
	ResolveImport = 1 << iota

	// ResolveModule is for download (part of "go get") and indicates
	// that the module adjustment should be done, but not vendor adjustment.
	ResolveModule

	// GetTestDeps is for download (part of "go get") and indicates
	// that test dependencies should be fetched too.
	GetTestDeps
)

// LoadImport scans the directory named by path, which must be an import path,
// but possibly a local import path (an absolute file system path or one beginning
// with ./ or ../). A local relative path is interpreted relative to srcDir.
// It returns a *Package describing the package found in that directory.
// LoadImport does not set tool flags and should only be used by
// this package, as part of a bigger load operation, and by GOPATH-based "go get".
// TODO(rsc): When GOPATH-based "go get" is removed, unexport this function.
func LoadImport(path, srcDir string, parent *Package, stk *ImportStack, importPos []token.Position, mode int) *Package {
	stk.Push(path)
	defer stk.Pop()

	if strings.HasPrefix(path, "mod/") {
		// Paths beginning with "mod/" might accidentally
		// look in the module cache directory tree in $GOPATH/pkg/mod/.
		// This prefix is owned by the Go core for possible use in the
		// standard library (since it does not begin with a domain name),
		// so it's OK to disallow entirely.
		return &Package{
			PackagePublic: PackagePublic{
				ImportPath: path,
				Error: &PackageError{
					ImportStack: stk.Copy(),
					Err:         fmt.Sprintf("disallowed import path %q", path),
				},
			},
		}
	}

	if strings.Contains(path, "@") {
		var text string
		if cfg.ModulesEnabled {
			text = "can only use path@version syntax with 'go get'"
		} else {
			text = "cannot use path@version syntax in GOPATH mode"
		}
		return &Package{
			PackagePublic: PackagePublic{
				ImportPath: path,
				Error: &PackageError{
					ImportStack: stk.Copy(),
					Err:         text,
				},
			},
		}
	}

	parentPath := ""
	if parent != nil {
		parentPath = parent.ImportPath
	}

	// Determine canonical identifier for this package.
	// For a local import the identifier is the pseudo-import path
	// we create from the full directory to the package.
	// Otherwise it is the usual import path.
	// For vendored imports, it is the expanded form.
	importPath := path
	origPath := path
	isLocal := build.IsLocalImport(path)
	var modDir string
	var modErr error
	if isLocal {
		importPath = dirToImportPath(filepath.Join(srcDir, path))
	} else if cfg.ModulesEnabled {
		var p string
		modDir, p, modErr = ModLookup(path)
		if modErr == nil {
			importPath = p
		}
	} else if mode&ResolveImport != 0 {
		// We do our own path resolution, because we want to
		// find out the key to use in packageCache without the
		// overhead of repeated calls to buildContext.Import.
		// The code is also needed in a few other places anyway.
		path = ResolveImportPath(parent, path)
		importPath = path
	} else if mode&ResolveModule != 0 {
		path = ModuleImportPath(parent, path)
		importPath = path
	}

	p := packageCache[importPath]
	if p != nil {
		p = reusePackage(p, stk)
	} else {
		p = new(Package)
		p.Internal.Local = isLocal
		p.ImportPath = importPath
		packageCache[importPath] = p

		// Load package.
		// Import always returns bp != nil, even if an error occurs,
		// in order to return partial information.
		var bp *build.Package
		var err error
		if modDir != "" {
			bp, err = cfg.BuildContext.ImportDir(modDir, 0)
		} else if modErr != nil {
			bp = new(build.Package)
			err = fmt.Errorf("unknown import path %q: %v", importPath, modErr)
		} else if cfg.ModulesEnabled && path != "unsafe" {
			bp = new(build.Package)
			err = fmt.Errorf("unknown import path %q: internal error: module loader did not resolve import", importPath)
		} else {
			buildMode := build.ImportComment
			if mode&ResolveImport == 0 || path != origPath {
				// Not vendoring, or we already found the vendored path.
				buildMode |= build.IgnoreVendor
			}
			bp, err = cfg.BuildContext.Import(path, srcDir, buildMode)
		}
		bp.ImportPath = importPath
		if cfg.GOBIN != "" {
			bp.BinDir = cfg.GOBIN
		} else if cfg.ModulesEnabled {
			bp.BinDir = ModBinDir()
		}
		if modDir == "" && err == nil && !isLocal && bp.ImportComment != "" && bp.ImportComment != path &&
			!strings.Contains(path, "/vendor/") && !strings.HasPrefix(path, "vendor/") {
			err = fmt.Errorf("code in directory %s expects import %q", bp.Dir, bp.ImportComment)
		}
		p.load(stk, bp, err)
		if p.Error != nil && p.Error.Pos == "" {
			p = setErrorPos(p, importPos)
		}

		if modDir == "" && origPath != cleanImport(origPath) {
			p.Error = &PackageError{
				ImportStack: stk.Copy(),
				Err:         fmt.Sprintf("non-canonical import path: %q should be %q", origPath, pathpkg.Clean(origPath)),
			}
			p.Incomplete = true
		}
	}

	// Checked on every import because the rules depend on the code doing the importing.
	if perr := disallowInternal(srcDir, parent, parentPath, p, stk); perr != p {
		return setErrorPos(perr, importPos)
	}
	if mode&ResolveImport != 0 {
		if perr := disallowVendor(srcDir, parent, parentPath, origPath, p, stk); perr != p {
			return setErrorPos(perr, importPos)
		}
	}

	if p.Name == "main" && parent != nil && parent.Dir != p.Dir {
		perr := *p
		perr.Error = &PackageError{
			ImportStack: stk.Copy(),
			Err:         fmt.Sprintf("import %q is a program, not an importable package", path),
		}
		return setErrorPos(&perr, importPos)
	}

	if p.Internal.Local && parent != nil && !parent.Internal.Local {
		perr := *p
		perr.Error = &PackageError{
			ImportStack: stk.Copy(),
			Err:         fmt.Sprintf("local import %q in non-local package", path),
		}
		return setErrorPos(&perr, importPos)
	}

	return p
}

func setErrorPos(p *Package, importPos []token.Position) *Package {
	if len(importPos) > 0 {
		pos := importPos[0]
		pos.Filename = base.ShortPath(pos.Filename)
		p.Error.Pos = pos.String()
	}
	return p
}

func cleanImport(path string) string {
	orig := path
	path = pathpkg.Clean(path)
	if strings.HasPrefix(orig, "./") && path != ".." && !strings.HasPrefix(path, "../") {
		path = "./" + path
	}
	return path
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

// ResolveImportPath returns the true meaning of path when it appears in parent.
// There are two different resolutions applied.
// First, there is Go 1.5 vendoring (golang.org/s/go15vendor).
// If vendor expansion doesn't trigger, then the path is also subject to
// Go 1.11 module legacy conversion (golang.org/issue/25069).
func ResolveImportPath(parent *Package, path string) (found string) {
	if cfg.ModulesEnabled {
		if _, p, e := ModLookup(path); e == nil {
			return p
		}
		return path
	}
	found = VendoredImportPath(parent, path)
	if found != path {
		return found
	}
	return ModuleImportPath(parent, path)
}

// dirAndRoot returns the source directory and workspace root
// for the package p, guaranteeing that root is a path prefix of dir.
func dirAndRoot(p *Package) (dir, root string) {
	dir = filepath.Clean(p.Dir)
	root = filepath.Join(p.Root, "src")
	if !str.HasFilePathPrefix(dir, root) || p.ImportPath != "command-line-arguments" && filepath.Join(root, p.ImportPath) != dir {
		// Look for symlinks before reporting error.
		dir = expandPath(dir)
		root = expandPath(root)
	}

	if !str.HasFilePathPrefix(dir, root) || len(dir) <= len(root) || dir[len(root)] != filepath.Separator || p.ImportPath != "command-line-arguments" && !p.Internal.Local && filepath.Join(root, p.ImportPath) != dir {
		base.Fatalf("unexpected directory layout:\n"+
			"	import path: %s\n"+
			"	root: %s\n"+
			"	dir: %s\n"+
			"	expand root: %s\n"+
			"	expand dir: %s\n"+
			"	separator: %s",
			p.ImportPath,
			filepath.Join(p.Root, "src"),
			filepath.Clean(p.Dir),
			root,
			dir,
			string(filepath.Separator))
	}

	return dir, root
}

// VendoredImportPath returns the vendor-expansion of path when it appears in parent.
// If parent is x/y/z, then path might expand to x/y/z/vendor/path, x/y/vendor/path,
// x/vendor/path, vendor/path, or else stay path if none of those exist.
// VendoredImportPath returns the expanded path or, if no expansion is found, the original.
func VendoredImportPath(parent *Package, path string) (found string) {
	if parent == nil || parent.Root == "" {
		return path
	}

	dir, root := dirAndRoot(parent)

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

var (
	modulePrefix   = []byte("\nmodule ")
	goModPathCache = make(map[string]string)
)

// goModPath returns the module path in the go.mod in dir, if any.
func goModPath(dir string) (path string) {
	path, ok := goModPathCache[dir]
	if ok {
		return path
	}
	defer func() {
		goModPathCache[dir] = path
	}()

	data, err := ioutil.ReadFile(filepath.Join(dir, "go.mod"))
	if err != nil {
		return ""
	}
	var i int
	if bytes.HasPrefix(data, modulePrefix[1:]) {
		i = 0
	} else {
		i = bytes.Index(data, modulePrefix)
		if i < 0 {
			return ""
		}
		i++
	}
	line := data[i:]

	// Cut line at \n, drop trailing \r if present.
	if j := bytes.IndexByte(line, '\n'); j >= 0 {
		line = line[:j]
	}
	if line[len(line)-1] == '\r' {
		line = line[:len(line)-1]
	}
	line = line[len("module "):]

	// If quoted, unquote.
	path = strings.TrimSpace(string(line))
	if path != "" && path[0] == '"' {
		s, err := strconv.Unquote(path)
		if err != nil {
			return ""
		}
		path = s
	}
	return path
}

// findVersionElement returns the slice indices of the final version element /vN in path.
// If there is no such element, it returns -1, -1.
func findVersionElement(path string) (i, j int) {
	j = len(path)
	for i = len(path) - 1; i >= 0; i-- {
		if path[i] == '/' {
			if isVersionElement(path[i:j]) {
				return i, j
			}
			j = i
		}
	}
	return -1, -1
}

// isVersionElement reports whether s is a well-formed path version element:
// v2, v3, v10, etc, but not v0, v05, v1.
func isVersionElement(s string) bool {
	if len(s) < 3 || s[0] != '/' || s[1] != 'v' || s[2] == '0' || s[2] == '1' && len(s) == 3 {
		return false
	}
	for i := 2; i < len(s); i++ {
		if s[i] < '0' || '9' < s[i] {
			return false
		}
	}
	return true
}

// ModuleImportPath translates import paths found in go modules
// back down to paths that can be resolved in ordinary builds.
//
// Define “new” code as code with a go.mod file in the same directory
// or a parent directory. If an import in new code says x/y/v2/z but
// x/y/v2/z does not exist and x/y/go.mod says “module x/y/v2”,
// then go build will read the import as x/y/z instead.
// See golang.org/issue/25069.
func ModuleImportPath(parent *Package, path string) (found string) {
	if parent == nil || parent.Root == "" {
		return path
	}

	// If there are no vN elements in path, leave it alone.
	// (The code below would do the same, but only after
	// some other file system accesses that we can avoid
	// here by returning early.)
	if i, _ := findVersionElement(path); i < 0 {
		return path
	}

	dir, root := dirAndRoot(parent)

	// Consider dir and parents, up to and including root.
	for i := len(dir); i >= len(root); i-- {
		if i < len(dir) && dir[i] != filepath.Separator {
			continue
		}
		if goModPath(dir[:i]) != "" {
			goto HaveGoMod
		}
	}
	// This code is not in a tree with a go.mod,
	// so apply no changes to the path.
	return path

HaveGoMod:
	// This import is in a tree with a go.mod.
	// Allow it to refer to code in GOPATH/src/x/y/z as x/y/v2/z
	// if GOPATH/src/x/y/go.mod says module "x/y/v2",

	// If x/y/v2/z exists, use it unmodified.
	if bp, _ := cfg.BuildContext.Import(path, "", build.IgnoreVendor); bp.Dir != "" {
		return path
	}

	// Otherwise look for a go.mod supplying a version element.
	// Some version-like elements may appear in paths but not
	// be module versions; we skip over those to look for module
	// versions. For example the module m/v2 might have a
	// package m/v2/api/v1/foo.
	limit := len(path)
	for limit > 0 {
		i, j := findVersionElement(path[:limit])
		if i < 0 {
			return path
		}
		if bp, _ := cfg.BuildContext.Import(path[:i], "", build.IgnoreVendor); bp.Dir != "" {
			if mpath := goModPath(bp.Dir); mpath != "" {
				// Found a valid go.mod file, so we're stopping the search.
				// If the path is m/v2/p and we found m/go.mod that says
				// "module m/v2", then we return "m/p".
				if mpath == path[:j] {
					return path[:i] + path[j:]
				}
				// Otherwise just return the original path.
				// We didn't find anything worth rewriting,
				// and the go.mod indicates that we should
				// not consider parent directories.
				return path
			}
		}
		limit = i
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
func reusePackage(p *Package, stk *ImportStack) *Package {
	// We use p.Internal.Imports==nil to detect a package that
	// is in the midst of its own loadPackage call
	// (all the recursion below happens before p.Internal.Imports gets set).
	if p.Internal.Imports == nil {
		if p.Error == nil {
			p.Error = &PackageError{
				ImportStack:   stk.Copy(),
				Err:           "import cycle not allowed",
				IsImportCycle: true,
			}
		}
		p.Incomplete = true
	}
	// Don't rewrite the import stack in the error if we have an import cycle.
	// If we do, we'll lose the path that describes the cycle.
	if p.Error != nil && !p.Error.IsImportCycle && stk.shorterThan(p.Error.ImportStack) {
		p.Error.ImportStack = stk.Copy()
	}
	return p
}

// disallowInternal checks that srcDir (containing package importerPath, if non-empty)
// is allowed to import p.
// If the import is allowed, disallowInternal returns the original package p.
// If not, it returns a new package containing just an appropriate error.
func disallowInternal(srcDir string, importer *Package, importerPath string, p *Package, stk *ImportStack) *Package {
	// golang.org/s/go14internal:
	// An import of a path containing the element “internal”
	// is disallowed if the importing code is outside the tree
	// rooted at the parent of the “internal” directory.

	// There was an error loading the package; stop here.
	if p.Error != nil {
		return p
	}

	// The generated 'testmain' package is allowed to access testing/internal/...,
	// as if it were generated into the testing directory tree
	// (it's actually in a temporary directory outside any Go tree).
	// This cleans up a former kludge in passing functionality to the testing package.
	if strings.HasPrefix(p.ImportPath, "testing/internal") && len(*stk) >= 2 && (*stk)[len(*stk)-2] == "testmain" {
		return p
	}

	// We can't check standard packages with gccgo.
	if cfg.BuildContext.Compiler == "gccgo" && p.Standard {
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

	if p.Module == nil {
		parent := p.Dir[:i+len(p.Dir)-len(p.ImportPath)]

		if str.HasFilePathPrefix(filepath.Clean(srcDir), filepath.Clean(parent)) {
			return p
		}

		// Look for symlinks before reporting error.
		srcDir = expandPath(srcDir)
		parent = expandPath(parent)
		if str.HasFilePathPrefix(filepath.Clean(srcDir), filepath.Clean(parent)) {
			return p
		}
	} else {
		// p is in a module, so make it available based on the importer's import path instead
		// of the file path (https://golang.org/issue/23970).
		if importerPath == "." {
			// The importer is a list of command-line files.
			// Pretend that the import path is the import path of the
			// directory containing them.
			importerPath = ModDirImportPath(importer.Dir)
		}
		parentOfInternal := p.ImportPath[:i]
		if str.HasPathPrefix(importerPath, parentOfInternal) {
			return p
		}
	}

	// Internal is present, and srcDir is outside parent's tree. Not allowed.
	perr := *p
	perr.Error = &PackageError{
		ImportStack: stk.Copy(),
		Err:         "use of internal package " + p.ImportPath + " not allowed",
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

// disallowVendor checks that srcDir (containing package importerPath, if non-empty)
// is allowed to import p as path.
// If the import is allowed, disallowVendor returns the original package p.
// If not, it returns a new package containing just an appropriate error.
func disallowVendor(srcDir string, importer *Package, importerPath, path string, p *Package, stk *ImportStack) *Package {
	// The stack includes p.ImportPath.
	// If that's the only thing on the stack, we started
	// with a name given on the command line, not an
	// import. Anything listed on the command line is fine.
	if len(*stk) == 1 {
		return p
	}

	// Modules must not import vendor packages in the standard library,
	// but the usual vendor visibility check will not catch them
	// because the module loader presents them with an ImportPath starting
	// with "golang_org/" instead of "vendor/".
	if p.Standard && !importer.Standard && strings.HasPrefix(p.ImportPath, "golang_org") {
		perr := *p
		perr.Error = &PackageError{
			ImportStack: stk.Copy(),
			Err:         "use of vendored package " + path + " not allowed",
		}
		perr.Incomplete = true
		return &perr
	}

	if perr := disallowVendorVisibility(srcDir, p, stk); perr != p {
		return perr
	}

	// Paths like x/vendor/y must be imported as y, never as x/vendor/y.
	if i, ok := FindVendor(path); ok {
		perr := *p
		perr.Error = &PackageError{
			ImportStack: stk.Copy(),
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
func disallowVendorVisibility(srcDir string, p *Package, stk *ImportStack) *Package {
	// The stack includes p.ImportPath.
	// If that's the only thing on the stack, we started
	// with a name given on the command line, not an
	// import. Anything listed on the command line is fine.
	if len(*stk) == 1 {
		return p
	}

	// Check for "vendor" element.
	i, ok := FindVendor(p.ImportPath)
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
	if str.HasFilePathPrefix(filepath.Clean(srcDir), filepath.Clean(parent)) {
		return p
	}

	// Look for symlinks before reporting error.
	srcDir = expandPath(srcDir)
	parent = expandPath(parent)
	if str.HasFilePathPrefix(filepath.Clean(srcDir), filepath.Clean(parent)) {
		return p
	}

	// Vendor is present, and srcDir is outside parent's tree. Not allowed.
	perr := *p
	perr.Error = &PackageError{
		ImportStack: stk.Copy(),
		Err:         "use of vendored package not allowed",
	}
	perr.Incomplete = true
	return &perr
}

// FindVendor looks for the last non-terminating "vendor" path element in the given import path.
// If there isn't one, FindVendor returns ok=false.
// Otherwise, FindVendor returns ok=true and the index of the "vendor".
//
// Note that terminating "vendor" elements don't count: "x/vendor" is its own package,
// not the vendored copy of an import "" (the empty import path).
// This will allow people to have packages or commands named vendor.
// This may help reduce breakage, or it may just be confusing. We'll see.
func FindVendor(path string) (index int, ok bool) {
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

type TargetDir int

const (
	ToTool    TargetDir = iota // to GOROOT/pkg/tool (default for cmd/*)
	ToBin                      // to bin dir inside package root (default for non-cmd/*)
	StalePath                  // an old import path; fail to build
)

// InstallTargetDir reports the target directory for installing the command p.
func InstallTargetDir(p *Package) TargetDir {
	if strings.HasPrefix(p.ImportPath, "code.google.com/p/go.tools/cmd/") {
		return StalePath
	}
	if p.Goroot && strings.HasPrefix(p.ImportPath, "cmd/") && p.Name == "main" {
		switch p.ImportPath {
		case "cmd/go", "cmd/gofmt":
			return ToBin
		}
		return ToTool
	}
	return ToBin
}

var cgoExclude = map[string]bool{
	"runtime/cgo": true,
}

var cgoSyscallExclude = map[string]bool{
	"runtime/cgo":  true,
	"runtime/race": true,
	"runtime/msan": true,
}

var foldPath = make(map[string]string)

// load populates p using information from bp, err, which should
// be the result of calling build.Context.Import.
func (p *Package) load(stk *ImportStack, bp *build.Package, err error) {
	p.copyBuild(bp)

	// The localPrefix is the path we interpret ./ imports relative to.
	// Synthesized main packages sometimes override this.
	if p.Internal.Local {
		p.Internal.LocalPrefix = dirToImportPath(p.Dir)
	}

	if err != nil {
		if _, ok := err.(*build.NoGoError); ok {
			err = &NoGoError{Package: p}
		}
		p.Incomplete = true
		err = base.ExpandScanner(err)
		p.Error = &PackageError{
			ImportStack: stk.Copy(),
			Err:         err.Error(),
		}
		return
	}

	useBindir := p.Name == "main"
	if !p.Standard {
		switch cfg.BuildBuildmode {
		case "c-archive", "c-shared", "plugin":
			useBindir = false
		}
	}

	if useBindir {
		// Report an error when the old code.google.com/p/go.tools paths are used.
		if InstallTargetDir(p) == StalePath {
			newPath := strings.Replace(p.ImportPath, "code.google.com/p/go.", "golang.org/x/", 1)
			e := fmt.Sprintf("the %v command has moved; use %v instead.", p.ImportPath, newPath)
			p.Error = &PackageError{Err: e}
			return
		}
		_, elem := filepath.Split(p.Dir)
		if cfg.ModulesEnabled {
			// NOTE(rsc): Using p.ImportPath instead of p.Dir
			// makes sure we install a package in the root of a
			// cached module directory as that package name
			// not name@v1.2.3.
			// Using p.ImportPath instead of p.Dir
			// is probably correct all the time,
			// even for non-module-enabled code,
			// but I'm not brave enough to change the
			// non-module behavior this late in the
			// release cycle. Maybe for Go 1.12.
			// See golang.org/issue/26869.
			_, elem = pathpkg.Split(p.ImportPath)

			// If this is example.com/mycmd/v2, it's more useful to install it as mycmd than as v2.
			// See golang.org/issue/24667.
			isVersion := func(v string) bool {
				if len(v) < 2 || v[0] != 'v' || v[1] < '1' || '9' < v[1] {
					return false
				}
				for i := 2; i < len(v); i++ {
					if c := v[i]; c < '0' || '9' < c {
						return false
					}
				}
				return true
			}
			if isVersion(elem) {
				_, elem = pathpkg.Split(pathpkg.Dir(p.ImportPath))
			}
		}
		full := cfg.BuildContext.GOOS + "_" + cfg.BuildContext.GOARCH + "/" + elem
		if cfg.BuildContext.GOOS != base.ToolGOOS || cfg.BuildContext.GOARCH != base.ToolGOARCH {
			// Install cross-compiled binaries to subdirectories of bin.
			elem = full
		}
		if p.Internal.Build.BinDir == "" && cfg.ModulesEnabled {
			p.Internal.Build.BinDir = ModBinDir()
		}
		if p.Internal.Build.BinDir != "" {
			// Install to GOBIN or bin of GOPATH entry.
			p.Target = filepath.Join(p.Internal.Build.BinDir, elem)
			if !p.Goroot && strings.Contains(elem, "/") && cfg.GOBIN != "" {
				// Do not create $GOBIN/goos_goarch/elem.
				p.Target = ""
				p.Internal.GobinSubdir = true
			}
		}
		if InstallTargetDir(p) == ToTool {
			// This is for 'go tool'.
			// Override all the usual logic and force it into the tool directory.
			if cfg.BuildToolchainName == "gccgo" {
				p.Target = filepath.Join(base.ToolDir, elem)
			} else {
				p.Target = filepath.Join(cfg.GOROOTpkg, "tool", full)
			}
		}
		if p.Target != "" && cfg.BuildContext.GOOS == "windows" {
			p.Target += ".exe"
		}
	} else if p.Internal.Local {
		// Local import turned into absolute path.
		// No permanent install target.
		p.Target = ""
	} else {
		p.Target = p.Internal.Build.PkgObj
		if cfg.BuildLinkshared {
			shlibnamefile := p.Target[:len(p.Target)-2] + ".shlibname"
			shlib, err := ioutil.ReadFile(shlibnamefile)
			if err != nil && !os.IsNotExist(err) {
				base.Fatalf("reading shlibname: %v", err)
			}
			if err == nil {
				libname := strings.TrimSpace(string(shlib))
				if cfg.BuildContext.Compiler == "gccgo" {
					p.Shlib = filepath.Join(p.Internal.Build.PkgTargetRoot, "shlibs", libname)
				} else {
					p.Shlib = filepath.Join(p.Internal.Build.PkgTargetRoot, libname)
				}
			}
		}
	}

	// Build augmented import list to add implicit dependencies.
	// Be careful not to add imports twice, just to avoid confusion.
	importPaths := p.Imports
	addImport := func(path string, forCompiler bool) {
		for _, p := range importPaths {
			if path == p {
				return
			}
		}
		importPaths = append(importPaths, path)
		if forCompiler {
			p.Internal.CompiledImports = append(p.Internal.CompiledImports, path)
		}
	}

	// Cgo translation adds imports of "unsafe", "runtime/cgo" and "syscall",
	// except for certain packages, to avoid circular dependencies.
	if p.UsesCgo() {
		addImport("unsafe", true)
	}
	if p.UsesCgo() && (!p.Standard || !cgoExclude[p.ImportPath]) && cfg.BuildContext.Compiler != "gccgo" {
		addImport("runtime/cgo", true)
	}
	if p.UsesCgo() && (!p.Standard || !cgoSyscallExclude[p.ImportPath]) {
		addImport("syscall", true)
	}

	// SWIG adds imports of some standard packages.
	if p.UsesSwig() {
		if cfg.BuildContext.Compiler != "gccgo" {
			addImport("runtime/cgo", true)
		}
		addImport("syscall", true)
		addImport("sync", true)

		// TODO: The .swig and .swigcxx files can use
		// %go_import directives to import other packages.
	}

	// The linker loads implicit dependencies.
	if p.Name == "main" && !p.Internal.ForceLibrary {
		for _, dep := range LinkerDeps(p) {
			addImport(dep, false)
		}
	}

	// Check for case-insensitive collision of input files.
	// To avoid problems on case-insensitive files, we reject any package
	// where two different input files have equal names under a case-insensitive
	// comparison.
	inputs := p.AllFiles()
	f1, f2 := str.FoldDup(inputs)
	if f1 != "" {
		p.Error = &PackageError{
			ImportStack: stk.Copy(),
			Err:         fmt.Sprintf("case-insensitive file name collision: %q and %q", f1, f2),
		}
		return
	}

	// If first letter of input file is ASCII, it must be alphanumeric.
	// This avoids files turning into flags when invoking commands,
	// and other problems we haven't thought of yet.
	// Also, _cgo_ files must be generated by us, not supplied.
	// They are allowed to have //go:cgo_ldflag directives.
	// The directory scan ignores files beginning with _,
	// so we shouldn't see any _cgo_ files anyway, but just be safe.
	for _, file := range inputs {
		if !SafeArg(file) || strings.HasPrefix(file, "_cgo_") {
			p.Error = &PackageError{
				ImportStack: stk.Copy(),
				Err:         fmt.Sprintf("invalid input file name %q", file),
			}
			return
		}
	}
	if name := pathpkg.Base(p.ImportPath); !SafeArg(name) {
		p.Error = &PackageError{
			ImportStack: stk.Copy(),
			Err:         fmt.Sprintf("invalid input directory name %q", name),
		}
		return
	}
	if !SafeArg(p.ImportPath) {
		p.Error = &PackageError{
			ImportStack: stk.Copy(),
			Err:         fmt.Sprintf("invalid import path %q", p.ImportPath),
		}
		return
	}

	// Build list of imported packages and full dependency list.
	imports := make([]*Package, 0, len(p.Imports))
	for i, path := range importPaths {
		if path == "C" {
			continue
		}
		p1 := LoadImport(path, p.Dir, p, stk, p.Internal.Build.ImportPos[path], ResolveImport)
		if p.Standard && p.Error == nil && !p1.Standard && p1.Error == nil {
			p.Error = &PackageError{
				ImportStack: stk.Copy(),
				Err:         fmt.Sprintf("non-standard import %q in standard package %q", path, p.ImportPath),
			}
			pos := p.Internal.Build.ImportPos[path]
			if len(pos) > 0 {
				p.Error.Pos = pos[0].String()
			}
		}

		path = p1.ImportPath
		importPaths[i] = path
		if i < len(p.Imports) {
			p.Imports[i] = path
		}

		imports = append(imports, p1)
		if p1.Incomplete {
			p.Incomplete = true
		}
	}
	p.Internal.Imports = imports

	deps := make(map[string]*Package)
	var q []*Package
	q = append(q, imports...)
	for i := 0; i < len(q); i++ {
		p1 := q[i]
		path := p1.ImportPath
		// The same import path could produce an error or not,
		// depending on what tries to import it.
		// Prefer to record entries with errors, so we can report them.
		p0 := deps[path]
		if p0 == nil || p1.Error != nil && (p0.Error == nil || len(p0.Error.ImportStack) > len(p1.Error.ImportStack)) {
			deps[path] = p1
			for _, p2 := range p1.Internal.Imports {
				if deps[p2.ImportPath] != p2 {
					q = append(q, p2)
				}
			}
		}
	}

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
		if p1.Error != nil {
			p.DepsErrors = append(p.DepsErrors, p1.Error)
		}
	}

	// unsafe is a fake package.
	if p.Standard && (p.ImportPath == "unsafe" || cfg.BuildContext.Compiler == "gccgo") {
		p.Target = ""
	}

	// If cgo is not enabled, ignore cgo supporting sources
	// just as we ignore go files containing import "C".
	if !cfg.BuildContext.CgoEnabled {
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

	setError := func(msg string) {
		p.Error = &PackageError{
			ImportStack: stk.Copy(),
			Err:         msg,
		}
	}

	// The gc toolchain only permits C source files with cgo or SWIG.
	if len(p.CFiles) > 0 && !p.UsesCgo() && !p.UsesSwig() && cfg.BuildContext.Compiler == "gc" {
		setError(fmt.Sprintf("C source files not allowed when not using cgo or SWIG: %s", strings.Join(p.CFiles, " ")))
		return
	}

	// C++, Objective-C, and Fortran source files are permitted only with cgo or SWIG,
	// regardless of toolchain.
	if len(p.CXXFiles) > 0 && !p.UsesCgo() && !p.UsesSwig() {
		setError(fmt.Sprintf("C++ source files not allowed when not using cgo or SWIG: %s", strings.Join(p.CXXFiles, " ")))
		return
	}
	if len(p.MFiles) > 0 && !p.UsesCgo() && !p.UsesSwig() {
		setError(fmt.Sprintf("Objective-C source files not allowed when not using cgo or SWIG: %s", strings.Join(p.MFiles, " ")))
		return
	}
	if len(p.FFiles) > 0 && !p.UsesCgo() && !p.UsesSwig() {
		setError(fmt.Sprintf("Fortran source files not allowed when not using cgo or SWIG: %s", strings.Join(p.FFiles, " ")))
		return
	}

	// Check for case-insensitive collisions of import paths.
	fold := str.ToFold(p.ImportPath)
	if other := foldPath[fold]; other == "" {
		foldPath[fold] = p.ImportPath
	} else if other != p.ImportPath {
		setError(fmt.Sprintf("case-insensitive import collision: %q and %q", p.ImportPath, other))
		return
	}

	if cfg.ModulesEnabled {
		p.Module = ModPackageModuleInfo(p.ImportPath)
		if p.Name == "main" {
			p.Internal.BuildInfo = ModPackageBuildInfo(p.ImportPath, p.Deps)
		}
	}
}

// SafeArg reports whether arg is a "safe" command-line argument,
// meaning that when it appears in a command-line, it probably
// doesn't have some special meaning other than its own name.
// Obviously args beginning with - are not safe (they look like flags).
// Less obviously, args beginning with @ are not safe (they look like
// GNU binutils flagfile specifiers, sometimes called "response files").
// To be conservative, we reject almost any arg beginning with non-alphanumeric ASCII.
// We accept leading . _ and / as likely in file system paths.
// There is a copy of this function in cmd/compile/internal/gc/noder.go.
func SafeArg(name string) bool {
	if name == "" {
		return false
	}
	c := name[0]
	return '0' <= c && c <= '9' || 'A' <= c && c <= 'Z' || 'a' <= c && c <= 'z' || c == '.' || c == '_' || c == '/' || c >= utf8.RuneSelf
}

// LinkerDeps returns the list of linker-induced dependencies for main package p.
func LinkerDeps(p *Package) []string {
	// Everything links runtime.
	deps := []string{"runtime"}

	// External linking mode forces an import of runtime/cgo.
	if externalLinkingForced(p) && cfg.BuildContext.Compiler != "gccgo" {
		deps = append(deps, "runtime/cgo")
	}
	// On ARM with GOARM=5, it forces an import of math, for soft floating point.
	if cfg.Goarch == "arm" {
		deps = append(deps, "math")
	}
	// Using the race detector forces an import of runtime/race.
	if cfg.BuildRace {
		deps = append(deps, "runtime/race")
	}
	// Using memory sanitizer forces an import of runtime/msan.
	if cfg.BuildMSan {
		deps = append(deps, "runtime/msan")
	}

	return deps
}

// externalLinkingForced reports whether external linking is being
// forced even for programs that do not use cgo.
func externalLinkingForced(p *Package) bool {
	// Some targets must use external linking even inside GOROOT.
	switch cfg.BuildContext.GOOS {
	case "android":
		return true
	case "darwin":
		switch cfg.BuildContext.GOARCH {
		case "arm", "arm64":
			return true
		}
	}

	if !cfg.BuildContext.CgoEnabled {
		return false
	}
	// Currently build modes c-shared, pie (on systems that do not
	// support PIE with internal linking mode (currently all
	// systems: issue #18968)), plugin, and -linkshared force
	// external linking mode, as of course does
	// -ldflags=-linkmode=external. External linking mode forces
	// an import of runtime/cgo.
	pieCgo := cfg.BuildBuildmode == "pie"
	linkmodeExternal := false
	if p != nil {
		ldflags := BuildLdflags.For(p)
		for i, a := range ldflags {
			if a == "-linkmode=external" {
				linkmodeExternal = true
			}
			if a == "-linkmode" && i+1 < len(ldflags) && ldflags[i+1] == "external" {
				linkmodeExternal = true
			}
		}
	}

	return cfg.BuildBuildmode == "c-shared" || cfg.BuildBuildmode == "plugin" || pieCgo || cfg.BuildLinkshared || linkmodeExternal
}

// mkAbs rewrites list, which must be paths relative to p.Dir,
// into a sorted list of absolute paths. It edits list in place but for
// convenience also returns list back to its caller.
func (p *Package) mkAbs(list []string) []string {
	for i, f := range list {
		list[i] = filepath.Join(p.Dir, f)
	}
	sort.Strings(list)
	return list
}

// InternalGoFiles returns the list of Go files being built for the package,
// using absolute paths.
func (p *Package) InternalGoFiles() []string {
	return p.mkAbs(str.StringList(p.GoFiles, p.CgoFiles, p.TestGoFiles))
}

// InternalXGoFiles returns the list of Go files being built for the XTest package,
// using absolute paths.
func (p *Package) InternalXGoFiles() []string {
	return p.mkAbs(p.XTestGoFiles)
}

// InternalGoFiles returns the list of all Go files possibly relevant for the package,
// using absolute paths. "Possibly relevant" means that files are not excluded
// due to build tags, but files with names beginning with . or _ are still excluded.
func (p *Package) InternalAllGoFiles() []string {
	var extra []string
	for _, f := range p.IgnoredGoFiles {
		if f != "" && f[0] != '.' || f[0] != '_' {
			extra = append(extra, f)
		}
	}
	return p.mkAbs(str.StringList(extra, p.GoFiles, p.CgoFiles, p.TestGoFiles, p.XTestGoFiles))
}

// usesSwig reports whether the package needs to run SWIG.
func (p *Package) UsesSwig() bool {
	return len(p.SwigFiles) > 0 || len(p.SwigCXXFiles) > 0
}

// usesCgo reports whether the package needs to run cgo
func (p *Package) UsesCgo() bool {
	return len(p.CgoFiles) > 0
}

// PackageList returns the list of packages in the dag rooted at roots
// as visited in a depth-first post-order traversal.
func PackageList(roots []*Package) []*Package {
	seen := map[*Package]bool{}
	all := []*Package{}
	var walk func(*Package)
	walk = func(p *Package) {
		if seen[p] {
			return
		}
		seen[p] = true
		for _, p1 := range p.Internal.Imports {
			walk(p1)
		}
		all = append(all, p)
	}
	for _, root := range roots {
		walk(root)
	}
	return all
}

// TestPackageList returns the list of packages in the dag rooted at roots
// as visited in a depth-first post-order traversal, including the test
// imports of the roots. This ignores errors in test packages.
func TestPackageList(roots []*Package) []*Package {
	seen := map[*Package]bool{}
	all := []*Package{}
	var walk func(*Package)
	walk = func(p *Package) {
		if seen[p] {
			return
		}
		seen[p] = true
		for _, p1 := range p.Internal.Imports {
			walk(p1)
		}
		all = append(all, p)
	}
	walkTest := func(root *Package, path string) {
		var stk ImportStack
		p1 := LoadImport(path, root.Dir, root, &stk, root.Internal.Build.TestImportPos[path], ResolveImport)
		if p1.Error == nil {
			walk(p1)
		}
	}
	for _, root := range roots {
		walk(root)
		for _, path := range root.TestImports {
			walkTest(root, path)
		}
		for _, path := range root.XTestImports {
			walkTest(root, path)
		}
	}
	return all
}

var cmdCache = map[string]*Package{}

func ClearCmdCache() {
	for name := range cmdCache {
		delete(cmdCache, name)
	}
}

// LoadPackage loads the package named by arg.
func LoadPackage(arg string, stk *ImportStack) *Package {
	p := loadPackage(arg, stk)
	setToolFlags(p)
	return p
}

// LoadPackageNoFlags is like LoadPackage
// but does not guarantee that the build tool flags are set in the result.
// It is only for use by GOPATH-based "go get"
// and is only appropriate for preliminary loading of packages.
// A real load using LoadPackage or (more likely)
// Packages, PackageAndErrors, or PackagesForBuild
// must be done before passing the package to any build
// steps, so that the tool flags can be set properly.
// TODO(rsc): When GOPATH-based "go get" is removed, delete this function.
func LoadPackageNoFlags(arg string, stk *ImportStack) *Package {
	return loadPackage(arg, stk)
}

// loadPackage is like loadImport but is used for command-line arguments,
// not for paths found in import statements. In addition to ordinary import paths,
// loadPackage accepts pseudo-paths beginning with cmd/ to denote commands
// in the Go command directory, as well as paths to those directories.
func loadPackage(arg string, stk *ImportStack) *Package {
	if build.IsLocalImport(arg) {
		dir := arg
		if !filepath.IsAbs(dir) {
			if abs, err := filepath.Abs(dir); err == nil {
				// interpret relative to current directory
				dir = abs
			}
		}
		if sub, ok := hasSubdir(cfg.GOROOTsrc, dir); ok && strings.HasPrefix(sub, "cmd/") && !strings.Contains(sub[4:], "/") {
			arg = sub
		}
	}
	if strings.HasPrefix(arg, "cmd/") && !strings.Contains(arg[4:], "/") {
		if p := cmdCache[arg]; p != nil {
			return p
		}
		stk.Push(arg)
		defer stk.Pop()

		bp, err := cfg.BuildContext.ImportDir(filepath.Join(cfg.GOROOTsrc, arg), 0)
		bp.ImportPath = arg
		bp.Goroot = true
		bp.BinDir = cfg.GOROOTbin
		if cfg.GOROOTbin != "" {
			bp.BinDir = cfg.GOROOTbin
		}
		bp.Root = cfg.GOROOT
		bp.SrcRoot = cfg.GOROOTsrc
		p := new(Package)
		cmdCache[arg] = p
		p.load(stk, bp, err)
		if p.Error == nil && p.Name != "main" {
			p.Error = &PackageError{
				ImportStack: stk.Copy(),
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
	if build.IsLocalImport(arg) || filepath.IsAbs(arg) {
		dir := arg
		if !filepath.IsAbs(arg) {
			dir = filepath.Join(base.Cwd, arg)
		}
		bp, _ := cfg.BuildContext.ImportDir(dir, build.FindOnly)
		if bp.ImportPath != "" && bp.ImportPath != "." {
			arg = bp.ImportPath
		}
	}

	return LoadImport(arg, base.Cwd, nil, stk, nil, 0)
}

// Packages returns the packages named by the
// command line arguments 'args'. If a named package
// cannot be loaded at all (for example, if the directory does not exist),
// then packages prints an error and does not include that
// package in the results. However, if errors occur trying
// to load dependencies of a named package, the named
// package is still returned, with p.Incomplete = true
// and details in p.DepsErrors.
func Packages(args []string) []*Package {
	var pkgs []*Package
	for _, pkg := range PackagesAndErrors(args) {
		if pkg.Error != nil {
			base.Errorf("can't load package: %s", pkg.Error)
			continue
		}
		pkgs = append(pkgs, pkg)
	}
	return pkgs
}

// PackagesAndErrors is like 'packages' but returns a
// *Package for every argument, even the ones that
// cannot be loaded at all.
// The packages that fail to load will have p.Error != nil.
func PackagesAndErrors(patterns []string) []*Package {
	if len(patterns) > 0 && strings.HasSuffix(patterns[0], ".go") {
		return []*Package{GoFilesPackage(patterns)}
	}

	matches := ImportPaths(patterns)
	var (
		pkgs    []*Package
		stk     ImportStack
		seenPkg = make(map[*Package]bool)
	)

	for _, m := range matches {
		for _, pkg := range m.Pkgs {
			p := loadPackage(pkg, &stk)
			p.Match = append(p.Match, m.Pattern)
			p.Internal.CmdlinePkg = true
			if m.Literal {
				// Note: do not set = m.Literal unconditionally
				// because maybe we'll see p matching both
				// a literal and also a non-literal pattern.
				p.Internal.CmdlinePkgLiteral = true
			}
			if seenPkg[p] {
				continue
			}
			seenPkg[p] = true
			pkgs = append(pkgs, p)
		}
	}

	// Now that CmdlinePkg is set correctly,
	// compute the effective flags for all loaded packages
	// (not just the ones matching the patterns but also
	// their dependencies).
	setToolFlags(pkgs...)

	return pkgs
}

func setToolFlags(pkgs ...*Package) {
	for _, p := range PackageList(pkgs) {
		p.Internal.Asmflags = BuildAsmflags.For(p)
		p.Internal.Gcflags = BuildGcflags.For(p)
		p.Internal.Ldflags = BuildLdflags.For(p)
		p.Internal.Gccgoflags = BuildGccgoflags.For(p)
	}
}

func ImportPaths(args []string) []*search.Match {
	if ModInit(); cfg.ModulesEnabled {
		return ModImportPaths(args)
	}
	return search.ImportPaths(args)
}

// PackagesForBuild is like Packages but exits
// if any of the packages or their dependencies have errors
// (cannot be built).
func PackagesForBuild(args []string) []*Package {
	pkgs := PackagesAndErrors(args)
	printed := map[*PackageError]bool{}
	for _, pkg := range pkgs {
		if pkg.Error != nil {
			base.Errorf("can't load package: %s", pkg.Error)
			printed[pkg.Error] = true
		}
		for _, err := range pkg.DepsErrors {
			// Since these are errors in dependencies,
			// the same error might show up multiple times,
			// once in each package that depends on it.
			// Only print each once.
			if !printed[err] {
				printed[err] = true
				base.Errorf("%s", err)
			}
		}
	}
	base.ExitIfErrors()

	// Check for duplicate loads of the same package.
	// That should be impossible, but if it does happen then
	// we end up trying to build the same package twice,
	// usually in parallel overwriting the same files,
	// which doesn't work very well.
	seen := map[string]bool{}
	reported := map[string]bool{}
	for _, pkg := range PackageList(pkgs) {
		if seen[pkg.ImportPath] && !reported[pkg.ImportPath] {
			reported[pkg.ImportPath] = true
			base.Errorf("internal error: duplicate loads of %s", pkg.ImportPath)
		}
		seen[pkg.ImportPath] = true
	}
	base.ExitIfErrors()

	return pkgs
}

// GoFilesPackage creates a package for building a collection of Go files
// (typically named on the command line). The target is named p.a for
// package p or named after the first Go file for package main.
func GoFilesPackage(gofiles []string) *Package {
	ModInit()

	for _, f := range gofiles {
		if !strings.HasSuffix(f, ".go") {
			base.Fatalf("named files must be .go files")
		}
	}

	var stk ImportStack
	ctxt := cfg.BuildContext
	ctxt.UseAllFiles = true

	// Synthesize fake "directory" that only shows the named files,
	// to make it look like this is a standard package or
	// command directory. So that local imports resolve
	// consistently, the files must all be in the same directory.
	var dirent []os.FileInfo
	var dir string
	for _, file := range gofiles {
		fi, err := os.Stat(file)
		if err != nil {
			base.Fatalf("%s", err)
		}
		if fi.IsDir() {
			base.Fatalf("%s is a directory, should be a Go file", file)
		}
		dir1, _ := filepath.Split(file)
		if dir1 == "" {
			dir1 = "./"
		}
		if dir == "" {
			dir = dir1
		} else if dir != dir1 {
			base.Fatalf("named files must all be in one directory; have %s and %s", dir, dir1)
		}
		dirent = append(dirent, fi)
	}
	ctxt.ReadDir = func(string) ([]os.FileInfo, error) { return dirent, nil }

	if cfg.ModulesEnabled {
		ModImportFromFiles(gofiles)
	}

	var err error
	if dir == "" {
		dir = base.Cwd
	}
	dir, err = filepath.Abs(dir)
	if err != nil {
		base.Fatalf("%s", err)
	}

	bp, err := ctxt.ImportDir(dir, 0)
	if ModDirImportPath != nil {
		// Use the effective import path of the directory
		// for deciding visibility during pkg.load.
		bp.ImportPath = ModDirImportPath(dir)
	}
	pkg := new(Package)
	pkg.Internal.Local = true
	pkg.Internal.CmdlineFiles = true
	stk.Push("main")
	pkg.load(&stk, bp, err)
	stk.Pop()
	pkg.Internal.LocalPrefix = dirToImportPath(dir)
	pkg.ImportPath = "command-line-arguments"
	pkg.Target = ""
	pkg.Match = gofiles

	if pkg.Name == "main" {
		_, elem := filepath.Split(gofiles[0])
		exe := elem[:len(elem)-len(".go")] + cfg.ExeSuffix
		if cfg.BuildO == "" {
			cfg.BuildO = exe
		}
		if cfg.GOBIN != "" {
			pkg.Target = filepath.Join(cfg.GOBIN, exe)
		} else if cfg.ModulesEnabled {
			pkg.Target = filepath.Join(ModBinDir(), exe)
		}
	}

	setToolFlags(pkg)

	return pkg
}
