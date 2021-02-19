// Copyright 2011 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

// Package load loads packages.
package load

import (
	"bytes"
	"context"
	"encoding/json"
	"errors"
	"fmt"
	"go/build"
	"go/scanner"
	"go/token"
	"io/fs"
	"os"
	"path"
	pathpkg "path"
	"path/filepath"
	"runtime"
	"sort"
	"strconv"
	"strings"
	"unicode"
	"unicode/utf8"

	"cmd/go/internal/base"
	"cmd/go/internal/cfg"
	"cmd/go/internal/fsys"
	"cmd/go/internal/modinfo"
	"cmd/go/internal/modload"
	"cmd/go/internal/par"
	"cmd/go/internal/search"
	"cmd/go/internal/str"
	"cmd/go/internal/trace"
	"cmd/internal/sys"

	"golang.org/x/mod/module"
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
	Root          string                `json:",omitempty"` // Go root, Go path dir, or module root dir containing this package
	ConflictDir   string                `json:",omitempty"` // Dir is hidden by this other directory
	ForTest       string                `json:",omitempty"` // package is only for use in named test
	Export        string                `json:",omitempty"` // file containing export data (set by go list -export)
	BuildID       string                `json:",omitempty"` // build ID of the compiled package (set by go list -export)
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
	GoFiles           []string `json:",omitempty"` // .go source files (excluding CgoFiles, TestGoFiles, XTestGoFiles)
	CgoFiles          []string `json:",omitempty"` // .go source files that import "C"
	CompiledGoFiles   []string `json:",omitempty"` // .go output from running cgo on CgoFiles
	IgnoredGoFiles    []string `json:",omitempty"` // .go source files ignored due to build constraints
	IgnoredOtherFiles []string `json:",omitempty"` // non-.go source files ignored due to build constraints
	CFiles            []string `json:",omitempty"` // .c source files
	CXXFiles          []string `json:",omitempty"` // .cc, .cpp and .cxx source files
	MFiles            []string `json:",omitempty"` // .m source files
	HFiles            []string `json:",omitempty"` // .h, .hh, .hpp and .hxx source files
	FFiles            []string `json:",omitempty"` // .f, .F, .for and .f90 Fortran source files
	SFiles            []string `json:",omitempty"` // .s source files
	SwigFiles         []string `json:",omitempty"` // .swig files
	SwigCXXFiles      []string `json:",omitempty"` // .swigcxx files
	SysoFiles         []string `json:",omitempty"` // .syso system object files added to package

	// Embedded files
	EmbedPatterns []string `json:",omitempty"` // //go:embed patterns
	EmbedFiles    []string `json:",omitempty"` // files matched by EmbedPatterns

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
	TestGoFiles        []string `json:",omitempty"` // _test.go files in package
	TestImports        []string `json:",omitempty"` // imports from TestGoFiles
	TestEmbedPatterns  []string `json:",omitempty"` // //go:embed patterns
	TestEmbedFiles     []string `json:",omitempty"` // files matched by TestEmbedPatterns
	XTestGoFiles       []string `json:",omitempty"` // _test.go files outside package
	XTestImports       []string `json:",omitempty"` // imports from XTestGoFiles
	XTestEmbedPatterns []string `json:",omitempty"` // //go:embed patterns
	XTestEmbedFiles    []string `json:",omitempty"` // files matched by XTestEmbedPatterns
}

// AllFiles returns the names of all the files considered for the package.
// This is used for sanity and security checks, so we include all files,
// even IgnoredGoFiles, because some subcommands consider them.
// The go/build package filtered others out (like foo_wrongGOARCH.s)
// and that's OK.
func (p *Package) AllFiles() []string {
	files := str.StringList(
		p.GoFiles,
		p.CgoFiles,
		// no p.CompiledGoFiles, because they are from GoFiles or generated by us
		p.IgnoredGoFiles,
		p.IgnoredOtherFiles,
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

	// EmbedFiles may overlap with the other files.
	// Dedup, but delay building the map as long as possible.
	// Only files in the current directory (no slash in name)
	// need to be checked against the files variable above.
	var have map[string]bool
	for _, file := range p.EmbedFiles {
		if !strings.Contains(file, "/") {
			if have == nil {
				have = make(map[string]bool)
				for _, file := range files {
					have[file] = true
				}
			}
			if have[file] {
				continue
			}
		}
		files = append(files, file)
	}
	return files
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
	Embed             map[string][]string  // //go:embed comment mapping

	Asmflags   []string // -asmflags for this package
	Gcflags    []string // -gcflags for this package
	Ldflags    []string // -ldflags for this package
	Gccgoflags []string // -gccgoflags for this package
}

// A NoGoError indicates that no Go files for the package were applicable to the
// build for that package.
//
// That may be because there were no files whatsoever, or because all files were
// excluded, or because all non-excluded files were test sources.
type NoGoError struct {
	Package *Package
}

func (e *NoGoError) Error() string {
	if len(e.Package.IgnoredGoFiles) > 0 {
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

// setLoadPackageDataError presents an error found when loading package data
// as a *PackageError. It has special cases for some common errors to improve
// messages shown to users and reduce redundancy.
//
// setLoadPackageDataError returns true if it's safe to load information about
// imported packages, for example, if there was a parse error loading imports
// in one file, but other files are okay.
func (p *Package) setLoadPackageDataError(err error, path string, stk *ImportStack, importPos []token.Position) {
	matchErr, isMatchErr := err.(*search.MatchError)
	if isMatchErr && matchErr.Match.Pattern() == path {
		if matchErr.Match.IsLiteral() {
			// The error has a pattern has a pattern similar to the import path.
			// It may be slightly different (./foo matching example.com/foo),
			// but close enough to seem redundant.
			// Unwrap the error so we don't show the pattern.
			err = matchErr.Err
		}
	}

	// Replace (possibly wrapped) *build.NoGoError with *load.NoGoError.
	// The latter is more specific about the cause.
	var nogoErr *build.NoGoError
	if errors.As(err, &nogoErr) {
		if p.Dir == "" && nogoErr.Dir != "" {
			p.Dir = nogoErr.Dir
		}
		err = &NoGoError{Package: p}
	}

	// Take only the first error from a scanner.ErrorList. PackageError only
	// has room for one position, so we report the first error with a position
	// instead of all of the errors without a position.
	var pos string
	var isScanErr bool
	if scanErr, ok := err.(scanner.ErrorList); ok && len(scanErr) > 0 {
		isScanErr = true // For stack push/pop below.

		scanPos := scanErr[0].Pos
		scanPos.Filename = base.ShortPath(scanPos.Filename)
		pos = scanPos.String()
		err = errors.New(scanErr[0].Msg)
	}

	// Report the error on the importing package if the problem is with the import declaration
	// for example, if the package doesn't exist or if the import path is malformed.
	// On the other hand, don't include a position if the problem is with the imported package,
	// for example there are no Go files (NoGoError), or there's a problem in the imported
	// package's source files themselves (scanner errors).
	//
	// TODO(matloob): Perhaps make each of those the errors in the first group
	// (including modload.ImportMissingError, ImportMissingSumError, and the
	// corresponding "cannot find package %q in any of" GOPATH-mode error
	// produced in build.(*Context).Import; modload.AmbiguousImportError,
	// and modload.PackageNotInModuleError; and the malformed module path errors
	// produced in golang.org/x/mod/module.CheckMod) implement an interface
	// to make it easier to check for them? That would save us from having to
	// move the modload errors into this package to avoid a package import cycle,
	// and from having to export an error type for the errors produced in build.
	if !isMatchErr && (nogoErr != nil || isScanErr) {
		stk.Push(path)
		defer stk.Pop()
	}

	p.Error = &PackageError{
		ImportStack: stk.Copy(),
		Pos:         pos,
		Err:         err,
	}

	if path != stk.Top() {
		p.Error.setPos(importPos)
	}
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
	p.IgnoredOtherFiles = pp.IgnoredOtherFiles
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
	p.EmbedPatterns = pp.EmbedPatterns
	p.TestEmbedPatterns = pp.TestEmbedPatterns
	p.XTestEmbedPatterns = pp.XTestEmbedPatterns
}

// A PackageError describes an error loading information about a package.
type PackageError struct {
	ImportStack      []string // shortest path from package named on command line to this one
	Pos              string   // position of error
	Err              error    // the error itself
	IsImportCycle    bool     // the error is an import cycle
	Hard             bool     // whether the error is soft or hard; soft errors are ignored in some places
	alwaysPrintStack bool     // whether to always print the ImportStack
}

func (p *PackageError) Error() string {
	// TODO(#43696): decide when to print the stack or the position based on
	// the error type and whether the package is in the main module.
	// Document the rationale.
	if p.Pos != "" && (len(p.ImportStack) == 0 || !p.alwaysPrintStack) {
		// Omit import stack. The full path to the file where the error
		// is the most important thing.
		return p.Pos + ": " + p.Err.Error()
	}

	// If the error is an ImportPathError, and the last path on the stack appears
	// in the error message, omit that path from the stack to avoid repetition.
	// If an ImportPathError wraps another ImportPathError that matches the
	// last path on the stack, we don't omit the path. An error like
	// "package A imports B: error loading C caused by B" would not be clearer
	// if "imports B" were omitted.
	if len(p.ImportStack) == 0 {
		return p.Err.Error()
	}
	var optpos string
	if p.Pos != "" {
		optpos = "\n\t" + p.Pos
	}
	return "package " + strings.Join(p.ImportStack, "\n\timports ") + optpos + ": " + p.Err.Error()
}

func (p *PackageError) Unwrap() error { return p.Err }

// PackageError implements MarshalJSON so that Err is marshaled as a string
// and non-essential fields are omitted.
func (p *PackageError) MarshalJSON() ([]byte, error) {
	perr := struct {
		ImportStack []string
		Pos         string
		Err         string
	}{p.ImportStack, p.Pos, p.Err.Error()}
	return json.Marshal(perr)
}

func (p *PackageError) setPos(posList []token.Position) {
	if len(posList) == 0 {
		return
	}
	pos := posList[0]
	pos.Filename = base.ShortPath(pos.Filename)
	p.Pos = pos.String()
}

// ImportPathError is a type of error that prevents a package from being loaded
// for a given import path. When such a package is loaded, a *Package is
// returned with Err wrapping an ImportPathError: the error is attached to
// the imported package, not the importing package.
//
// The string returned by ImportPath must appear in the string returned by
// Error. Errors that wrap ImportPathError (such as PackageError) may omit
// the import path.
type ImportPathError interface {
	error
	ImportPath() string
}

var (
	_ ImportPathError = (*importError)(nil)
	_ ImportPathError = (*modload.ImportMissingError)(nil)
	_ ImportPathError = (*modload.ImportMissingSumError)(nil)
)

type importError struct {
	importPath string
	err        error // created with fmt.Errorf
}

func ImportErrorf(path, format string, args ...interface{}) ImportPathError {
	err := &importError{importPath: path, err: fmt.Errorf(format, args...)}
	if errStr := err.Error(); !strings.Contains(errStr, path) {
		panic(fmt.Sprintf("path %q not in error %q", path, errStr))
	}
	return err
}

func (e *importError) Error() string {
	return e.err.Error()
}

func (e *importError) Unwrap() error {
	// Don't return e.err directly, since we're only wrapping an error if %w
	// was passed to ImportErrorf.
	return errors.Unwrap(e.err)
}

func (e *importError) ImportPath() string {
	return e.importPath
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

func (s *ImportStack) Top() string {
	if len(*s) == 0 {
		return ""
	}
	return (*s)[len(*s)-1]
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

// packageCache is a lookup cache for LoadImport,
// so that if we look up a package multiple times
// we return the same pointer each time.
var packageCache = map[string]*Package{}

// ClearPackageCache clears the in-memory package cache and the preload caches.
// It is only for use by GOPATH-based "go get".
// TODO(jayconrod): When GOPATH-based "go get" is removed, delete this function.
func ClearPackageCache() {
	for name := range packageCache {
		delete(packageCache, name)
	}
	resolvedImportCache.Clear()
	packageDataCache.Clear()
}

// ClearPackageCachePartial clears packages with the given import paths from the
// in-memory package cache and the preload caches. It is only for use by
// GOPATH-based "go get".
// TODO(jayconrod): When GOPATH-based "go get" is removed, delete this function.
func ClearPackageCachePartial(args []string) {
	shouldDelete := make(map[string]bool)
	for _, arg := range args {
		shouldDelete[arg] = true
		if p := packageCache[arg]; p != nil {
			delete(packageCache, arg)
		}
	}
	resolvedImportCache.DeleteIf(func(key interface{}) bool {
		return shouldDelete[key.(importSpec).path]
	})
	packageDataCache.DeleteIf(func(key interface{}) bool {
		return shouldDelete[key.(string)]
	})
}

// ReloadPackageNoFlags is like LoadImport but makes sure
// not to use the package cache.
// It is only for use by GOPATH-based "go get".
// TODO(rsc): When GOPATH-based "go get" is removed, delete this function.
func ReloadPackageNoFlags(arg string, stk *ImportStack) *Package {
	p := packageCache[arg]
	if p != nil {
		delete(packageCache, arg)
		resolvedImportCache.DeleteIf(func(key interface{}) bool {
			return key.(importSpec).path == p.ImportPath
		})
		packageDataCache.Delete(p.ImportPath)
	}
	return LoadImport(context.TODO(), arg, base.Cwd, nil, stk, nil, 0)
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
func LoadImport(ctx context.Context, path, srcDir string, parent *Package, stk *ImportStack, importPos []token.Position, mode int) *Package {
	return loadImport(ctx, nil, path, srcDir, parent, stk, importPos, mode)
}

func loadImport(ctx context.Context, pre *preload, path, srcDir string, parent *Package, stk *ImportStack, importPos []token.Position, mode int) *Package {
	if path == "" {
		panic("LoadImport called with empty package path")
	}

	var parentPath, parentRoot string
	parentIsStd := false
	if parent != nil {
		parentPath = parent.ImportPath
		parentRoot = parent.Root
		parentIsStd = parent.Standard
	}
	bp, loaded, err := loadPackageData(path, parentPath, srcDir, parentRoot, parentIsStd, mode)
	if loaded && pre != nil && !IgnoreImports {
		pre.preloadImports(bp.Imports, bp)
	}
	if bp == nil {
		if importErr, ok := err.(ImportPathError); !ok || importErr.ImportPath() != path {
			// Only add path to the error's import stack if it's not already present on the error.
			stk.Push(path)
			defer stk.Pop()
		}
		return &Package{
			PackagePublic: PackagePublic{
				ImportPath: path,
				Error: &PackageError{
					ImportStack: stk.Copy(),
					Err:         err,
				},
			},
		}
	}

	importPath := bp.ImportPath
	p := packageCache[importPath]
	if p != nil {
		stk.Push(path)
		p = reusePackage(p, stk)
		stk.Pop()
	} else {
		p = new(Package)
		p.Internal.Local = build.IsLocalImport(path)
		p.ImportPath = importPath
		packageCache[importPath] = p

		// Load package.
		// loadPackageData may return bp != nil even if an error occurs,
		// in order to return partial information.
		p.load(ctx, path, stk, importPos, bp, err)

		if !cfg.ModulesEnabled && path != cleanImport(path) {
			p.Error = &PackageError{
				ImportStack: stk.Copy(),
				Err:         ImportErrorf(path, "non-canonical import path %q: should be %q", path, pathpkg.Clean(path)),
			}
			p.Incomplete = true
			p.Error.setPos(importPos)
		}
	}

	// Checked on every import because the rules depend on the code doing the importing.
	if perr := disallowInternal(srcDir, parent, parentPath, p, stk); perr != p {
		perr.Error.setPos(importPos)
		return perr
	}
	if mode&ResolveImport != 0 {
		if perr := disallowVendor(srcDir, path, parentPath, p, stk); perr != p {
			perr.Error.setPos(importPos)
			return perr
		}
	}

	if p.Name == "main" && parent != nil && parent.Dir != p.Dir {
		perr := *p
		perr.Error = &PackageError{
			ImportStack: stk.Copy(),
			Err:         ImportErrorf(path, "import %q is a program, not an importable package", path),
		}
		perr.Error.setPos(importPos)
		return &perr
	}

	if p.Internal.Local && parent != nil && !parent.Internal.Local {
		perr := *p
		var err error
		if path == "." {
			err = ImportErrorf(path, "%s: cannot import current directory", path)
		} else {
			err = ImportErrorf(path, "local import %q in non-local package", path)
		}
		perr.Error = &PackageError{
			ImportStack: stk.Copy(),
			Err:         err,
		}
		perr.Error.setPos(importPos)
		return &perr
	}

	return p
}

// loadPackageData loads information needed to construct a *Package. The result
// is cached, and later calls to loadPackageData for the same package will return
// the same data.
//
// loadPackageData returns a non-nil package even if err is non-nil unless
// the package path is malformed (for example, the path contains "mod/" or "@").
//
// loadPackageData returns a boolean, loaded, which is true if this is the
// first time the package was loaded. Callers may preload imports in this case.
func loadPackageData(path, parentPath, parentDir, parentRoot string, parentIsStd bool, mode int) (bp *build.Package, loaded bool, err error) {
	if path == "" {
		panic("loadPackageData called with empty package path")
	}

	if strings.HasPrefix(path, "mod/") {
		// Paths beginning with "mod/" might accidentally
		// look in the module cache directory tree in $GOPATH/pkg/mod/.
		// This prefix is owned by the Go core for possible use in the
		// standard library (since it does not begin with a domain name),
		// so it's OK to disallow entirely.
		return nil, false, fmt.Errorf("disallowed import path %q", path)
	}

	if strings.Contains(path, "@") {
		return nil, false, errors.New("can only use path@version syntax with 'go get' and 'go install' in module-aware mode")
	}

	// Determine canonical package path and directory.
	// For a local import the identifier is the pseudo-import path
	// we create from the full directory to the package.
	// Otherwise it is the usual import path.
	// For vendored imports, it is the expanded form.
	//
	// Note that when modules are enabled, local import paths are normally
	// canonicalized by modload.LoadPackages before now. However, if there's an
	// error resolving a local path, it will be returned untransformed
	// so that 'go list -e' reports something useful.
	importKey := importSpec{
		path:        path,
		parentPath:  parentPath,
		parentDir:   parentDir,
		parentRoot:  parentRoot,
		parentIsStd: parentIsStd,
		mode:        mode,
	}
	r := resolvedImportCache.Do(importKey, func() interface{} {
		var r resolvedImport
		if build.IsLocalImport(path) {
			r.dir = filepath.Join(parentDir, path)
			r.path = dirToImportPath(r.dir)
		} else if cfg.ModulesEnabled {
			r.dir, r.path, r.err = modload.Lookup(parentPath, parentIsStd, path)
		} else if mode&ResolveImport != 0 {
			// We do our own path resolution, because we want to
			// find out the key to use in packageCache without the
			// overhead of repeated calls to buildContext.Import.
			// The code is also needed in a few other places anyway.
			r.path = resolveImportPath(path, parentPath, parentDir, parentRoot, parentIsStd)
		} else if mode&ResolveModule != 0 {
			r.path = moduleImportPath(path, parentPath, parentDir, parentRoot)
		}
		if r.path == "" {
			r.path = path
		}
		return r
	}).(resolvedImport)
	// Invariant: r.path is set to the resolved import path. If the path cannot
	// be resolved, r.path is set to path, the source import path.
	// r.path is never empty.

	// Load the package from its directory. If we already found the package's
	// directory when resolving its import path, use that.
	data := packageDataCache.Do(r.path, func() interface{} {
		loaded = true
		var data packageData
		if r.dir != "" {
			var buildMode build.ImportMode
			if !cfg.ModulesEnabled {
				buildMode = build.ImportComment
			}
			data.p, data.err = cfg.BuildContext.ImportDir(r.dir, buildMode)
			if data.p.Root == "" && cfg.ModulesEnabled {
				if info := modload.PackageModuleInfo(path); info != nil {
					data.p.Root = info.Dir
				}
			}
		} else if r.err != nil {
			data.p = new(build.Package)
			data.err = r.err
		} else if cfg.ModulesEnabled && path != "unsafe" {
			data.p = new(build.Package)
			data.err = fmt.Errorf("unknown import path %q: internal error: module loader did not resolve import", r.path)
		} else {
			buildMode := build.ImportComment
			if mode&ResolveImport == 0 || r.path != path {
				// Not vendoring, or we already found the vendored path.
				buildMode |= build.IgnoreVendor
			}
			data.p, data.err = cfg.BuildContext.Import(r.path, parentDir, buildMode)
		}
		data.p.ImportPath = r.path

		// Set data.p.BinDir in cases where go/build.Context.Import
		// may give us a path we don't want.
		if !data.p.Goroot {
			if cfg.GOBIN != "" {
				data.p.BinDir = cfg.GOBIN
			} else if cfg.ModulesEnabled {
				data.p.BinDir = modload.BinDir()
			}
		}

		if !cfg.ModulesEnabled && data.err == nil &&
			data.p.ImportComment != "" && data.p.ImportComment != path &&
			!strings.Contains(path, "/vendor/") && !strings.HasPrefix(path, "vendor/") {
			data.err = fmt.Errorf("code in directory %s expects import %q", data.p.Dir, data.p.ImportComment)
		}
		return data
	}).(packageData)

	return data.p, loaded, data.err
}

// importSpec describes an import declaration in source code. It is used as a
// cache key for resolvedImportCache.
type importSpec struct {
	path                              string
	parentPath, parentDir, parentRoot string
	parentIsStd                       bool
	mode                              int
}

// resolvedImport holds a canonical identifier for a package. It may also contain
// a path to the package's directory and an error if one occurred. resolvedImport
// is the value type in resolvedImportCache.
type resolvedImport struct {
	path, dir string
	err       error
}

// packageData holds information loaded from a package. It is the value type
// in packageDataCache.
type packageData struct {
	p   *build.Package
	err error
}

// resolvedImportCache maps import strings (importSpec) to canonical package names
// (resolvedImport).
var resolvedImportCache par.Cache

// packageDataCache maps canonical package names (string) to package metadata
// (packageData).
var packageDataCache par.Cache

// preloadWorkerCount is the number of concurrent goroutines that can load
// packages. Experimentally, there are diminishing returns with more than
// 4 workers. This was measured on the following machines.
//
// * MacBookPro with a 4-core Intel Core i7 CPU
// * Linux workstation with 6-core Intel Xeon CPU
// * Linux workstation with 24-core Intel Xeon CPU
//
// It is very likely (though not confirmed) that this workload is limited
// by memory bandwidth. We don't have a good way to determine the number of
// workers that would saturate the bus though, so runtime.GOMAXPROCS
// seems like a reasonable default.
var preloadWorkerCount = runtime.GOMAXPROCS(0)

// preload holds state for managing concurrent preloading of package data.
//
// A preload should be created with newPreload before loading a large
// package graph. flush must be called when package loading is complete
// to ensure preload goroutines are no longer active. This is necessary
// because of global mutable state that cannot safely be read and written
// concurrently. In particular, packageDataCache may be cleared by "go get"
// in GOPATH mode, and modload.loaded (accessed via modload.Lookup) may be
// modified by modload.LoadPackages.
type preload struct {
	cancel chan struct{}
	sema   chan struct{}
}

// newPreload creates a new preloader. flush must be called later to avoid
// accessing global state while it is being modified.
func newPreload() *preload {
	pre := &preload{
		cancel: make(chan struct{}),
		sema:   make(chan struct{}, preloadWorkerCount),
	}
	return pre
}

// preloadMatches loads data for package paths matched by patterns.
// When preloadMatches returns, some packages may not be loaded yet, but
// loadPackageData and loadImport are always safe to call.
func (pre *preload) preloadMatches(matches []*search.Match) {
	for _, m := range matches {
		for _, pkg := range m.Pkgs {
			select {
			case <-pre.cancel:
				return
			case pre.sema <- struct{}{}:
				go func(pkg string) {
					mode := 0 // don't use vendoring or module import resolution
					bp, loaded, err := loadPackageData(pkg, "", base.Cwd, "", false, mode)
					<-pre.sema
					if bp != nil && loaded && err == nil && !IgnoreImports {
						pre.preloadImports(bp.Imports, bp)
					}
				}(pkg)
			}
		}
	}
}

// preloadImports queues a list of imports for preloading.
// When preloadImports returns, some packages may not be loaded yet,
// but loadPackageData and loadImport are always safe to call.
func (pre *preload) preloadImports(imports []string, parent *build.Package) {
	parentIsStd := parent.Goroot && parent.ImportPath != "" && search.IsStandardImportPath(parent.ImportPath)
	for _, path := range imports {
		if path == "C" || path == "unsafe" {
			continue
		}
		select {
		case <-pre.cancel:
			return
		case pre.sema <- struct{}{}:
			go func(path string) {
				bp, loaded, err := loadPackageData(path, parent.ImportPath, parent.Dir, parent.Root, parentIsStd, ResolveImport)
				<-pre.sema
				if bp != nil && loaded && err == nil && !IgnoreImports {
					pre.preloadImports(bp.Imports, bp)
				}
			}(path)
		}
	}
}

// flush stops pending preload operations. flush blocks until preload calls to
// loadPackageData have completed. The preloader will not make any new calls
// to loadPackageData.
func (pre *preload) flush() {
	// flush is usually deferred.
	// Don't hang program waiting for workers on panic.
	if v := recover(); v != nil {
		panic(v)
	}

	close(pre.cancel)
	for i := 0; i < preloadWorkerCount; i++ {
		pre.sema <- struct{}{}
	}
}

func cleanImport(path string) string {
	orig := path
	path = pathpkg.Clean(path)
	if strings.HasPrefix(orig, "./") && path != ".." && !strings.HasPrefix(path, "../") {
		path = "./" + path
	}
	return path
}

var isDirCache par.Cache

func isDir(path string) bool {
	return isDirCache.Do(path, func() interface{} {
		fi, err := fsys.Stat(path)
		return err == nil && fi.IsDir()
	}).(bool)
}

// ResolveImportPath returns the true meaning of path when it appears in parent.
// There are two different resolutions applied.
// First, there is Go 1.5 vendoring (golang.org/s/go15vendor).
// If vendor expansion doesn't trigger, then the path is also subject to
// Go 1.11 module legacy conversion (golang.org/issue/25069).
func ResolveImportPath(parent *Package, path string) (found string) {
	var parentPath, parentDir, parentRoot string
	parentIsStd := false
	if parent != nil {
		parentPath = parent.ImportPath
		parentDir = parent.Dir
		parentRoot = parent.Root
		parentIsStd = parent.Standard
	}
	return resolveImportPath(path, parentPath, parentDir, parentRoot, parentIsStd)
}

func resolveImportPath(path, parentPath, parentDir, parentRoot string, parentIsStd bool) (found string) {
	if cfg.ModulesEnabled {
		if _, p, e := modload.Lookup(parentPath, parentIsStd, path); e == nil {
			return p
		}
		return path
	}
	found = vendoredImportPath(path, parentPath, parentDir, parentRoot)
	if found != path {
		return found
	}
	return moduleImportPath(path, parentPath, parentDir, parentRoot)
}

// dirAndRoot returns the source directory and workspace root
// for the package p, guaranteeing that root is a path prefix of dir.
func dirAndRoot(path string, dir, root string) (string, string) {
	origDir, origRoot := dir, root
	dir = filepath.Clean(dir)
	root = filepath.Join(root, "src")
	if !str.HasFilePathPrefix(dir, root) || path != "command-line-arguments" && filepath.Join(root, path) != dir {
		// Look for symlinks before reporting error.
		dir = expandPath(dir)
		root = expandPath(root)
	}

	if !str.HasFilePathPrefix(dir, root) || len(dir) <= len(root) || dir[len(root)] != filepath.Separator || path != "command-line-arguments" && !build.IsLocalImport(path) && filepath.Join(root, path) != dir {
		base.Fatalf("unexpected directory layout:\n"+
			"	import path: %s\n"+
			"	root: %s\n"+
			"	dir: %s\n"+
			"	expand root: %s\n"+
			"	expand dir: %s\n"+
			"	separator: %s",
			path,
			filepath.Join(origRoot, "src"),
			filepath.Clean(origDir),
			origRoot,
			origDir,
			string(filepath.Separator))
	}

	return dir, root
}

// vendoredImportPath returns the vendor-expansion of path when it appears in parent.
// If parent is x/y/z, then path might expand to x/y/z/vendor/path, x/y/vendor/path,
// x/vendor/path, vendor/path, or else stay path if none of those exist.
// vendoredImportPath returns the expanded path or, if no expansion is found, the original.
func vendoredImportPath(path, parentPath, parentDir, parentRoot string) (found string) {
	if parentRoot == "" {
		return path
	}

	dir, root := dirAndRoot(parentPath, parentDir, parentRoot)

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
			importPath := parentPath
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
	goModPathCache par.Cache
)

// goModPath returns the module path in the go.mod in dir, if any.
func goModPath(dir string) (path string) {
	return goModPathCache.Do(dir, func() interface{} {
		data, err := os.ReadFile(filepath.Join(dir, "go.mod"))
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
	}).(string)
}

// findVersionElement returns the slice indices of the final version element /vN in path.
// If there is no such element, it returns -1, -1.
func findVersionElement(path string) (i, j int) {
	j = len(path)
	for i = len(path) - 1; i >= 0; i-- {
		if path[i] == '/' {
			if isVersionElement(path[i+1 : j]) {
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
	if len(s) < 2 || s[0] != 'v' || s[1] == '0' || s[1] == '1' && len(s) == 2 {
		return false
	}
	for i := 1; i < len(s); i++ {
		if s[i] < '0' || '9' < s[i] {
			return false
		}
	}
	return true
}

// moduleImportPath translates import paths found in go modules
// back down to paths that can be resolved in ordinary builds.
//
// Define “new” code as code with a go.mod file in the same directory
// or a parent directory. If an import in new code says x/y/v2/z but
// x/y/v2/z does not exist and x/y/go.mod says “module x/y/v2”,
// then go build will read the import as x/y/z instead.
// See golang.org/issue/25069.
func moduleImportPath(path, parentPath, parentDir, parentRoot string) (found string) {
	if parentRoot == "" {
		return path
	}

	// If there are no vN elements in path, leave it alone.
	// (The code below would do the same, but only after
	// some other file system accesses that we can avoid
	// here by returning early.)
	if i, _ := findVersionElement(path); i < 0 {
		return path
	}

	dir, root := dirAndRoot(parentPath, parentDir, parentRoot)

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
	files, _ := os.ReadDir(dir)
	for _, f := range files {
		if !f.IsDir() && strings.HasSuffix(f.Name(), ".go") {
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
				Err:           errors.New("import cycle not allowed"),
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
	if str.HasPathPrefix(p.ImportPath, "testing/internal") && importerPath == "testmain" {
		return p
	}

	// We can't check standard packages with gccgo.
	if cfg.BuildContext.Compiler == "gccgo" && p.Standard {
		return p
	}

	// The sort package depends on internal/reflectlite, but during bootstrap
	// the path rewriting causes the normal internal checks to fail.
	// Instead, just ignore the internal rules during bootstrap.
	if p.Standard && strings.HasPrefix(importerPath, "bootstrap/") {
		return p
	}

	// importerPath is empty: we started
	// with a name given on the command line, not an
	// import. Anything listed on the command line is fine.
	if importerPath == "" {
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
		if importer.Internal.CmdlineFiles {
			// The importer is a list of command-line files.
			// Pretend that the import path is the import path of the
			// directory containing them.
			// If the directory is outside the main module, this will resolve to ".",
			// which is not a prefix of any valid module.
			importerPath = modload.DirImportPath(importer.Dir)
		}
		parentOfInternal := p.ImportPath[:i]
		if str.HasPathPrefix(importerPath, parentOfInternal) {
			return p
		}
	}

	// Internal is present, and srcDir is outside parent's tree. Not allowed.
	perr := *p
	perr.Error = &PackageError{
		alwaysPrintStack: true,
		ImportStack:      stk.Copy(),
		Err:              ImportErrorf(p.ImportPath, "use of internal package "+p.ImportPath+" not allowed"),
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
func disallowVendor(srcDir string, path string, importerPath string, p *Package, stk *ImportStack) *Package {
	// If the importerPath is empty, we started
	// with a name given on the command line, not an
	// import. Anything listed on the command line is fine.
	if importerPath == "" {
		return p
	}

	if perr := disallowVendorVisibility(srcDir, p, importerPath, stk); perr != p {
		return perr
	}

	// Paths like x/vendor/y must be imported as y, never as x/vendor/y.
	if i, ok := FindVendor(path); ok {
		perr := *p
		perr.Error = &PackageError{
			ImportStack: stk.Copy(),
			Err:         ImportErrorf(path, "%s must be imported as %s", path, path[i+len("vendor/"):]),
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
func disallowVendorVisibility(srcDir string, p *Package, importerPath string, stk *ImportStack) *Package {
	// The stack does not include p.ImportPath.
	// If there's nothing on the stack, we started
	// with a name given on the command line, not an
	// import. Anything listed on the command line is fine.
	if importerPath == "" {
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
		Err:         errors.New("use of vendored package not allowed"),
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

// exeFromImportPath returns an executable name
// for a package using the import path.
//
// The executable name is the last element of the import path.
// In module-aware mode, an additional rule is used on import paths
// consisting of two or more path elements. If the last element is
// a vN path element specifying the major version, then the
// second last element of the import path is used instead.
func (p *Package) exeFromImportPath() string {
	_, elem := pathpkg.Split(p.ImportPath)
	if cfg.ModulesEnabled {
		// If this is example.com/mycmd/v2, it's more useful to
		// install it as mycmd than as v2. See golang.org/issue/24667.
		if elem != p.ImportPath && isVersionElement(elem) {
			_, elem = pathpkg.Split(pathpkg.Dir(p.ImportPath))
		}
	}
	return elem
}

// exeFromFiles returns an executable name for a package
// using the first element in GoFiles or CgoFiles collections without the prefix.
//
// Returns empty string in case of empty collection.
func (p *Package) exeFromFiles() string {
	var src string
	if len(p.GoFiles) > 0 {
		src = p.GoFiles[0]
	} else if len(p.CgoFiles) > 0 {
		src = p.CgoFiles[0]
	} else {
		return ""
	}
	_, elem := filepath.Split(src)
	return elem[:len(elem)-len(".go")]
}

// DefaultExecName returns the default executable name for a package
func (p *Package) DefaultExecName() string {
	if p.Internal.CmdlineFiles {
		return p.exeFromFiles()
	}
	return p.exeFromImportPath()
}

// load populates p using information from bp, err, which should
// be the result of calling build.Context.Import.
// stk contains the import stack, not including path itself.
func (p *Package) load(ctx context.Context, path string, stk *ImportStack, importPos []token.Position, bp *build.Package, err error) {
	p.copyBuild(bp)

	// The localPrefix is the path we interpret ./ imports relative to.
	// Synthesized main packages sometimes override this.
	if p.Internal.Local {
		p.Internal.LocalPrefix = dirToImportPath(p.Dir)
	}

	// setError sets p.Error if it hasn't already been set. We may proceed
	// after encountering some errors so that 'go list -e' has more complete
	// output. If there's more than one error, we should report the first.
	setError := func(err error) {
		if p.Error == nil {
			p.Error = &PackageError{
				ImportStack: stk.Copy(),
				Err:         err,
			}

			// Add the importer's position information if the import position exists, and
			// the current package being examined is the importer.
			// If we have not yet accepted package p onto the import stack,
			// then the cause of the error is not within p itself: the error
			// must be either in an explicit command-line argument,
			// or on the importer side (indicated by a non-empty importPos).
			if path != stk.Top() && len(importPos) > 0 {
				p.Error.setPos(importPos)
			}
		}
	}

	if err != nil {
		p.Incomplete = true
		p.setLoadPackageDataError(err, path, stk, importPos)
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
			// TODO(matloob): remove this branch, and StalePath itself. code.google.com/p/go is so
			// old, even this code checking for it is stale now!
			newPath := strings.Replace(p.ImportPath, "code.google.com/p/go.", "golang.org/x/", 1)
			e := ImportErrorf(p.ImportPath, "the %v command has moved; use %v instead.", p.ImportPath, newPath)
			setError(e)
			return
		}
		elem := p.DefaultExecName()
		full := cfg.BuildContext.GOOS + "_" + cfg.BuildContext.GOARCH + "/" + elem
		if cfg.BuildContext.GOOS != base.ToolGOOS || cfg.BuildContext.GOARCH != base.ToolGOARCH {
			// Install cross-compiled binaries to subdirectories of bin.
			elem = full
		}
		if p.Internal.Build.BinDir == "" && cfg.ModulesEnabled {
			p.Internal.Build.BinDir = modload.BinDir()
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
		if cfg.BuildLinkshared && p.Target != "" {
			// TODO(bcmills): The reliance on p.Target implies that -linkshared does
			// not work for any package that lacks a Target — such as a non-main
			// package in module mode. We should probably fix that.
			shlibnamefile := p.Target[:len(p.Target)-2] + ".shlibname"
			shlib, err := os.ReadFile(shlibnamefile)
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
		addImport("unsafe", true)
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

	// Check for case-insensitive collisions of import paths.
	fold := str.ToFold(p.ImportPath)
	if other := foldPath[fold]; other == "" {
		foldPath[fold] = p.ImportPath
	} else if other != p.ImportPath {
		setError(ImportErrorf(p.ImportPath, "case-insensitive import collision: %q and %q", p.ImportPath, other))
		return
	}

	if !SafeArg(p.ImportPath) {
		setError(ImportErrorf(p.ImportPath, "invalid import path %q", p.ImportPath))
		return
	}

	// Errors after this point are caused by this package, not the importing
	// package. Pushing the path here prevents us from reporting the error
	// with the position of the import declaration.
	stk.Push(path)
	defer stk.Pop()

	p.EmbedFiles, p.Internal.Embed, err = resolveEmbed(p.Dir, p.EmbedPatterns)
	if err != nil {
		p.Incomplete = true
		setError(err)
		embedErr := err.(*EmbedError)
		p.Error.setPos(p.Internal.Build.EmbedPatternPos[embedErr.Pattern])
	}

	// Check for case-insensitive collision of input files.
	// To avoid problems on case-insensitive files, we reject any package
	// where two different input files have equal names under a case-insensitive
	// comparison.
	inputs := p.AllFiles()
	f1, f2 := str.FoldDup(inputs)
	if f1 != "" {
		setError(fmt.Errorf("case-insensitive file name collision: %q and %q", f1, f2))
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
			setError(fmt.Errorf("invalid input file name %q", file))
			return
		}
	}
	if name := pathpkg.Base(p.ImportPath); !SafeArg(name) {
		setError(fmt.Errorf("invalid input directory name %q", name))
		return
	}

	// Build list of imported packages and full dependency list.
	imports := make([]*Package, 0, len(p.Imports))
	for i, path := range importPaths {
		if path == "C" {
			continue
		}
		p1 := LoadImport(ctx, path, p.Dir, p, stk, p.Internal.Build.ImportPos[path], ResolveImport)

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
	p.collectDeps()

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

	// The gc toolchain only permits C source files with cgo or SWIG.
	if len(p.CFiles) > 0 && !p.UsesCgo() && !p.UsesSwig() && cfg.BuildContext.Compiler == "gc" {
		setError(fmt.Errorf("C source files not allowed when not using cgo or SWIG: %s", strings.Join(p.CFiles, " ")))
		return
	}

	// C++, Objective-C, and Fortran source files are permitted only with cgo or SWIG,
	// regardless of toolchain.
	if len(p.CXXFiles) > 0 && !p.UsesCgo() && !p.UsesSwig() {
		setError(fmt.Errorf("C++ source files not allowed when not using cgo or SWIG: %s", strings.Join(p.CXXFiles, " ")))
		return
	}
	if len(p.MFiles) > 0 && !p.UsesCgo() && !p.UsesSwig() {
		setError(fmt.Errorf("Objective-C source files not allowed when not using cgo or SWIG: %s", strings.Join(p.MFiles, " ")))
		return
	}
	if len(p.FFiles) > 0 && !p.UsesCgo() && !p.UsesSwig() {
		setError(fmt.Errorf("Fortran source files not allowed when not using cgo or SWIG: %s", strings.Join(p.FFiles, " ")))
		return
	}

	if cfg.ModulesEnabled && p.Error == nil {
		mainPath := p.ImportPath
		if p.Internal.CmdlineFiles {
			mainPath = "command-line-arguments"
		}
		p.Module = modload.PackageModuleInfo(mainPath)
		if p.Name == "main" && len(p.DepsErrors) == 0 {
			p.Internal.BuildInfo = modload.PackageBuildInfo(mainPath, p.Deps)
		}
	}
}

// An EmbedError indicates a problem with a go:embed directive.
type EmbedError struct {
	Pattern string
	Err     error
}

func (e *EmbedError) Error() string {
	return fmt.Sprintf("pattern %s: %v", e.Pattern, e.Err)
}

func (e *EmbedError) Unwrap() error {
	return e.Err
}

// ResolveEmbed resolves //go:embed patterns and returns only the file list.
// For use by go mod vendor to find embedded files it should copy into the
// vendor directory.
// TODO(#42504): Once go mod vendor uses load.PackagesAndErrors, just
// call (*Package).ResolveEmbed
func ResolveEmbed(dir string, patterns []string) ([]string, error) {
	files, _, err := resolveEmbed(dir, patterns)
	return files, err
}

// resolveEmbed resolves //go:embed patterns to precise file lists.
// It sets files to the list of unique files matched (for go list),
// and it sets pmap to the more precise mapping from
// patterns to files.
func resolveEmbed(pkgdir string, patterns []string) (files []string, pmap map[string][]string, err error) {
	var pattern string
	defer func() {
		if err != nil {
			err = &EmbedError{
				Pattern: pattern,
				Err:     err,
			}
		}
	}()

	// TODO(rsc): All these messages need position information for better error reports.
	pmap = make(map[string][]string)
	have := make(map[string]int)
	dirOK := make(map[string]bool)
	pid := 0 // pattern ID, to allow reuse of have map
	for _, pattern = range patterns {
		pid++

		// Check pattern is valid for //go:embed.
		if _, err := path.Match(pattern, ""); err != nil || !validEmbedPattern(pattern) {
			return nil, nil, fmt.Errorf("invalid pattern syntax")
		}

		// Glob to find matches.
		match, err := fsys.Glob(pkgdir + string(filepath.Separator) + filepath.FromSlash(pattern))
		if err != nil {
			return nil, nil, err
		}

		// Filter list of matches down to the ones that will still exist when
		// the directory is packaged up as a module. (If p.Dir is in the module cache,
		// only those files exist already, but if p.Dir is in the current module,
		// then there may be other things lying around, like symbolic links or .git directories.)
		var list []string
		for _, file := range match {
			rel := filepath.ToSlash(file[len(pkgdir)+1:]) // file, relative to p.Dir

			what := "file"
			info, err := fsys.Lstat(file)
			if err != nil {
				return nil, nil, err
			}
			if info.IsDir() {
				what = "directory"
			}

			// Check that directories along path do not begin a new module
			// (do not contain a go.mod).
			for dir := file; len(dir) > len(pkgdir)+1 && !dirOK[dir]; dir = filepath.Dir(dir) {
				if _, err := fsys.Stat(filepath.Join(dir, "go.mod")); err == nil {
					return nil, nil, fmt.Errorf("cannot embed %s %s: in different module", what, rel)
				}
				if dir != file {
					if info, err := fsys.Lstat(dir); err == nil && !info.IsDir() {
						return nil, nil, fmt.Errorf("cannot embed %s %s: in non-directory %s", what, rel, dir[len(pkgdir)+1:])
					}
				}
				dirOK[dir] = true
				if elem := filepath.Base(dir); isBadEmbedName(elem) {
					if dir == file {
						return nil, nil, fmt.Errorf("cannot embed %s %s: invalid name %s", what, rel, elem)
					} else {
						return nil, nil, fmt.Errorf("cannot embed %s %s: in invalid directory %s", what, rel, elem)
					}
				}
			}

			switch {
			default:
				return nil, nil, fmt.Errorf("cannot embed irregular file %s", rel)

			case info.Mode().IsRegular():
				if have[rel] != pid {
					have[rel] = pid
					list = append(list, rel)
				}

			case info.IsDir():
				// Gather all files in the named directory, stopping at module boundaries
				// and ignoring files that wouldn't be packaged into a module.
				count := 0
				err := fsys.Walk(file, func(path string, info os.FileInfo, err error) error {
					if err != nil {
						return err
					}
					rel := filepath.ToSlash(path[len(pkgdir)+1:])
					name := info.Name()
					if path != file && (isBadEmbedName(name) || name[0] == '.' || name[0] == '_') {
						// Ignore bad names, assuming they won't go into modules.
						// Also avoid hidden files that user may not know about.
						// See golang.org/issue/42328.
						if info.IsDir() {
							return fs.SkipDir
						}
						return nil
					}
					if info.IsDir() {
						if _, err := fsys.Stat(filepath.Join(path, "go.mod")); err == nil {
							return filepath.SkipDir
						}
						return nil
					}
					if !info.Mode().IsRegular() {
						return nil
					}
					count++
					if have[rel] != pid {
						have[rel] = pid
						list = append(list, rel)
					}
					return nil
				})
				if err != nil {
					return nil, nil, err
				}
				if count == 0 {
					return nil, nil, fmt.Errorf("cannot embed directory %s: contains no embeddable files", rel)
				}
			}
		}

		if len(list) == 0 {
			return nil, nil, fmt.Errorf("no matching files found")
		}
		sort.Strings(list)
		pmap[pattern] = list
	}

	for file := range have {
		files = append(files, file)
	}
	sort.Strings(files)
	return files, pmap, nil
}

func validEmbedPattern(pattern string) bool {
	return pattern != "." && fs.ValidPath(pattern)
}

// isBadEmbedName reports whether name is the base name of a file that
// can't or won't be included in modules and therefore shouldn't be treated
// as existing for embedding.
func isBadEmbedName(name string) bool {
	if err := module.CheckFilePath(name); err != nil {
		return true
	}
	switch name {
	// Empty string should be impossible but make it bad.
	case "":
		return true
	// Version control directories won't be present in module.
	case ".bzr", ".hg", ".git", ".svn":
		return true
	}
	return false
}

// collectDeps populates p.Deps and p.DepsErrors by iterating over
// p.Internal.Imports.
//
// TODO(jayconrod): collectDeps iterates over transitive imports for every
// package. We should only need to visit direct imports.
func (p *Package) collectDeps() {
	deps := make(map[string]*Package)
	var q []*Package
	q = append(q, p.Internal.Imports...)
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
	if !cfg.BuildContext.CgoEnabled {
		return false
	}

	// Some targets must use external linking even inside GOROOT.
	switch cfg.BuildContext.GOOS {
	case "android":
		if cfg.BuildContext.GOARCH != "arm64" {
			return true
		}
	case "ios":
		return true
	}

	// Currently build modes c-shared, pie (on systems that do not
	// support PIE with internal linking mode (currently all
	// systems: issue #18968)), plugin, and -linkshared force
	// external linking mode, as of course does
	// -ldflags=-linkmode=external. External linking mode forces
	// an import of runtime/cgo.
	// If there are multiple -linkmode options, the last one wins.
	pieCgo := cfg.BuildBuildmode == "pie" && !sys.InternalLinkPIESupported(cfg.BuildContext.GOOS, cfg.BuildContext.GOARCH)
	linkmodeExternal := false
	if p != nil {
		ldflags := BuildLdflags.For(p)
		for i := len(ldflags) - 1; i >= 0; i-- {
			a := ldflags[i]
			if a == "-linkmode=external" ||
				a == "-linkmode" && i+1 < len(ldflags) && ldflags[i+1] == "external" {
				linkmodeExternal = true
				break
			} else if a == "-linkmode=internal" ||
				a == "-linkmode" && i+1 < len(ldflags) && ldflags[i+1] == "internal" {
				break
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
	return p.mkAbs(str.StringList(p.IgnoredGoFiles, p.GoFiles, p.CgoFiles, p.TestGoFiles, p.XTestGoFiles))
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
func TestPackageList(ctx context.Context, roots []*Package) []*Package {
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
		p1 := LoadImport(ctx, path, root.Dir, root, &stk, root.Internal.Build.TestImportPos[path], ResolveImport)
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

// LoadImportWithFlags loads the package with the given import path and
// sets tool flags on that package. This function is useful loading implicit
// dependencies (like sync/atomic for coverage).
// TODO(jayconrod): delete this function and set flags automatically
// in LoadImport instead.
func LoadImportWithFlags(path, srcDir string, parent *Package, stk *ImportStack, importPos []token.Position, mode int) *Package {
	p := LoadImport(context.TODO(), path, srcDir, parent, stk, importPos, mode)
	setToolFlags(p)
	return p
}

// ModResolveTests indicates whether calls to the module loader should also
// resolve test dependencies of the requested packages.
//
// If ModResolveTests is true, then the module loader needs to resolve test
// dependencies at the same time as packages; otherwise, the test dependencies
// of those packages could be missing, and resolving those missing dependencies
// could change the selected versions of modules that provide other packages.
//
// TODO(#40775): Change this from a global variable to an explicit function
// argument where needed.
var ModResolveTests bool

// PackagesAndErrors returns the packages named by the command line arguments
// 'patterns'. If a named package cannot be loaded, PackagesAndErrors returns
// a *Package with the Error field describing the failure. If errors are found
// loading imported packages, the DepsErrors field is set. The Incomplete field
// may be set as well.
//
// To obtain a flat list of packages, use PackageList.
// To report errors loading packages, use ReportPackageErrors.
func PackagesAndErrors(ctx context.Context, patterns []string) []*Package {
	ctx, span := trace.StartSpan(ctx, "load.PackagesAndErrors")
	defer span.Done()

	for _, p := range patterns {
		// Listing is only supported with all patterns referring to either:
		// - Files that are part of the same directory.
		// - Explicit package paths or patterns.
		if strings.HasSuffix(p, ".go") {
			// We need to test whether the path is an actual Go file and not a
			// package path or pattern ending in '.go' (see golang.org/issue/34653).
			if fi, err := fsys.Stat(p); err == nil && !fi.IsDir() {
				return []*Package{GoFilesPackage(ctx, patterns)}
			}
		}
	}

	var matches []*search.Match
	if modload.Init(); cfg.ModulesEnabled {
		loadOpts := modload.PackageOpts{
			ResolveMissingImports: true,
			LoadTests:             ModResolveTests,
			SilenceErrors:         true,
		}
		matches, _ = modload.LoadPackages(ctx, loadOpts, patterns...)
	} else {
		matches = search.ImportPaths(patterns)
	}

	var (
		pkgs    []*Package
		stk     ImportStack
		seenPkg = make(map[*Package]bool)
	)

	pre := newPreload()
	defer pre.flush()
	pre.preloadMatches(matches)

	for _, m := range matches {
		for _, pkg := range m.Pkgs {
			if pkg == "" {
				panic(fmt.Sprintf("ImportPaths returned empty package for pattern %s", m.Pattern()))
			}
			p := loadImport(ctx, pre, pkg, base.Cwd, nil, &stk, nil, 0)
			p.Match = append(p.Match, m.Pattern())
			p.Internal.CmdlinePkg = true
			if m.IsLiteral() {
				// Note: do not set = m.IsLiteral unconditionally
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

		if len(m.Errs) > 0 {
			// In addition to any packages that were actually resolved from the
			// pattern, there was some error in resolving the pattern itself.
			// Report it as a synthetic package.
			p := new(Package)
			p.ImportPath = m.Pattern()
			// Pass an empty ImportStack and nil importPos: the error arose from a pattern, not an import.
			var stk ImportStack
			var importPos []token.Position
			p.setLoadPackageDataError(m.Errs[0], m.Pattern(), &stk, importPos)
			p.Incomplete = true
			p.Match = append(p.Match, m.Pattern())
			p.Internal.CmdlinePkg = true
			if m.IsLiteral() {
				p.Internal.CmdlinePkgLiteral = true
			}
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

// CheckPackageErrors prints errors encountered loading pkgs and their
// dependencies, then exits with a non-zero status if any errors were found.
func CheckPackageErrors(pkgs []*Package) {
	printed := map[*PackageError]bool{}
	for _, pkg := range pkgs {
		if pkg.Error != nil {
			base.Errorf("%v", pkg.Error)
			printed[pkg.Error] = true
		}
		for _, err := range pkg.DepsErrors {
			// Since these are errors in dependencies,
			// the same error might show up multiple times,
			// once in each package that depends on it.
			// Only print each once.
			if !printed[err] {
				printed[err] = true
				base.Errorf("%v", err)
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
}

func setToolFlags(pkgs ...*Package) {
	for _, p := range PackageList(pkgs) {
		p.Internal.Asmflags = BuildAsmflags.For(p)
		p.Internal.Gcflags = BuildGcflags.For(p)
		p.Internal.Ldflags = BuildLdflags.For(p)
		p.Internal.Gccgoflags = BuildGccgoflags.For(p)
	}
}

// GoFilesPackage creates a package for building a collection of Go files
// (typically named on the command line). The target is named p.a for
// package p or named after the first Go file for package main.
func GoFilesPackage(ctx context.Context, gofiles []string) *Package {
	modload.Init()

	for _, f := range gofiles {
		if !strings.HasSuffix(f, ".go") {
			pkg := new(Package)
			pkg.Internal.Local = true
			pkg.Internal.CmdlineFiles = true
			pkg.Name = f
			pkg.Error = &PackageError{
				Err: fmt.Errorf("named files must be .go files: %s", pkg.Name),
			}
			return pkg
		}
	}

	var stk ImportStack
	ctxt := cfg.BuildContext
	ctxt.UseAllFiles = true

	// Synthesize fake "directory" that only shows the named files,
	// to make it look like this is a standard package or
	// command directory. So that local imports resolve
	// consistently, the files must all be in the same directory.
	var dirent []fs.FileInfo
	var dir string
	for _, file := range gofiles {
		fi, err := fsys.Stat(file)
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
	ctxt.ReadDir = func(string) ([]fs.FileInfo, error) { return dirent, nil }

	if cfg.ModulesEnabled {
		modload.ImportFromFiles(ctx, gofiles)
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
	pkg := new(Package)
	pkg.Internal.Local = true
	pkg.Internal.CmdlineFiles = true
	pkg.load(ctx, "command-line-arguments", &stk, nil, bp, err)
	pkg.Internal.LocalPrefix = dirToImportPath(dir)
	pkg.ImportPath = "command-line-arguments"
	pkg.Target = ""
	pkg.Match = gofiles

	if pkg.Name == "main" {
		exe := pkg.DefaultExecName() + cfg.ExeSuffix

		if cfg.GOBIN != "" {
			pkg.Target = filepath.Join(cfg.GOBIN, exe)
		} else if cfg.ModulesEnabled {
			pkg.Target = filepath.Join(modload.BinDir(), exe)
		}
	}

	setToolFlags(pkg)

	return pkg
}
