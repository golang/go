// Copyright 2018 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

package modload

// This file contains the module-mode package loader, as well as some accessory
// functions pertaining to the package import graph.
//
// There are two exported entry points into package loading — LoadPackages and
// ImportFromFiles — both implemented in terms of loadFromRoots, which itself
// manipulates an instance of the loader struct.
//
// Although most of the loading state is maintained in the loader struct,
// one key piece - the build list - is a global, so that it can be modified
// separate from the loading operation, such as during "go get"
// upgrades/downgrades or in "go mod" operations.
// TODO(#40775): It might be nice to make the loader take and return
// a buildList rather than hard-coding use of the global.
//
// Loading is an iterative process. On each iteration, we try to load the
// requested packages and their transitive imports, then try to resolve modules
// for any imported packages that are still missing.
//
// The first step of each iteration identifies a set of “root” packages.
// Normally the root packages are exactly those matching the named pattern
// arguments. However, for the "all" meta-pattern, the final set of packages is
// computed from the package import graph, and therefore cannot be an initial
// input to loading that graph. Instead, the root packages for the "all" pattern
// are those contained in the main module, and allPatternIsRoot parameter to the
// loader instructs it to dynamically expand those roots to the full "all"
// pattern as loading progresses.
//
// The pkgInAll flag on each loadPkg instance tracks whether that
// package is known to match the "all" meta-pattern.
// A package matches the "all" pattern if:
// 	- it is in the main module, or
// 	- it is imported by any test in the main module, or
// 	- it is imported by another package in "all", or
// 	- the main module specifies a go version ≤ 1.15, and the package is imported
// 	  by a *test of* another package in "all".
//
// When we implement lazy loading, we will record the modules providing packages
// in "all" even when we are only loading individual packages, so we set the
// pkgInAll flag regardless of the whether the "all" pattern is a root.
// (This is necessary to maintain the “import invariant” described in
// https://golang.org/design/36460-lazy-module-loading.)
//
// Because "go mod vendor" prunes out the tests of vendored packages, the
// behavior of the "all" pattern with -mod=vendor in Go 1.11–1.15 is the same
// as the "all" pattern (regardless of the -mod flag) in 1.16+.
// The allClosesOverTests parameter to the loader indicates whether the "all"
// pattern should close over tests (as in Go 1.11–1.15) or stop at only those
// packages transitively imported by the packages and tests in the main module
// ("all" in Go 1.16+ and "go mod vendor" in Go 1.11+).
//
// Note that it is possible for a loaded package NOT to be in "all" even when we
// are loading the "all" pattern. For example, packages that are transitive
// dependencies of other roots named on the command line must be loaded, but are
// not in "all". (The mod_notall test illustrates this behavior.)
// Similarly, if the LoadTests flag is set but the "all" pattern does not close
// over test dependencies, then when we load the test of a package that is in
// "all" but outside the main module, the dependencies of that test will not
// necessarily themselves be in "all". (That configuration does not arise in Go
// 1.11–1.15, but it will be possible in Go 1.16+.)
//
// Loading proceeds from the roots, using a parallel work-queue with a limit on
// the amount of active work (to avoid saturating disks, CPU cores, and/or
// network connections). Each package is added to the queue the first time it is
// imported by another package. When we have finished identifying the imports of
// a package, we add the test for that package if it is needed. A test may be
// needed if:
// 	- the package matches a root pattern and tests of the roots were requested, or
// 	- the package is in the main module and the "all" pattern is requested
// 	  (because the "all" pattern includes the dependencies of tests in the main
// 	  module), or
// 	- the package is in "all" and the definition of "all" we are using includes
// 	  dependencies of tests (as is the case in Go ≤1.15).
//
// After all available packages have been loaded, we examine the results to
// identify any requested or imported packages that are still missing, and if
// so, which modules we could add to the module graph in order to make the
// missing packages available. We add those to the module graph and iterate,
// until either all packages resolve successfully or we cannot identify any
// module that would resolve any remaining missing package.
//
// If the main module is “tidy” (that is, if "go mod tidy" is a no-op for it)
// and all requested packages are in "all", then loading completes in a single
// iteration.
// TODO(bcmills): We should also be able to load in a single iteration if the
// requested packages all come from modules that are themselves tidy, regardless
// of whether those packages are in "all". Today, that requires two iterations
// if those packages are not found in existing dependencies of the main module.

import (
	"bytes"
	"context"
	"errors"
	"fmt"
	"go/build"
	"io/fs"
	"os"
	"path"
	pathpkg "path"
	"path/filepath"
	"reflect"
	"runtime"
	"sort"
	"strings"
	"sync"
	"sync/atomic"

	"cmd/go/internal/base"
	"cmd/go/internal/cfg"
	"cmd/go/internal/fsys"
	"cmd/go/internal/imports"
	"cmd/go/internal/modfetch"
	"cmd/go/internal/mvs"
	"cmd/go/internal/par"
	"cmd/go/internal/search"
	"cmd/go/internal/str"

	"golang.org/x/mod/module"
)

// loaded is the most recently-used package loader.
// It holds details about individual packages.
var loaded *loader

// PackageOpts control the behavior of the LoadPackages function.
type PackageOpts struct {
	// Tags are the build tags in effect (as interpreted by the
	// cmd/go/internal/imports package).
	// If nil, treated as equivalent to imports.Tags().
	Tags map[string]bool

	// ResolveMissingImports indicates that we should attempt to add module
	// dependencies as needed to resolve imports of packages that are not found.
	//
	// For commands that support the -mod flag, resolving imports may still fail
	// if the flag is set to "readonly" (the default) or "vendor".
	ResolveMissingImports bool

	// AllowPackage, if non-nil, is called after identifying the module providing
	// each package. If AllowPackage returns a non-nil error, that error is set
	// for the package, and the imports and test of that package will not be
	// loaded.
	//
	// AllowPackage may be invoked concurrently by multiple goroutines,
	// and may be invoked multiple times for a given package path.
	AllowPackage func(ctx context.Context, path string, mod module.Version) error

	// LoadTests loads the test dependencies of each package matching a requested
	// pattern. If ResolveMissingImports is also true, test dependencies will be
	// resolved if missing.
	LoadTests bool

	// UseVendorAll causes the "all" package pattern to be interpreted as if
	// running "go mod vendor" (or building with "-mod=vendor").
	//
	// This is a no-op for modules that declare 'go 1.16' or higher, for which this
	// is the default (and only) interpretation of the "all" pattern in module mode.
	UseVendorAll bool

	// AllowErrors indicates that LoadPackages should not terminate the process if
	// an error occurs.
	AllowErrors bool

	// SilenceErrors indicates that LoadPackages should not print errors
	// that occur while loading packages. SilenceErrors implies AllowErrors.
	SilenceErrors bool

	// SilenceMissingStdImports indicates that LoadPackages should not print
	// errors or terminate the process if an imported package is missing, and the
	// import path looks like it might be in the standard library (perhaps in a
	// future version).
	SilenceMissingStdImports bool

	// SilenceUnmatchedWarnings suppresses the warnings normally emitted for
	// patterns that did not match any packages.
	SilenceUnmatchedWarnings bool
}

// LoadPackages identifies the set of packages matching the given patterns and
// loads the packages in the import graph rooted at that set.
func LoadPackages(ctx context.Context, opts PackageOpts, patterns ...string) (matches []*search.Match, loadedPackages []string) {
	LoadModFile(ctx)
	if opts.Tags == nil {
		opts.Tags = imports.Tags()
	}

	patterns = search.CleanPatterns(patterns)
	matches = make([]*search.Match, 0, len(patterns))
	allPatternIsRoot := false
	for _, pattern := range patterns {
		matches = append(matches, search.NewMatch(pattern))
		if pattern == "all" {
			allPatternIsRoot = true
		}
	}

	updateMatches := func(ld *loader) {
		for _, m := range matches {
			switch {
			case m.IsLocal():
				// Evaluate list of file system directories on first iteration.
				if m.Dirs == nil {
					matchLocalDirs(m)
				}

				// Make a copy of the directory list and translate to import paths.
				// Note that whether a directory corresponds to an import path
				// changes as the build list is updated, and a directory can change
				// from not being in the build list to being in it and back as
				// the exact version of a particular module increases during
				// the loader iterations.
				m.Pkgs = m.Pkgs[:0]
				for _, dir := range m.Dirs {
					pkg, err := resolveLocalPackage(dir)
					if err != nil {
						if !m.IsLiteral() && (err == errPkgIsBuiltin || err == errPkgIsGorootSrc) {
							continue // Don't include "builtin" or GOROOT/src in wildcard patterns.
						}

						// If we're outside of a module, ensure that the failure mode
						// indicates that.
						ModRoot()

						if ld != nil {
							m.AddError(err)
						}
						continue
					}
					m.Pkgs = append(m.Pkgs, pkg)
				}

			case m.IsLiteral():
				m.Pkgs = []string{m.Pattern()}

			case strings.Contains(m.Pattern(), "..."):
				m.Errs = m.Errs[:0]
				matchPackages(ctx, m, opts.Tags, includeStd, buildList)

			case m.Pattern() == "all":
				if ld == nil {
					// The initial roots are the packages in the main module.
					// loadFromRoots will expand that to "all".
					m.Errs = m.Errs[:0]
					matchPackages(ctx, m, opts.Tags, omitStd, []module.Version{Target})
				} else {
					// Starting with the packages in the main module,
					// enumerate the full list of "all".
					m.Pkgs = ld.computePatternAll()
				}

			case m.Pattern() == "std" || m.Pattern() == "cmd":
				if m.Pkgs == nil {
					m.MatchPackages() // Locate the packages within GOROOT/src.
				}

			default:
				panic(fmt.Sprintf("internal error: modload missing case for pattern %s", m.Pattern()))
			}
		}
	}

	loaded = loadFromRoots(loaderParams{
		PackageOpts: opts,

		allClosesOverTests: index.allPatternClosesOverTests() && !opts.UseVendorAll,
		allPatternIsRoot:   allPatternIsRoot,

		listRoots: func() (roots []string) {
			updateMatches(nil)
			for _, m := range matches {
				roots = append(roots, m.Pkgs...)
			}
			return roots
		},
	})

	// One last pass to finalize wildcards.
	updateMatches(loaded)

	// Report errors, if any.
	checkMultiplePaths()
	for _, pkg := range loaded.pkgs {
		if pkg.err != nil {
			if sumErr := (*ImportMissingSumError)(nil); errors.As(pkg.err, &sumErr) {
				if importer := pkg.stack; importer != nil {
					sumErr.importer = importer.path
					sumErr.importerVersion = importer.mod.Version
					sumErr.importerIsTest = importer.testOf != nil
				}
			}
			silence := opts.SilenceErrors
			if stdErr := (*ImportMissingError)(nil); errors.As(pkg.err, &stdErr) &&
				stdErr.isStd && opts.SilenceMissingStdImports {
				silence = true
			}

			if !silence {
				if opts.AllowErrors {
					fmt.Fprintf(os.Stderr, "%s: %v\n", pkg.stackText(), pkg.err)
				} else {
					base.Errorf("%s: %v", pkg.stackText(), pkg.err)
				}
			}
		}
		if !pkg.isTest() {
			loadedPackages = append(loadedPackages, pkg.path)
		}
	}
	if !opts.SilenceErrors {
		// Also list errors in matching patterns (such as directory permission
		// errors for wildcard patterns).
		for _, match := range matches {
			for _, err := range match.Errs {
				if opts.AllowErrors {
					fmt.Fprintf(os.Stderr, "%v\n", err)
				} else {
					base.Errorf("%v", err)
				}
			}
		}
	}
	base.ExitIfErrors()

	if !opts.SilenceUnmatchedWarnings {
		search.WarnUnmatched(matches)
	}

	// Success! Update go.mod (if needed) and return the results.
	WriteGoMod()
	sort.Strings(loadedPackages)
	return matches, loadedPackages
}

// matchLocalDirs is like m.MatchDirs, but tries to avoid scanning directories
// outside of the standard library and active modules.
func matchLocalDirs(m *search.Match) {
	if !m.IsLocal() {
		panic(fmt.Sprintf("internal error: resolveLocalDirs on non-local pattern %s", m.Pattern()))
	}

	if i := strings.Index(m.Pattern(), "..."); i >= 0 {
		// The pattern is local, but it is a wildcard. Its packages will
		// only resolve to paths if they are inside of the standard
		// library, the main module, or some dependency of the main
		// module. Verify that before we walk the filesystem: a filesystem
		// walk in a directory like /var or /etc can be very expensive!
		dir := filepath.Dir(filepath.Clean(m.Pattern()[:i+3]))
		absDir := dir
		if !filepath.IsAbs(dir) {
			absDir = filepath.Join(base.Cwd, dir)
		}
		if search.InDir(absDir, cfg.GOROOTsrc) == "" && search.InDir(absDir, ModRoot()) == "" && pathInModuleCache(absDir) == "" {
			m.Dirs = []string{}
			m.AddError(fmt.Errorf("directory prefix %s outside available modules", base.ShortPath(absDir)))
			return
		}
	}

	m.MatchDirs()
}

// resolveLocalPackage resolves a filesystem path to a package path.
func resolveLocalPackage(dir string) (string, error) {
	var absDir string
	if filepath.IsAbs(dir) {
		absDir = filepath.Clean(dir)
	} else {
		absDir = filepath.Join(base.Cwd, dir)
	}

	bp, err := cfg.BuildContext.ImportDir(absDir, 0)
	if err != nil && (bp == nil || len(bp.IgnoredGoFiles) == 0) {
		// golang.org/issue/32917: We should resolve a relative path to a
		// package path only if the relative path actually contains the code
		// for that package.
		//
		// If the named directory does not exist or contains no Go files,
		// the package does not exist.
		// Other errors may affect package loading, but not resolution.
		if _, err := fsys.Stat(absDir); err != nil {
			if os.IsNotExist(err) {
				// Canonicalize OS-specific errors to errDirectoryNotFound so that error
				// messages will be easier for users to search for.
				return "", &fs.PathError{Op: "stat", Path: absDir, Err: errDirectoryNotFound}
			}
			return "", err
		}
		if _, noGo := err.(*build.NoGoError); noGo {
			// A directory that does not contain any Go source files — even ignored
			// ones! — is not a Go package, and we can't resolve it to a package
			// path because that path could plausibly be provided by some other
			// module.
			//
			// Any other error indicates that the package “exists” (at least in the
			// sense that it cannot exist in any other module), but has some other
			// problem (such as a syntax error).
			return "", err
		}
	}

	if modRoot != "" && absDir == modRoot {
		if absDir == cfg.GOROOTsrc {
			return "", errPkgIsGorootSrc
		}
		return targetPrefix, nil
	}

	// Note: The checks for @ here are just to avoid misinterpreting
	// the module cache directories (formerly GOPATH/src/mod/foo@v1.5.2/bar).
	// It's not strictly necessary but helpful to keep the checks.
	if modRoot != "" && strings.HasPrefix(absDir, modRoot+string(filepath.Separator)) && !strings.Contains(absDir[len(modRoot):], "@") {
		suffix := filepath.ToSlash(absDir[len(modRoot):])
		if strings.HasPrefix(suffix, "/vendor/") {
			if cfg.BuildMod != "vendor" {
				return "", fmt.Errorf("without -mod=vendor, directory %s has no package path", absDir)
			}

			readVendorList()
			pkg := strings.TrimPrefix(suffix, "/vendor/")
			if _, ok := vendorPkgModule[pkg]; !ok {
				return "", fmt.Errorf("directory %s is not a package listed in vendor/modules.txt", absDir)
			}
			return pkg, nil
		}

		if targetPrefix == "" {
			pkg := strings.TrimPrefix(suffix, "/")
			if pkg == "builtin" {
				// "builtin" is a pseudo-package with a real source file.
				// It's not included in "std", so it shouldn't resolve from "."
				// within module "std" either.
				return "", errPkgIsBuiltin
			}
			return pkg, nil
		}

		pkg := targetPrefix + suffix
		if _, ok, err := dirInModule(pkg, targetPrefix, modRoot, true); err != nil {
			return "", err
		} else if !ok {
			return "", &PackageNotInModuleError{Mod: Target, Pattern: pkg}
		}
		return pkg, nil
	}

	if sub := search.InDir(absDir, cfg.GOROOTsrc); sub != "" && sub != "." && !strings.Contains(sub, "@") {
		pkg := filepath.ToSlash(sub)
		if pkg == "builtin" {
			return "", errPkgIsBuiltin
		}
		return pkg, nil
	}

	pkg := pathInModuleCache(absDir)
	if pkg == "" {
		return "", fmt.Errorf("directory %s outside available modules", base.ShortPath(absDir))
	}
	return pkg, nil
}

var (
	errDirectoryNotFound = errors.New("directory not found")
	errPkgIsGorootSrc    = errors.New("GOROOT/src is not an importable package")
	errPkgIsBuiltin      = errors.New(`"builtin" is a pseudo-package, not an importable package`)
)

// pathInModuleCache returns the import path of the directory dir,
// if dir is in the module cache copy of a module in our build list.
func pathInModuleCache(dir string) string {
	tryMod := func(m module.Version) (string, bool) {
		var root string
		var err error
		if repl := Replacement(m); repl.Path != "" && repl.Version == "" {
			root = repl.Path
			if !filepath.IsAbs(root) {
				root = filepath.Join(ModRoot(), root)
			}
		} else if repl.Path != "" {
			root, err = modfetch.DownloadDir(repl)
		} else {
			root, err = modfetch.DownloadDir(m)
		}
		if err != nil {
			return "", false
		}

		sub := search.InDir(dir, root)
		if sub == "" {
			return "", false
		}
		sub = filepath.ToSlash(sub)
		if strings.Contains(sub, "/vendor/") || strings.HasPrefix(sub, "vendor/") || strings.Contains(sub, "@") {
			return "", false
		}

		return path.Join(m.Path, filepath.ToSlash(sub)), true
	}

	for _, m := range buildList[1:] {
		if importPath, ok := tryMod(m); ok {
			// checkMultiplePaths ensures that a module can be used for at most one
			// requirement, so this must be it.
			return importPath
		}
	}
	return ""
}

// ImportFromFiles adds modules to the build list as needed
// to satisfy the imports in the named Go source files.
func ImportFromFiles(ctx context.Context, gofiles []string) {
	LoadModFile(ctx)

	tags := imports.Tags()
	imports, testImports, err := imports.ScanFiles(gofiles, tags)
	if err != nil {
		base.Fatalf("go: %v", err)
	}

	loaded = loadFromRoots(loaderParams{
		PackageOpts: PackageOpts{
			Tags:                  tags,
			ResolveMissingImports: true,
		},
		allClosesOverTests: index.allPatternClosesOverTests(),
		listRoots: func() (roots []string) {
			roots = append(roots, imports...)
			roots = append(roots, testImports...)
			return roots
		},
	})
	WriteGoMod()
}

// DirImportPath returns the effective import path for dir,
// provided it is within the main module, or else returns ".".
func DirImportPath(dir string) string {
	if !HasModRoot() {
		return "."
	}
	LoadModFile(context.TODO())

	if !filepath.IsAbs(dir) {
		dir = filepath.Join(base.Cwd, dir)
	} else {
		dir = filepath.Clean(dir)
	}

	if dir == modRoot {
		return targetPrefix
	}
	if strings.HasPrefix(dir, modRoot+string(filepath.Separator)) {
		suffix := filepath.ToSlash(dir[len(modRoot):])
		if strings.HasPrefix(suffix, "/vendor/") {
			return strings.TrimPrefix(suffix, "/vendor/")
		}
		return targetPrefix + suffix
	}
	return "."
}

// TargetPackages returns the list of packages in the target (top-level) module
// matching pattern, which may be relative to the working directory, under all
// build tag settings.
func TargetPackages(ctx context.Context, pattern string) *search.Match {
	// TargetPackages is relative to the main module, so ensure that the main
	// module is a thing that can contain packages.
	LoadModFile(ctx)
	ModRoot()

	m := search.NewMatch(pattern)
	matchPackages(ctx, m, imports.AnyTags(), omitStd, []module.Version{Target})
	return m
}

// ImportMap returns the actual package import path
// for an import path found in source code.
// If the given import path does not appear in the source code
// for the packages that have been loaded, ImportMap returns the empty string.
func ImportMap(path string) string {
	pkg, ok := loaded.pkgCache.Get(path).(*loadPkg)
	if !ok {
		return ""
	}
	return pkg.path
}

// PackageDir returns the directory containing the source code
// for the package named by the import path.
func PackageDir(path string) string {
	pkg, ok := loaded.pkgCache.Get(path).(*loadPkg)
	if !ok {
		return ""
	}
	return pkg.dir
}

// PackageModule returns the module providing the package named by the import path.
func PackageModule(path string) module.Version {
	pkg, ok := loaded.pkgCache.Get(path).(*loadPkg)
	if !ok {
		return module.Version{}
	}
	return pkg.mod
}

// PackageImports returns the imports for the package named by the import path.
// Test imports will be returned as well if tests were loaded for the package
// (i.e., if "all" was loaded or if LoadTests was set and the path was matched
// by a command line argument). PackageImports will return nil for
// unknown package paths.
func PackageImports(path string) (imports, testImports []string) {
	pkg, ok := loaded.pkgCache.Get(path).(*loadPkg)
	if !ok {
		return nil, nil
	}
	imports = make([]string, len(pkg.imports))
	for i, p := range pkg.imports {
		imports[i] = p.path
	}
	if pkg.test != nil {
		testImports = make([]string, len(pkg.test.imports))
		for i, p := range pkg.test.imports {
			testImports[i] = p.path
		}
	}
	return imports, testImports
}

// Lookup returns the source directory, import path, and any loading error for
// the package at path as imported from the package in parentDir.
// Lookup requires that one of the Load functions in this package has already
// been called.
func Lookup(parentPath string, parentIsStd bool, path string) (dir, realPath string, err error) {
	if path == "" {
		panic("Lookup called with empty package path")
	}

	if parentIsStd {
		path = loaded.stdVendor(parentPath, path)
	}
	pkg, ok := loaded.pkgCache.Get(path).(*loadPkg)
	if !ok {
		// The loader should have found all the relevant paths.
		// There are a few exceptions, though:
		//	- during go list without -test, the p.Resolve calls to process p.TestImports and p.XTestImports
		//	  end up here to canonicalize the import paths.
		//	- during any load, non-loaded packages like "unsafe" end up here.
		//	- during any load, build-injected dependencies like "runtime/cgo" end up here.
		//	- because we ignore appengine/* in the module loader,
		//	  the dependencies of any actual appengine/* library end up here.
		dir := findStandardImportPath(path)
		if dir != "" {
			return dir, path, nil
		}
		return "", "", errMissing
	}
	return pkg.dir, pkg.path, pkg.err
}

// A loader manages the process of loading information about
// the required packages for a particular build,
// checking that the packages are available in the module set,
// and updating the module set if needed.
type loader struct {
	loaderParams

	work *par.Queue

	// reset on each iteration
	roots    []*loadPkg
	pkgCache *par.Cache // package path (string) → *loadPkg
	pkgs     []*loadPkg // transitive closure of loaded packages and tests; populated in buildStacks

	// computed at end of iterations
	direct map[string]bool // imported directly by main module
}

// loaderParams configure the packages loaded by, and the properties reported
// by, a loader instance.
type loaderParams struct {
	PackageOpts

	allClosesOverTests bool // Does the "all" pattern include the transitive closure of tests of packages in "all"?
	allPatternIsRoot   bool // Is the "all" pattern an additional root?

	listRoots func() []string
}

func (ld *loader) reset() {
	select {
	case <-ld.work.Idle():
	default:
		panic("loader.reset when not idle")
	}

	ld.roots = nil
	ld.pkgCache = new(par.Cache)
	ld.pkgs = nil
}

// A loadPkg records information about a single loaded package.
type loadPkg struct {
	// Populated at construction time:
	path   string // import path
	testOf *loadPkg

	// Populated at construction time and updated by (*loader).applyPkgFlags:
	flags atomicLoadPkgFlags

	// Populated by (*loader).load:
	mod         module.Version // module providing package
	dir         string         // directory containing source code
	err         error          // error loading package
	imports     []*loadPkg     // packages imported by this one
	testImports []string       // test-only imports, saved for use by pkg.test.
	inStd       bool

	// Populated by (*loader).pkgTest:
	testOnce sync.Once
	test     *loadPkg

	// Populated by postprocessing in (*loader).buildStacks:
	stack *loadPkg // package importing this one in minimal import stack for this pkg
}

// loadPkgFlags is a set of flags tracking metadata about a package.
type loadPkgFlags int8

const (
	// pkgInAll indicates that the package is in the "all" package pattern,
	// regardless of whether we are loading the "all" package pattern.
	//
	// When the pkgInAll flag and pkgImportsLoaded flags are both set, the caller
	// who set the last of those flags must propagate the pkgInAll marking to all
	// of the imports of the marked package.
	//
	// A test is marked with pkgInAll if that test would promote the packages it
	// imports to be in "all" (such as when the test is itself within the main
	// module, or when ld.allClosesOverTests is true).
	pkgInAll loadPkgFlags = 1 << iota

	// pkgIsRoot indicates that the package matches one of the root package
	// patterns requested by the caller.
	//
	// If LoadTests is set, then when pkgIsRoot and pkgImportsLoaded are both set,
	// the caller who set the last of those flags must populate a test for the
	// package (in the pkg.test field).
	//
	// If the "all" pattern is included as a root, then non-test packages in "all"
	// are also roots (and must be marked pkgIsRoot).
	pkgIsRoot

	// pkgImportsLoaded indicates that the imports and testImports fields of a
	// loadPkg have been populated.
	pkgImportsLoaded
)

// has reports whether all of the flags in cond are set in f.
func (f loadPkgFlags) has(cond loadPkgFlags) bool {
	return f&cond == cond
}

// An atomicLoadPkgFlags stores a loadPkgFlags for which individual flags can be
// added atomically.
type atomicLoadPkgFlags struct {
	bits int32
}

// update sets the given flags in af (in addition to any flags already set).
//
// update returns the previous flag state so that the caller may determine which
// flags were newly-set.
func (af *atomicLoadPkgFlags) update(flags loadPkgFlags) (old loadPkgFlags) {
	for {
		old := atomic.LoadInt32(&af.bits)
		new := old | int32(flags)
		if new == old || atomic.CompareAndSwapInt32(&af.bits, old, new) {
			return loadPkgFlags(old)
		}
	}
}

// has reports whether all of the flags in cond are set in af.
func (af *atomicLoadPkgFlags) has(cond loadPkgFlags) bool {
	return loadPkgFlags(atomic.LoadInt32(&af.bits))&cond == cond
}

// isTest reports whether pkg is a test of another package.
func (pkg *loadPkg) isTest() bool {
	return pkg.testOf != nil
}

var errMissing = errors.New("cannot find package")

// loadFromRoots attempts to load the build graph needed to process a set of
// root packages and their dependencies.
//
// The set of root packages is returned by the params.listRoots function, and
// expanded to the full set of packages by tracing imports (and possibly tests)
// as needed.
func loadFromRoots(params loaderParams) *loader {
	ld := &loader{
		loaderParams: params,
		work:         par.NewQueue(runtime.GOMAXPROCS(0)),
	}

	var err error
	reqs := &mvsReqs{buildList: buildList}
	buildList, err = mvs.BuildList(Target, reqs)
	if err != nil {
		base.Fatalf("go: %v", err)
	}

	addedModuleFor := make(map[string]bool)
	for {
		ld.reset()

		// Load the root packages and their imports.
		// Note: the returned roots can change on each iteration,
		// since the expansion of package patterns depends on the
		// build list we're using.
		inRoots := map[*loadPkg]bool{}
		for _, path := range ld.listRoots() {
			root := ld.pkg(path, pkgIsRoot)
			if !inRoots[root] {
				ld.roots = append(ld.roots, root)
				inRoots[root] = true
			}
		}

		// ld.pkg adds imported packages to the work queue and calls applyPkgFlags,
		// which adds tests (and test dependencies) as needed.
		//
		// When all of the work in the queue has completed, we'll know that the
		// transitive closure of dependencies has been loaded.
		<-ld.work.Idle()

		ld.buildStacks()

		if !ld.ResolveMissingImports || (!HasModRoot() && !allowMissingModuleImports) {
			// We've loaded as much as we can without resolving missing imports.
			break
		}
		modAddedBy := ld.resolveMissingImports(addedModuleFor)
		if len(modAddedBy) == 0 {
			break
		}

		// Recompute buildList with all our additions.
		reqs = &mvsReqs{buildList: buildList}
		buildList, err = mvs.BuildList(Target, reqs)
		if err != nil {
			// If an error was found in a newly added module, report the package
			// import stack instead of the module requirement stack. Packages
			// are more descriptive.
			if err, ok := err.(*mvs.BuildListError); ok {
				if pkg := modAddedBy[err.Module()]; pkg != nil {
					base.Fatalf("go: %s: %v", pkg.stackText(), err.Err)
				}
			}
			base.Fatalf("go: %v", err)
		}
	}
	base.ExitIfErrors()

	// Compute directly referenced dependency modules.
	ld.direct = make(map[string]bool)
	for _, pkg := range ld.pkgs {
		if pkg.mod == Target {
			for _, dep := range pkg.imports {
				if dep.mod.Path != "" && dep.mod.Path != Target.Path && index != nil {
					_, explicit := index.require[dep.mod]
					if allowWriteGoMod && cfg.BuildMod == "readonly" && !explicit {
						// TODO(#40775): attach error to package instead of using
						// base.Errorf. Ideally, 'go list' should not fail because of this,
						// but today, LoadPackages calls WriteGoMod unconditionally, which
						// would fail with a less clear message.
						base.Errorf("go: %[1]s: package %[2]s imported from implicitly required module; to add missing requirements, run:\n\tgo get %[2]s@%[3]s", pkg.path, dep.path, dep.mod.Version)
					}
					ld.direct[dep.mod.Path] = true
				}
			}
		}
	}
	base.ExitIfErrors()

	// If we didn't scan all of the imports from the main module, or didn't use
	// imports.AnyTags, then we didn't necessarily load every package that
	// contributes “direct” imports — so we can't safely mark existing
	// dependencies as indirect-only.
	// Conservatively mark those dependencies as direct.
	if modFile != nil && (!ld.allPatternIsRoot || !reflect.DeepEqual(ld.Tags, imports.AnyTags())) {
		for _, r := range modFile.Require {
			if !r.Indirect {
				ld.direct[r.Mod.Path] = true
			}
		}
	}

	return ld
}

// resolveMissingImports adds module dependencies to the global build list
// in order to resolve missing packages from pkgs.
//
// The newly-resolved packages are added to the addedModuleFor map, and
// resolveMissingImports returns a map from each newly-added module version to
// the first package for which that module was added.
func (ld *loader) resolveMissingImports(addedModuleFor map[string]bool) (modAddedBy map[module.Version]*loadPkg) {
	var needPkgs []*loadPkg
	for _, pkg := range ld.pkgs {
		if pkg.err == nil {
			continue
		}
		if pkg.isTest() {
			// If we are missing a test, we are also missing its non-test version, and
			// we should only add the missing import once.
			continue
		}
		if !errors.As(pkg.err, new(*ImportMissingError)) {
			// Leave other errors for Import or load.Packages to report.
			continue
		}

		needPkgs = append(needPkgs, pkg)

		pkg := pkg
		ld.work.Add(func() {
			pkg.mod, pkg.err = queryImport(context.TODO(), pkg.path)
		})
	}
	<-ld.work.Idle()

	modAddedBy = map[module.Version]*loadPkg{}
	for _, pkg := range needPkgs {
		if pkg.err != nil {
			continue
		}

		fmt.Fprintf(os.Stderr, "go: found %s in %s %s\n", pkg.path, pkg.mod.Path, pkg.mod.Version)
		if addedModuleFor[pkg.path] {
			// TODO(bcmills): This should only be an error if pkg.mod is the same
			// version we already tried to add previously.
			base.Fatalf("go: %s: looping trying to add package", pkg.stackText())
		}
		if modAddedBy[pkg.mod] == nil {
			modAddedBy[pkg.mod] = pkg
			buildList = append(buildList, pkg.mod)
		}
	}

	return modAddedBy
}

// pkg locates the *loadPkg for path, creating and queuing it for loading if
// needed, and updates its state to reflect the given flags.
//
// The imports of the returned *loadPkg will be loaded asynchronously in the
// ld.work queue, and its test (if requested) will also be populated once
// imports have been resolved. When ld.work goes idle, all transitive imports of
// the requested package (and its test, if requested) will have been loaded.
func (ld *loader) pkg(path string, flags loadPkgFlags) *loadPkg {
	if flags.has(pkgImportsLoaded) {
		panic("internal error: (*loader).pkg called with pkgImportsLoaded flag set")
	}

	pkg := ld.pkgCache.Do(path, func() interface{} {
		pkg := &loadPkg{
			path: path,
		}
		ld.applyPkgFlags(pkg, flags)

		ld.work.Add(func() { ld.load(pkg) })
		return pkg
	}).(*loadPkg)

	ld.applyPkgFlags(pkg, flags)
	return pkg
}

// applyPkgFlags updates pkg.flags to set the given flags and propagate the
// (transitive) effects of those flags, possibly loading or enqueueing further
// packages as a result.
func (ld *loader) applyPkgFlags(pkg *loadPkg, flags loadPkgFlags) {
	if flags == 0 {
		return
	}

	if flags.has(pkgInAll) && ld.allPatternIsRoot && !pkg.isTest() {
		// This package matches a root pattern by virtue of being in "all".
		flags |= pkgIsRoot
	}

	old := pkg.flags.update(flags)
	new := old | flags
	if new == old || !new.has(pkgImportsLoaded) {
		// We either didn't change the state of pkg, or we don't know anything about
		// its dependencies yet. Either way, we can't usefully load its test or
		// update its dependencies.
		return
	}

	if !pkg.isTest() {
		// Check whether we should add (or update the flags for) a test for pkg.
		// ld.pkgTest is idempotent and extra invocations are inexpensive,
		// so it's ok if we call it more than is strictly necessary.
		wantTest := false
		switch {
		case ld.allPatternIsRoot && pkg.mod == Target:
			// We are loading the "all" pattern, which includes packages imported by
			// tests in the main module. This package is in the main module, so we
			// need to identify the imports of its test even if LoadTests is not set.
			//
			// (We will filter out the extra tests explicitly in computePatternAll.)
			wantTest = true

		case ld.allPatternIsRoot && ld.allClosesOverTests && new.has(pkgInAll):
			// This variant of the "all" pattern includes imports of tests of every
			// package that is itself in "all", and pkg is in "all", so its test is
			// also in "all" (as above).
			wantTest = true

		case ld.LoadTests && new.has(pkgIsRoot):
			// LoadTest explicitly requests tests of “the root packages”.
			wantTest = true
		}

		if wantTest {
			var testFlags loadPkgFlags
			if pkg.mod == Target || (ld.allClosesOverTests && new.has(pkgInAll)) {
				// Tests of packages in the main module are in "all", in the sense that
				// they cause the packages they import to also be in "all". So are tests
				// of packages in "all" if "all" closes over test dependencies.
				testFlags |= pkgInAll
			}
			ld.pkgTest(pkg, testFlags)
		}
	}

	if new.has(pkgInAll) && !old.has(pkgInAll|pkgImportsLoaded) {
		// We have just marked pkg with pkgInAll, or we have just loaded its
		// imports, or both. Now is the time to propagate pkgInAll to the imports.
		for _, dep := range pkg.imports {
			ld.applyPkgFlags(dep, pkgInAll)
		}
	}
}

// load loads an individual package.
func (ld *loader) load(pkg *loadPkg) {
	if strings.Contains(pkg.path, "@") {
		// Leave for error during load.
		return
	}
	if build.IsLocalImport(pkg.path) || filepath.IsAbs(pkg.path) {
		// Leave for error during load.
		// (Module mode does not allow local imports.)
		return
	}

	if search.IsMetaPackage(pkg.path) {
		pkg.err = &invalidImportError{
			importPath: pkg.path,
			err:        fmt.Errorf("%q is not an importable package; see 'go help packages'", pkg.path),
		}
		return
	}

	pkg.mod, pkg.dir, pkg.err = importFromBuildList(context.TODO(), pkg.path, buildList)
	if pkg.dir == "" {
		return
	}
	if pkg.mod == Target {
		// Go ahead and mark pkg as in "all". This provides the invariant that a
		// package that is *only* imported by other packages in "all" is always
		// marked as such before loading its imports.
		//
		// We don't actually rely on that invariant at the moment, but it may
		// improve efficiency somewhat and makes the behavior a bit easier to reason
		// about (by reducing churn on the flag bits of dependencies), and costs
		// essentially nothing (these atomic flag ops are essentially free compared
		// to scanning source code for imports).
		ld.applyPkgFlags(pkg, pkgInAll)
	}
	if ld.AllowPackage != nil {
		if err := ld.AllowPackage(context.TODO(), pkg.path, pkg.mod); err != nil {
			pkg.err = err
		}
	}

	pkg.inStd = (search.IsStandardImportPath(pkg.path) && search.InDir(pkg.dir, cfg.GOROOTsrc) != "")

	var imports, testImports []string

	if cfg.BuildContext.Compiler == "gccgo" && pkg.inStd {
		// We can't scan standard packages for gccgo.
	} else {
		var err error
		imports, testImports, err = scanDir(pkg.dir, ld.Tags)
		if err != nil {
			pkg.err = err
			return
		}
	}

	pkg.imports = make([]*loadPkg, 0, len(imports))
	var importFlags loadPkgFlags
	if pkg.flags.has(pkgInAll) {
		importFlags = pkgInAll
	}
	for _, path := range imports {
		if pkg.inStd {
			// Imports from packages in "std" and "cmd" should resolve using
			// GOROOT/src/vendor even when "std" is not the main module.
			path = ld.stdVendor(pkg.path, path)
		}
		pkg.imports = append(pkg.imports, ld.pkg(path, importFlags))
	}
	pkg.testImports = testImports

	ld.applyPkgFlags(pkg, pkgImportsLoaded)
}

// pkgTest locates the test of pkg, creating it if needed, and updates its state
// to reflect the given flags.
//
// pkgTest requires that the imports of pkg have already been loaded (flagged
// with pkgImportsLoaded).
func (ld *loader) pkgTest(pkg *loadPkg, testFlags loadPkgFlags) *loadPkg {
	if pkg.isTest() {
		panic("pkgTest called on a test package")
	}

	createdTest := false
	pkg.testOnce.Do(func() {
		pkg.test = &loadPkg{
			path:   pkg.path,
			testOf: pkg,
			mod:    pkg.mod,
			dir:    pkg.dir,
			err:    pkg.err,
			inStd:  pkg.inStd,
		}
		ld.applyPkgFlags(pkg.test, testFlags)
		createdTest = true
	})

	test := pkg.test
	if createdTest {
		test.imports = make([]*loadPkg, 0, len(pkg.testImports))
		var importFlags loadPkgFlags
		if test.flags.has(pkgInAll) {
			importFlags = pkgInAll
		}
		for _, path := range pkg.testImports {
			if pkg.inStd {
				path = ld.stdVendor(test.path, path)
			}
			test.imports = append(test.imports, ld.pkg(path, importFlags))
		}
		pkg.testImports = nil
		ld.applyPkgFlags(test, pkgImportsLoaded)
	} else {
		ld.applyPkgFlags(test, testFlags)
	}

	return test
}

// stdVendor returns the canonical import path for the package with the given
// path when imported from the standard-library package at parentPath.
func (ld *loader) stdVendor(parentPath, path string) string {
	if search.IsStandardImportPath(path) {
		return path
	}

	if str.HasPathPrefix(parentPath, "cmd") {
		if Target.Path != "cmd" {
			vendorPath := pathpkg.Join("cmd", "vendor", path)
			if _, err := os.Stat(filepath.Join(cfg.GOROOTsrc, filepath.FromSlash(vendorPath))); err == nil {
				return vendorPath
			}
		}
	} else if Target.Path != "std" || str.HasPathPrefix(parentPath, "vendor") {
		// If we are outside of the 'std' module, resolve imports from within 'std'
		// to the vendor directory.
		//
		// Do the same for importers beginning with the prefix 'vendor/' even if we
		// are *inside* of the 'std' module: the 'vendor/' packages that resolve
		// globally from GOROOT/src/vendor (and are listed as part of 'go list std')
		// are distinct from the real module dependencies, and cannot import
		// internal packages from the real module.
		//
		// (Note that although the 'vendor/' packages match the 'std' *package*
		// pattern, they are not part of the std *module*, and do not affect
		// 'go mod tidy' and similar module commands when working within std.)
		vendorPath := pathpkg.Join("vendor", path)
		if _, err := os.Stat(filepath.Join(cfg.GOROOTsrc, filepath.FromSlash(vendorPath))); err == nil {
			return vendorPath
		}
	}

	// Not vendored: resolve from modules.
	return path
}

// computePatternAll returns the list of packages matching pattern "all",
// starting with a list of the import paths for the packages in the main module.
func (ld *loader) computePatternAll() (all []string) {
	for _, pkg := range ld.pkgs {
		if pkg.flags.has(pkgInAll) && !pkg.isTest() {
			all = append(all, pkg.path)
		}
	}
	sort.Strings(all)
	return all
}

// scanDir is like imports.ScanDir but elides known magic imports from the list,
// so that we do not go looking for packages that don't really exist.
//
// The standard magic import is "C", for cgo.
//
// The only other known magic imports are appengine and appengine/*.
// These are so old that they predate "go get" and did not use URL-like paths.
// Most code today now uses google.golang.org/appengine instead,
// but not all code has been so updated. When we mostly ignore build tags
// during "go vendor", we look into "// +build appengine" files and
// may see these legacy imports. We drop them so that the module
// search does not look for modules to try to satisfy them.
func scanDir(dir string, tags map[string]bool) (imports_, testImports []string, err error) {
	imports_, testImports, err = imports.ScanDir(dir, tags)

	filter := func(x []string) []string {
		w := 0
		for _, pkg := range x {
			if pkg != "C" && pkg != "appengine" && !strings.HasPrefix(pkg, "appengine/") &&
				pkg != "appengine_internal" && !strings.HasPrefix(pkg, "appengine_internal/") {
				x[w] = pkg
				w++
			}
		}
		return x[:w]
	}

	return filter(imports_), filter(testImports), err
}

// buildStacks computes minimal import stacks for each package,
// for use in error messages. When it completes, packages that
// are part of the original root set have pkg.stack == nil,
// and other packages have pkg.stack pointing at the next
// package up the import stack in their minimal chain.
// As a side effect, buildStacks also constructs ld.pkgs,
// the list of all packages loaded.
func (ld *loader) buildStacks() {
	if len(ld.pkgs) > 0 {
		panic("buildStacks")
	}
	for _, pkg := range ld.roots {
		pkg.stack = pkg // sentinel to avoid processing in next loop
		ld.pkgs = append(ld.pkgs, pkg)
	}
	for i := 0; i < len(ld.pkgs); i++ { // not range: appending to ld.pkgs in loop
		pkg := ld.pkgs[i]
		for _, next := range pkg.imports {
			if next.stack == nil {
				next.stack = pkg
				ld.pkgs = append(ld.pkgs, next)
			}
		}
		if next := pkg.test; next != nil && next.stack == nil {
			next.stack = pkg
			ld.pkgs = append(ld.pkgs, next)
		}
	}
	for _, pkg := range ld.roots {
		pkg.stack = nil
	}
}

// stackText builds the import stack text to use when
// reporting an error in pkg. It has the general form
//
//	root imports
//		other imports
//		other2 tested by
//		other2.test imports
//		pkg
//
func (pkg *loadPkg) stackText() string {
	var stack []*loadPkg
	for p := pkg; p != nil; p = p.stack {
		stack = append(stack, p)
	}

	var buf bytes.Buffer
	for i := len(stack) - 1; i >= 0; i-- {
		p := stack[i]
		fmt.Fprint(&buf, p.path)
		if p.testOf != nil {
			fmt.Fprint(&buf, ".test")
		}
		if i > 0 {
			if stack[i-1].testOf == p {
				fmt.Fprint(&buf, " tested by\n\t")
			} else {
				fmt.Fprint(&buf, " imports\n\t")
			}
		}
	}
	return buf.String()
}

// why returns the text to use in "go mod why" output about the given package.
// It is less ornate than the stackText but contains the same information.
func (pkg *loadPkg) why() string {
	var buf strings.Builder
	var stack []*loadPkg
	for p := pkg; p != nil; p = p.stack {
		stack = append(stack, p)
	}

	for i := len(stack) - 1; i >= 0; i-- {
		p := stack[i]
		if p.testOf != nil {
			fmt.Fprintf(&buf, "%s.test\n", p.testOf.path)
		} else {
			fmt.Fprintf(&buf, "%s\n", p.path)
		}
	}
	return buf.String()
}

// Why returns the "go mod why" output stanza for the given package,
// without the leading # comment.
// The package graph must have been loaded already, usually by LoadPackages.
// If there is no reason for the package to be in the current build,
// Why returns an empty string.
func Why(path string) string {
	pkg, ok := loaded.pkgCache.Get(path).(*loadPkg)
	if !ok {
		return ""
	}
	return pkg.why()
}

// WhyDepth returns the number of steps in the Why listing.
// If there is no reason for the package to be in the current build,
// WhyDepth returns 0.
func WhyDepth(path string) int {
	n := 0
	pkg, _ := loaded.pkgCache.Get(path).(*loadPkg)
	for p := pkg; p != nil; p = p.stack {
		n++
	}
	return n
}
