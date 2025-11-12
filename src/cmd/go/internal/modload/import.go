// Copyright 2018 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

package modload

import (
	"context"
	"errors"
	"fmt"
	"go/build"
	"io/fs"
	"os"
	pathpkg "path"
	"path/filepath"
	"sort"
	"strings"

	"cmd/go/internal/cfg"
	"cmd/go/internal/fsys"
	"cmd/go/internal/gover"
	"cmd/go/internal/modfetch"
	"cmd/go/internal/modindex"
	"cmd/go/internal/search"
	"cmd/go/internal/str"
	"cmd/internal/par"

	"golang.org/x/mod/module"
)

type ImportMissingError struct {
	Path                      string
	Module                    module.Version
	QueryErr                  error
	modContainingCWD          module.Version
	allowMissingModuleImports bool

	// modRoot is dependent on the value of ImportingMainModule and should be
	// kept in sync.
	modRoot             string
	ImportingMainModule module.Version

	// isStd indicates whether we would expect to find the package in the standard
	// library. This is normally true for all dotless import paths, but replace
	// directives can cause us to treat the replaced paths as also being in
	// modules.
	isStd bool

	// importerGoVersion is the version the module containing the import error
	// specified. It is only set when isStd is true.
	importerGoVersion string

	// replaced the highest replaced version of the module where the replacement
	// contains the package. replaced is only set if the replacement is unused.
	replaced module.Version

	// newMissingVersion is set to a newer version of Module if one is present
	// in the build list. When set, we can't automatically upgrade.
	newMissingVersion string
}

func (e *ImportMissingError) Error() string {
	if e.Module.Path == "" {
		if e.isStd {
			msg := fmt.Sprintf("package %s is not in std (%s)", e.Path, filepath.Join(cfg.GOROOT, "src", e.Path))
			if e.importerGoVersion != "" {
				msg += fmt.Sprintf("\nnote: imported by a module that requires go %s", e.importerGoVersion)
			}
			return msg
		}
		if e.QueryErr != nil && !errors.Is(e.QueryErr, ErrNoModRoot) {
			return fmt.Sprintf("cannot find module providing package %s: %v", e.Path, e.QueryErr)
		}
		if cfg.BuildMod == "mod" || (cfg.BuildMod == "readonly" && e.allowMissingModuleImports) {
			return "cannot find module providing package " + e.Path
		}

		if e.replaced.Path != "" {
			suggestArg := e.replaced.Path
			if !module.IsZeroPseudoVersion(e.replaced.Version) {
				suggestArg = e.replaced.String()
			}
			return fmt.Sprintf("module %s provides package %s and is replaced but not required; to add it:\n\tgo get %s", e.replaced.Path, e.Path, suggestArg)
		}

		message := fmt.Sprintf("no required module provides package %s", e.Path)
		if e.QueryErr != nil {
			return fmt.Sprintf("%s: %v", message, e.QueryErr)
		}
		if e.ImportingMainModule.Path != "" && e.ImportingMainModule != e.modContainingCWD {
			return fmt.Sprintf("%s; to add it:\n\tcd %s\n\tgo get %s", message, e.modRoot, e.Path)
		}
		return fmt.Sprintf("%s; to add it:\n\tgo get %s", message, e.Path)
	}

	if e.newMissingVersion != "" {
		return fmt.Sprintf("package %s provided by %s at latest version %s but not at required version %s", e.Path, e.Module.Path, e.Module.Version, e.newMissingVersion)
	}

	return fmt.Sprintf("missing module for import: %s@%s provides %s", e.Module.Path, e.Module.Version, e.Path)
}

func (e *ImportMissingError) Unwrap() error {
	return e.QueryErr
}

func (e *ImportMissingError) ImportPath() string {
	return e.Path
}

// An AmbiguousImportError indicates an import of a package found in multiple
// modules in the build list, or found in both the main module and its vendor
// directory.
type AmbiguousImportError struct {
	importPath string
	Dirs       []string
	Modules    []module.Version // Either empty or 1:1 with Dirs.
}

func (e *AmbiguousImportError) ImportPath() string {
	return e.importPath
}

func (e *AmbiguousImportError) Error() string {
	locType := "modules"
	if len(e.Modules) == 0 {
		locType = "directories"
	}

	var buf strings.Builder
	fmt.Fprintf(&buf, "ambiguous import: found package %s in multiple %s:", e.importPath, locType)

	for i, dir := range e.Dirs {
		buf.WriteString("\n\t")
		if i < len(e.Modules) {
			m := e.Modules[i]
			buf.WriteString(m.Path)
			if m.Version != "" {
				fmt.Fprintf(&buf, " %s", m.Version)
			}
			fmt.Fprintf(&buf, " (%s)", dir)
		} else {
			buf.WriteString(dir)
		}
	}

	return buf.String()
}

// A DirectImportFromImplicitDependencyError indicates a package directly
// imported by a package or test in the main module that is satisfied by a
// dependency that is not explicit in the main module's go.mod file.
type DirectImportFromImplicitDependencyError struct {
	ImporterPath string
	ImportedPath string
	Module       module.Version
}

func (e *DirectImportFromImplicitDependencyError) Error() string {
	return fmt.Sprintf("package %s imports %s from implicitly required module; to add missing requirements, run:\n\tgo get %s@%s", e.ImporterPath, e.ImportedPath, e.Module.Path, e.Module.Version)
}

func (e *DirectImportFromImplicitDependencyError) ImportPath() string {
	return e.ImporterPath
}

// ImportMissingSumError is reported in readonly mode when we need to check
// if a module contains a package, but we don't have a sum for its .zip file.
// We might need sums for multiple modules to verify the package is unique.
//
// TODO(#43653): consolidate multiple errors of this type into a single error
// that suggests a 'go get' command for root packages that transitively import
// packages from modules with missing sums. load.CheckPackageErrors would be
// a good place to consolidate errors, but we'll need to attach the import
// stack here.
type ImportMissingSumError struct {
	importPath                string
	found                     bool
	mods                      []module.Version
	importer, importerVersion string // optional, but used for additional context
	importerIsTest            bool
}

func (e *ImportMissingSumError) Error() string {
	var importParen string
	if e.importer != "" {
		importParen = fmt.Sprintf(" (imported by %s)", e.importer)
	}
	var message string
	if e.found {
		message = fmt.Sprintf("missing go.sum entry needed to verify package %s%s is provided by exactly one module", e.importPath, importParen)
	} else {
		message = fmt.Sprintf("missing go.sum entry for module providing package %s%s", e.importPath, importParen)
	}
	var hint string
	if e.importer == "" {
		// Importing package is unknown, or the missing package was named on the
		// command line. Recommend 'go mod download' for the modules that could
		// provide the package, since that shouldn't change go.mod.
		if len(e.mods) > 0 {
			args := make([]string, len(e.mods))
			for i, mod := range e.mods {
				args[i] = mod.Path
			}
			hint = fmt.Sprintf("; to add:\n\tgo mod download %s", strings.Join(args, " "))
		}
	} else {
		// Importing package is known (common case). Recommend 'go get' on the
		// current version of the importing package.
		tFlag := ""
		if e.importerIsTest {
			tFlag = " -t"
		}
		version := ""
		if e.importerVersion != "" {
			version = "@" + e.importerVersion
		}
		hint = fmt.Sprintf("; to add:\n\tgo get%s %s%s", tFlag, e.importer, version)
	}
	return message + hint
}

func (e *ImportMissingSumError) ImportPath() string {
	return e.importPath
}

type invalidImportError struct {
	importPath string
	err        error
}

func (e *invalidImportError) ImportPath() string {
	return e.importPath
}

func (e *invalidImportError) Error() string {
	return e.err.Error()
}

func (e *invalidImportError) Unwrap() error {
	return e.err
}

// importFromModules finds the module and directory in the dependency graph of
// rs containing the package with the given import path. If mg is nil,
// importFromModules attempts to locate the module using only the main module
// and the roots of rs before it loads the full graph.
//
// The answer must be unique: importFromModules returns an error if multiple
// modules are observed to provide the same package.
//
// importFromModules can return a module with an empty m.Path, for packages in
// the standard library.
//
// importFromModules can return an empty directory string, for fake packages
// like "C" and "unsafe".
//
// If the package is not present in any module selected from the requirement
// graph, importFromModules returns an *ImportMissingError.
//
// If the package is present in exactly one module, importFromModules will
// return the module, its root directory, and a list of other modules that
// lexically could have provided the package but did not.
//
// If skipModFile is true, the go.mod file for the package is not loaded. This
// allows 'go mod tidy' to preserve a minor checksum-preservation bug
// (https://go.dev/issue/56222) for modules with 'go' versions between 1.17 and
// 1.20, preventing unnecessary go.sum churn and network access in those
// modules.
func importFromModules(loaderstate *State, ctx context.Context, path string, rs *Requirements, mg *ModuleGraph, skipModFile bool) (m module.Version, modroot, dir string, altMods []module.Version, err error) {
	invalidf := func(format string, args ...any) (module.Version, string, string, []module.Version, error) {
		return module.Version{}, "", "", nil, &invalidImportError{
			importPath: path,
			err:        fmt.Errorf(format, args...),
		}
	}

	if strings.Contains(path, "@") {
		return invalidf("import path %q should not have @version", path)
	}
	if build.IsLocalImport(path) {
		return invalidf("%q is relative, but relative import paths are not supported in module mode", path)
	}
	if filepath.IsAbs(path) {
		return invalidf("%q is not a package path; see 'go help packages'", path)
	}
	if search.IsMetaPackage(path) {
		return invalidf("%q is not an importable package; see 'go help packages'", path)
	}

	if path == "C" {
		// There's no directory for import "C".
		return module.Version{}, "", "", nil, nil
	}
	// Before any further lookup, check that the path is valid.
	if err := module.CheckImportPath(path); err != nil {
		return module.Version{}, "", "", nil, &invalidImportError{importPath: path, err: err}
	}

	// Check each module on the build list.
	var dirs, roots []string
	var mods []module.Version

	// Is the package in the standard library?
	pathIsStd := search.IsStandardImportPath(path)
	if pathIsStd && modindex.IsStandardPackage(cfg.GOROOT, cfg.BuildContext.Compiler, path) {
		for _, mainModule := range loaderstate.MainModules.Versions() {
			if loaderstate.MainModules.InGorootSrc(mainModule) {
				if dir, ok, err := dirInModule(path, loaderstate.MainModules.PathPrefix(mainModule), loaderstate.MainModules.ModRoot(mainModule), true); err != nil {
					return module.Version{}, loaderstate.MainModules.ModRoot(mainModule), dir, nil, err
				} else if ok {
					return mainModule, loaderstate.MainModules.ModRoot(mainModule), dir, nil, nil
				}
			}
		}
		dir := filepath.Join(cfg.GOROOTsrc, path)
		modroot = cfg.GOROOTsrc
		if str.HasPathPrefix(path, "cmd") {
			modroot = filepath.Join(cfg.GOROOTsrc, "cmd")
		}
		dirs = append(dirs, dir)
		roots = append(roots, modroot)
		mods = append(mods, module.Version{})
	}
	// -mod=vendor is special.
	// Everything must be in the main modules or the main module's or workspace's vendor directory.
	if cfg.BuildMod == "vendor" {
		var mainErr error
		for _, mainModule := range loaderstate.MainModules.Versions() {
			modRoot := loaderstate.MainModules.ModRoot(mainModule)
			if modRoot != "" {
				dir, mainOK, err := dirInModule(path, loaderstate.MainModules.PathPrefix(mainModule), modRoot, true)
				if mainErr == nil {
					mainErr = err
				}
				if mainOK {
					mods = append(mods, mainModule)
					dirs = append(dirs, dir)
					roots = append(roots, modRoot)
				}
			}
		}

		if loaderstate.HasModRoot() {
			vendorDir := VendorDir(loaderstate)
			dir, inVendorDir, _ := dirInModule(path, "", vendorDir, false)
			if inVendorDir {
				readVendorList(vendorDir)
				// If vendorPkgModule does not contain an entry for path then it's probably either because
				// vendor/modules.txt does not exist or the user manually added directories to the vendor directory.
				// Go 1.23 and later require vendored packages to be present in modules.txt to be imported.
				_, ok := vendorPkgModule[path]
				if ok || (gover.Compare(loaderstate.MainModules.GoVersion(loaderstate), gover.ExplicitModulesTxtImportVersion) < 0) {
					mods = append(mods, vendorPkgModule[path])
					dirs = append(dirs, dir)
					roots = append(roots, vendorDir)
				} else {
					subCommand := "mod"
					if loaderstate.inWorkspaceMode() {
						subCommand = "work"
					}
					fmt.Fprintf(os.Stderr, "go: ignoring package %s which exists in the vendor directory but is missing from vendor/modules.txt. To sync the vendor directory run go %s vendor.\n", path, subCommand)
				}
			}
		}

		if len(dirs) > 1 {
			return module.Version{}, "", "", nil, &AmbiguousImportError{importPath: path, Dirs: dirs}
		}

		if mainErr != nil {
			return module.Version{}, "", "", nil, mainErr
		}

		if len(mods) == 0 {
			return module.Version{}, "", "", nil, &ImportMissingError{
				Path:                      path,
				modContainingCWD:          loaderstate.MainModules.ModContainingCWD(),
				allowMissingModuleImports: loaderstate.allowMissingModuleImports,
			}
		}

		return mods[0], roots[0], dirs[0], nil, nil
	}

	// Iterate over possible modules for the path, not all selected modules.
	// Iterating over selected modules would make the overall loading time
	// O(M × P) for M modules providing P imported packages, whereas iterating
	// over path prefixes is only O(P × k) with maximum path depth k. For
	// large projects both M and P may be very large (note that M ≤ P), but k
	// will tend to remain smallish (if for no other reason than filesystem
	// path limitations).
	//
	// We perform this iteration either one or two times. If mg is initially nil,
	// then we first attempt to load the package using only the main module and
	// its root requirements. If that does not identify the package, or if mg is
	// already non-nil, then we attempt to load the package using the full
	// requirements in mg.
	for {
		var sumErrMods, altMods []module.Version
		for prefix := path; prefix != "."; prefix = pathpkg.Dir(prefix) {
			if gover.IsToolchain(prefix) {
				// Do not use the synthetic "go" module for "go/ast".
				continue
			}
			var (
				v  string
				ok bool
			)
			if mg == nil {
				v, ok = rs.rootSelected(loaderstate, prefix)
			} else {
				v, ok = mg.Selected(prefix), true
			}
			if !ok || v == "none" {
				continue
			}
			m := module.Version{Path: prefix, Version: v}

			root, isLocal, err := fetch(loaderstate, ctx, m)
			if err != nil {
				if _, ok := errors.AsType[*sumMissingError](err); ok {
					// We are missing a sum needed to fetch a module in the build list.
					// We can't verify that the package is unique, and we may not find
					// the package at all. Keep checking other modules to decide which
					// error to report. Multiple sums may be missing if we need to look in
					// multiple nested modules to resolve the import; we'll report them all.
					sumErrMods = append(sumErrMods, m)
					continue
				}
				// Report fetch error.
				// Note that we don't know for sure this module is necessary,
				// but it certainly _could_ provide the package, and even if we
				// continue the loop and find the package in some other module,
				// we need to look at this module to make sure the import is
				// not ambiguous.
				return module.Version{}, "", "", nil, err
			}
			if dir, ok, err := dirInModule(path, m.Path, root, isLocal); err != nil {
				return module.Version{}, "", "", nil, err
			} else if ok {
				mods = append(mods, m)
				roots = append(roots, root)
				dirs = append(dirs, dir)
			} else {
				altMods = append(altMods, m)
			}
		}

		if len(mods) > 1 {
			// We produce the list of directories from longest to shortest candidate
			// module path, but the AmbiguousImportError should report them from
			// shortest to longest. Reverse them now.
			for i := 0; i < len(mods)/2; i++ {
				j := len(mods) - 1 - i
				mods[i], mods[j] = mods[j], mods[i]
				roots[i], roots[j] = roots[j], roots[i]
				dirs[i], dirs[j] = dirs[j], dirs[i]
			}
			return module.Version{}, "", "", nil, &AmbiguousImportError{importPath: path, Dirs: dirs, Modules: mods}
		}

		if len(sumErrMods) > 0 {
			for i := 0; i < len(sumErrMods)/2; i++ {
				j := len(sumErrMods) - 1 - i
				sumErrMods[i], sumErrMods[j] = sumErrMods[j], sumErrMods[i]
			}
			return module.Version{}, "", "", nil, &ImportMissingSumError{
				importPath: path,
				mods:       sumErrMods,
				found:      len(mods) > 0,
			}
		}

		if len(mods) == 1 {
			// We've found the unique module containing the package.
			// However, in order to actually compile it we need to know what
			// Go language version to use, which requires its go.mod file.
			//
			// If the module graph is pruned and this is a test-only dependency
			// of a package in "all", we didn't necessarily load that file
			// when we read the module graph, so do it now to be sure.
			if !skipModFile && cfg.BuildMod != "vendor" && mods[0].Path != "" && !loaderstate.MainModules.Contains(mods[0].Path) {
				if _, err := goModSummary(loaderstate, mods[0]); err != nil {
					return module.Version{}, "", "", nil, err
				}
			}
			return mods[0], roots[0], dirs[0], altMods, nil
		}

		if mg != nil {
			// We checked the full module graph and still didn't find the
			// requested package.
			var queryErr error
			if !loaderstate.HasModRoot() {
				queryErr = NewNoMainModulesError(loaderstate)
			}
			return module.Version{}, "", "", nil, &ImportMissingError{
				Path:                      path,
				QueryErr:                  queryErr,
				isStd:                     pathIsStd,
				modContainingCWD:          loaderstate.MainModules.ModContainingCWD(),
				allowMissingModuleImports: loaderstate.allowMissingModuleImports,
			}
		}

		// So far we've checked the root dependencies.
		// Load the full module graph and try again.
		mg, err = rs.Graph(loaderstate, ctx)
		if err != nil {
			// We might be missing one or more transitive (implicit) dependencies from
			// the module graph, so we can't return an ImportMissingError here — one
			// of the missing modules might actually contain the package in question,
			// in which case we shouldn't go looking for it in some new dependency.
			return module.Version{}, "", "", nil, err
		}
	}
}

// queryImport attempts to locate a module that can be added to the current
// build list to provide the package with the given import path.
//
// Unlike QueryPattern, queryImport prefers to add a replaced version of a
// module *before* checking the proxies for a version to add.
func queryImport(loaderstate *State, ctx context.Context, path string, rs *Requirements) (module.Version, error) {
	// To avoid spurious remote fetches, try the latest replacement for each
	// module (golang.org/issue/26241).
	var mods []module.Version
	if loaderstate.MainModules != nil { // TODO(#48912): Ensure MainModules exists at this point, and remove the check.
		for mp, mv := range loaderstate.MainModules.HighestReplaced() {
			if !maybeInModule(path, mp) {
				continue
			}
			if mv == "" {
				// The only replacement is a wildcard that doesn't specify a version, so
				// synthesize a pseudo-version with an appropriate major version and a
				// timestamp below any real timestamp. That way, if the main module is
				// used from within some other module, the user will be able to upgrade
				// the requirement to any real version they choose.
				if _, pathMajor, ok := module.SplitPathVersion(mp); ok && len(pathMajor) > 0 {
					mv = module.ZeroPseudoVersion(pathMajor[1:])
				} else {
					mv = module.ZeroPseudoVersion("v0")
				}
			}
			mg, err := rs.Graph(loaderstate, ctx)
			if err != nil {
				return module.Version{}, err
			}
			if gover.ModCompare(mp, mg.Selected(mp), mv) >= 0 {
				// We can't resolve the import by adding mp@mv to the module graph,
				// because the selected version of mp is already at least mv.
				continue
			}
			mods = append(mods, module.Version{Path: mp, Version: mv})
		}
	}

	// Every module path in mods is a prefix of the import path.
	// As in QueryPattern, prefer the longest prefix that satisfies the import.
	sort.Slice(mods, func(i, j int) bool {
		return len(mods[i].Path) > len(mods[j].Path)
	})
	for _, m := range mods {
		root, isLocal, err := fetch(loaderstate, ctx, m)
		if err != nil {
			if _, ok := errors.AsType[*sumMissingError](err); ok {
				return module.Version{}, &ImportMissingSumError{importPath: path}
			}
			return module.Version{}, err
		}
		if _, ok, err := dirInModule(path, m.Path, root, isLocal); err != nil {
			return m, err
		} else if ok {
			if cfg.BuildMod == "readonly" {
				return module.Version{}, &ImportMissingError{
					Path:                      path,
					replaced:                  m,
					modContainingCWD:          loaderstate.MainModules.ModContainingCWD(),
					allowMissingModuleImports: loaderstate.allowMissingModuleImports,
				}
			}
			return m, nil
		}
	}
	if len(mods) > 0 && module.CheckPath(path) != nil {
		// The package path is not valid to fetch remotely,
		// so it can only exist in a replaced module,
		// and we know from the above loop that it is not.
		replacement := Replacement(loaderstate, mods[0])
		return module.Version{}, &PackageNotInModuleError{
			Mod:         mods[0],
			Query:       "latest",
			Pattern:     path,
			Replacement: replacement,
		}
	}

	if search.IsStandardImportPath(path) {
		// This package isn't in the standard library, isn't in any module already
		// in the build list, and isn't in any other module that the user has
		// shimmed in via a "replace" directive.
		// Moreover, the import path is reserved for the standard library, so
		// QueryPattern cannot possibly find a module containing this package.
		//
		// Instead of trying QueryPattern, report an ImportMissingError immediately.
		return module.Version{}, &ImportMissingError{
			Path:                      path,
			isStd:                     true,
			modContainingCWD:          loaderstate.MainModules.ModContainingCWD(),
			allowMissingModuleImports: loaderstate.allowMissingModuleImports,
		}
	}

	if (cfg.BuildMod == "readonly" || cfg.BuildMod == "vendor") && !loaderstate.allowMissingModuleImports {
		// In readonly mode, we can't write go.mod, so we shouldn't try to look up
		// the module. If readonly mode was enabled explicitly, include that in
		// the error message.
		// In vendor mode, we cannot use the network or module cache, so we
		// shouldn't try to look up the module
		var queryErr error
		if cfg.BuildModExplicit {
			queryErr = fmt.Errorf("import lookup disabled by -mod=%s", cfg.BuildMod)
		} else if cfg.BuildModReason != "" {
			queryErr = fmt.Errorf("import lookup disabled by -mod=%s\n\t(%s)", cfg.BuildMod, cfg.BuildModReason)
		}
		return module.Version{}, &ImportMissingError{
			Path:                      path,
			QueryErr:                  queryErr,
			modContainingCWD:          loaderstate.MainModules.ModContainingCWD(),
			allowMissingModuleImports: loaderstate.allowMissingModuleImports,
		}
	}

	// Look up module containing the package, for addition to the build list.
	// Goal is to determine the module, download it to dir,
	// and return m, dir, ImportMissingError.
	fmt.Fprintf(os.Stderr, "go: finding module for package %s\n", path)

	mg, err := rs.Graph(loaderstate, ctx)
	if err != nil {
		return module.Version{}, err
	}

	candidates, err := QueryPackages(loaderstate, ctx, path, "latest", mg.Selected, loaderstate.CheckAllowed)
	if err != nil {
		if errors.Is(err, fs.ErrNotExist) {
			// Return "cannot find module providing package […]" instead of whatever
			// low-level error QueryPattern produced.
			return module.Version{}, &ImportMissingError{
				Path:                      path,
				QueryErr:                  err,
				modContainingCWD:          loaderstate.MainModules.ModContainingCWD(),
				allowMissingModuleImports: loaderstate.allowMissingModuleImports,
			}
		} else {
			return module.Version{}, err
		}
	}

	candidate0MissingVersion := ""
	for i, c := range candidates {
		if v := mg.Selected(c.Mod.Path); gover.ModCompare(c.Mod.Path, v, c.Mod.Version) > 0 {
			// QueryPattern proposed that we add module c.Mod to provide the package,
			// but we already depend on a newer version of that module (and that
			// version doesn't have the package).
			//
			// This typically happens when a package is present at the "@latest"
			// version (e.g., v1.0.0) of a module, but we have a newer version
			// of the same module in the build list (e.g., v1.0.1-beta), and
			// the package is not present there.
			if i == 0 {
				candidate0MissingVersion = v
			}
			continue
		}
		return c.Mod, nil
	}
	return module.Version{}, &ImportMissingError{
		Path:                      path,
		Module:                    candidates[0].Mod,
		newMissingVersion:         candidate0MissingVersion,
		modContainingCWD:          loaderstate.MainModules.ModContainingCWD(),
		allowMissingModuleImports: loaderstate.allowMissingModuleImports,
	}
}

// maybeInModule reports whether, syntactically,
// a package with the given import path could be supplied
// by a module with the given module path (mpath).
func maybeInModule(path, mpath string) bool {
	return mpath == path ||
		len(path) > len(mpath) && path[len(mpath)] == '/' && path[:len(mpath)] == mpath
}

var (
	haveGoModCache   par.Cache[string, bool]    // dir → bool
	haveGoFilesCache par.ErrCache[string, bool] // dir → haveGoFiles
)

// PkgIsInLocalModule reports whether the directory of the package with
// the given pkgpath, exists in the module with the given modpath
// at the given modroot, and contains go source files.
func PkgIsInLocalModule(pkgpath, modpath, modroot string) bool {
	const isLocal = true
	_, haveGoFiles, err := dirInModule(pkgpath, modpath, modroot, isLocal)
	return err == nil && haveGoFiles
}

// dirInModule locates the directory that would hold the package named by the given path,
// if it were in the module with module path mpath and root mdir.
// If path is syntactically not within mpath,
// or if mdir is a local file tree (isLocal == true) and the directory
// that would hold path is in a sub-module (covered by a go.mod below mdir),
// dirInModule returns "", false, nil.
//
// Otherwise, dirInModule returns the name of the directory where
// Go source files would be expected, along with a boolean indicating
// whether there are in fact Go source files in that directory.
// A non-nil error indicates that the existence of the directory and/or
// source files could not be determined, for example due to a permission error.
//
// TODO(matloob): Could we use the modindex to check packages in indexed modules?
func dirInModule(path, mpath, mdir string, isLocal bool) (dir string, haveGoFiles bool, err error) {
	// Determine where to expect the package.
	if path == mpath {
		dir = mdir
	} else if mpath == "" { // vendor directory
		dir = filepath.Join(mdir, path)
	} else if len(path) > len(mpath) && path[len(mpath)] == '/' && path[:len(mpath)] == mpath {
		dir = filepath.Join(mdir, path[len(mpath)+1:])
	} else {
		return "", false, nil
	}

	// Check that there aren't other modules in the way.
	// This check is unnecessary inside the module cache
	// and important to skip in the vendor directory,
	// where all the module trees have been overlaid.
	// So we only check local module trees
	// (the main module, and any directory trees pointed at by replace directives).
	if isLocal {
		for d := dir; d != mdir && len(d) > len(mdir); {
			haveGoMod := haveGoModCache.Do(d, func() bool {
				fi, err := fsys.Stat(filepath.Join(d, "go.mod"))
				return err == nil && !fi.IsDir()
			})

			if haveGoMod {
				return "", false, nil
			}
			parent := filepath.Dir(d)
			if parent == d {
				// Break the loop, as otherwise we'd loop
				// forever if d=="." and mdir=="".
				break
			}
			d = parent
		}
	}

	// Now committed to returning dir (not "").

	// Are there Go source files in the directory?
	// We don't care about build tags, not even "go:build ignore".
	// We're just looking for a plausible directory.
	haveGoFiles, err = haveGoFilesCache.Do(dir, func() (bool, error) {
		// modindex.GetPackage will return ErrNotIndexed for any directories which
		// are reached through a symlink, so that they will be handled by
		// fsys.IsGoDir below.
		if ip, err := modindex.GetPackage(mdir, dir); err == nil {
			return ip.IsGoDir()
		} else if !errors.Is(err, modindex.ErrNotIndexed) {
			return false, err
		}
		return fsys.IsGoDir(dir)
	})

	return dir, haveGoFiles, err
}

// fetch downloads the given module (or its replacement)
// and returns its location.
//
// The isLocal return value reports whether the replacement,
// if any, is local to the filesystem.
func fetch(loaderstate *State, ctx context.Context, mod module.Version) (dir string, isLocal bool, err error) {
	if modRoot := loaderstate.MainModules.ModRoot(mod); modRoot != "" {
		return modRoot, true, nil
	}
	if r := Replacement(loaderstate, mod); r.Path != "" {
		if r.Version == "" {
			dir = r.Path
			if !filepath.IsAbs(dir) {
				dir = filepath.Join(replaceRelativeTo(loaderstate), dir)
			}
			// Ensure that the replacement directory actually exists:
			// dirInModule does not report errors for missing modules,
			// so if we don't report the error now, later failures will be
			// very mysterious.
			if _, err := fsys.Stat(dir); err != nil {
				// TODO(bcmills): We should also read dir/go.mod here and check its Go version,
				// and return a gover.TooNewError if appropriate.

				if os.IsNotExist(err) {
					// Semantically the module version itself “exists” — we just don't
					// have its source code. Remove the equivalence to os.ErrNotExist,
					// and make the message more concise while we're at it.
					err = fmt.Errorf("replacement directory %s does not exist", r.Path)
				} else {
					err = fmt.Errorf("replacement directory %s: %w", r.Path, err)
				}
				return dir, true, module.VersionError(mod, err)
			}
			return dir, true, nil
		}
		mod = r
	}

	if mustHaveSums(loaderstate) && !modfetch.HaveSum(mod) {
		return "", false, module.VersionError(mod, &sumMissingError{})
	}

	dir, err = modfetch.Download(ctx, mod)
	return dir, false, err
}

// mustHaveSums reports whether we require that all checksums
// needed to load or build packages are already present in the go.sum file.
func mustHaveSums(loaderstate *State) bool {
	return loaderstate.HasModRoot() && cfg.BuildMod == "readonly" && !loaderstate.inWorkspaceMode()
}

type sumMissingError struct {
	suggestion string
}

func (e *sumMissingError) Error() string {
	return "missing go.sum entry" + e.suggestion
}
