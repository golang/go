// Copyright 2018 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

package modload

import (
	"context"
	"errors"
	"fmt"
	"go/build"
	"internal/goroot"
	"io/fs"
	"os"
	"path/filepath"
	"sort"
	"strings"

	"cmd/go/internal/cfg"
	"cmd/go/internal/fsys"
	"cmd/go/internal/modfetch"
	"cmd/go/internal/par"
	"cmd/go/internal/search"

	"golang.org/x/mod/module"
	"golang.org/x/mod/semver"
)

type ImportMissingError struct {
	Path     string
	Module   module.Version
	QueryErr error

	// isStd indicates whether we would expect to find the package in the standard
	// library. This is normally true for all dotless import paths, but replace
	// directives can cause us to treat the replaced paths as also being in
	// modules.
	isStd bool

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
			return fmt.Sprintf("package %s is not in GOROOT (%s)", e.Path, filepath.Join(cfg.GOROOT, "src", e.Path))
		}
		if e.QueryErr != nil && e.QueryErr != ErrNoModRoot {
			return fmt.Sprintf("cannot find module providing package %s: %v", e.Path, e.QueryErr)
		}
		if cfg.BuildMod == "mod" || (cfg.BuildMod == "readonly" && allowMissingModuleImports) {
			return "cannot find module providing package " + e.Path
		}

		if e.replaced.Path != "" {
			suggestArg := e.replaced.Path
			if !modfetch.IsZeroPseudoVersion(e.replaced.Version) {
				suggestArg = e.replaced.String()
			}
			return fmt.Sprintf("module %s provides package %s and is replaced but not required; to add it:\n\tgo get %s", e.replaced.Path, e.Path, suggestArg)
		}

		message := fmt.Sprintf("no required module provides package %s", e.Path)
		if e.QueryErr != nil {
			return fmt.Sprintf("%s: %v", message, e.QueryErr)
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

// ImportMissingSumError is reported in readonly mode when we need to check
// if a module contains a package, but we don't have a sum for its .zip file.
// We might need sums for multiple modules to verify the package is unique.
//
// TODO(#43653): consolidate multiple errors of this type into a single error
// that suggests a 'go get' command for root packages that transtively import
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
		args := make([]string, len(e.mods))
		for i, mod := range e.mods {
			args[i] = mod.Path
		}
		hint = fmt.Sprintf("; to add:\n\tgo mod download %s", strings.Join(args, " "))
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

// importFromBuildList finds the module and directory in the build list
// containing the package with the given import path. The answer must be unique:
// importFromBuildList returns an error if multiple modules attempt to provide
// the same package.
//
// importFromBuildList can return a module with an empty m.Path, for packages in
// the standard library.
//
// importFromBuildList can return an empty directory string, for fake packages
// like "C" and "unsafe".
//
// If the package cannot be found in buildList,
// importFromBuildList returns an *ImportMissingError.
func importFromBuildList(ctx context.Context, path string, buildList []module.Version) (m module.Version, dir string, err error) {
	if strings.Contains(path, "@") {
		return module.Version{}, "", fmt.Errorf("import path should not have @version")
	}
	if build.IsLocalImport(path) {
		return module.Version{}, "", fmt.Errorf("relative import not supported")
	}
	if path == "C" || path == "unsafe" {
		// There's no directory for import "C" or import "unsafe".
		return module.Version{}, "", nil
	}
	// Before any further lookup, check that the path is valid.
	if err := module.CheckImportPath(path); err != nil {
		return module.Version{}, "", &invalidImportError{importPath: path, err: err}
	}

	// Is the package in the standard library?
	pathIsStd := search.IsStandardImportPath(path)
	if pathIsStd && goroot.IsStandardPackage(cfg.GOROOT, cfg.BuildContext.Compiler, path) {
		if targetInGorootSrc {
			if dir, ok, err := dirInModule(path, targetPrefix, ModRoot(), true); err != nil {
				return module.Version{}, dir, err
			} else if ok {
				return Target, dir, nil
			}
		}
		dir := filepath.Join(cfg.GOROOT, "src", path)
		return module.Version{}, dir, nil
	}

	// -mod=vendor is special.
	// Everything must be in the main module or the main module's vendor directory.
	if cfg.BuildMod == "vendor" {
		mainDir, mainOK, mainErr := dirInModule(path, targetPrefix, ModRoot(), true)
		vendorDir, vendorOK, _ := dirInModule(path, "", filepath.Join(ModRoot(), "vendor"), false)
		if mainOK && vendorOK {
			return module.Version{}, "", &AmbiguousImportError{importPath: path, Dirs: []string{mainDir, vendorDir}}
		}
		// Prefer to return main directory if there is one,
		// Note that we're not checking that the package exists.
		// We'll leave that for load.
		if !vendorOK && mainDir != "" {
			return Target, mainDir, nil
		}
		if mainErr != nil {
			return module.Version{}, "", mainErr
		}
		readVendorList()
		return vendorPkgModule[path], vendorDir, nil
	}

	// Check each module on the build list.
	var dirs []string
	var mods []module.Version
	var sumErrMods []module.Version
	for _, m := range buildList {
		if !maybeInModule(path, m.Path) {
			// Avoid possibly downloading irrelevant modules.
			continue
		}
		needSum := true
		root, isLocal, err := fetch(ctx, m, needSum)
		if err != nil {
			if sumErr := (*sumMissingError)(nil); errors.As(err, &sumErr) {
				// We are missing a sum needed to fetch a module in the build list.
				// We can't verify that the package is unique, and we may not find
				// the package at all. Keep checking other modules to decide which
				// error to report. Multiple sums may be missing if we need to look in
				// multiple nested modules to resolve the import.
				sumErrMods = append(sumErrMods, m)
				continue
			}
			// Report fetch error.
			// Note that we don't know for sure this module is necessary,
			// but it certainly _could_ provide the package, and even if we
			// continue the loop and find the package in some other module,
			// we need to look at this module to make sure the import is
			// not ambiguous.
			return module.Version{}, "", err
		}
		if dir, ok, err := dirInModule(path, m.Path, root, isLocal); err != nil {
			return module.Version{}, "", err
		} else if ok {
			mods = append(mods, m)
			dirs = append(dirs, dir)
		}
	}
	if len(mods) > 1 {
		return module.Version{}, "", &AmbiguousImportError{importPath: path, Dirs: dirs, Modules: mods}
	}
	if len(sumErrMods) > 0 {
		return module.Version{}, "", &ImportMissingSumError{
			importPath: path,
			mods:       sumErrMods,
			found:      len(mods) > 0,
		}
	}
	if len(mods) == 1 {
		return mods[0], dirs[0], nil
	}

	var queryErr error
	if !HasModRoot() {
		queryErr = ErrNoModRoot
	}
	return module.Version{}, "", &ImportMissingError{Path: path, QueryErr: queryErr, isStd: pathIsStd}
}

// queryImport attempts to locate a module that can be added to the current
// build list to provide the package with the given import path.
//
// Unlike QueryPattern, queryImport prefers to add a replaced version of a
// module *before* checking the proxies for a version to add.
func queryImport(ctx context.Context, path string) (module.Version, error) {
	// To avoid spurious remote fetches, try the latest replacement for each
	// module (golang.org/issue/26241).
	if index != nil {
		var mods []module.Version
		for mp, mv := range index.highestReplaced {
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
					mv = modfetch.ZeroPseudoVersion(pathMajor[1:])
				} else {
					mv = modfetch.ZeroPseudoVersion("v0")
				}
			}
			mods = append(mods, module.Version{Path: mp, Version: mv})
		}

		// Every module path in mods is a prefix of the import path.
		// As in QueryPattern, prefer the longest prefix that satisfies the import.
		sort.Slice(mods, func(i, j int) bool {
			return len(mods[i].Path) > len(mods[j].Path)
		})
		for _, m := range mods {
			needSum := true
			root, isLocal, err := fetch(ctx, m, needSum)
			if err != nil {
				if sumErr := (*sumMissingError)(nil); errors.As(err, &sumErr) {
					return module.Version{}, &ImportMissingSumError{importPath: path}
				}
				return module.Version{}, err
			}
			if _, ok, err := dirInModule(path, m.Path, root, isLocal); err != nil {
				return m, err
			} else if ok {
				if cfg.BuildMod == "readonly" {
					return module.Version{}, &ImportMissingError{Path: path, replaced: m}
				}
				return m, nil
			}
		}
		if len(mods) > 0 && module.CheckPath(path) != nil {
			// The package path is not valid to fetch remotely,
			// so it can only exist in a replaced module,
			// and we know from the above loop that it is not.
			return module.Version{}, &PackageNotInModuleError{
				Mod:         mods[0],
				Query:       "latest",
				Pattern:     path,
				Replacement: Replacement(mods[0]),
			}
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
		return module.Version{}, &ImportMissingError{Path: path, isStd: true}
	}

	if cfg.BuildMod == "readonly" && !allowMissingModuleImports {
		// In readonly mode, we can't write go.mod, so we shouldn't try to look up
		// the module. If readonly mode was enabled explicitly, include that in
		// the error message.
		var queryErr error
		if cfg.BuildModExplicit {
			queryErr = fmt.Errorf("import lookup disabled by -mod=%s", cfg.BuildMod)
		} else if cfg.BuildModReason != "" {
			queryErr = fmt.Errorf("import lookup disabled by -mod=%s\n\t(%s)", cfg.BuildMod, cfg.BuildModReason)
		}
		return module.Version{}, &ImportMissingError{Path: path, QueryErr: queryErr}
	}

	// Look up module containing the package, for addition to the build list.
	// Goal is to determine the module, download it to dir,
	// and return m, dir, ImpportMissingError.
	fmt.Fprintf(os.Stderr, "go: finding module for package %s\n", path)

	candidates, err := QueryPackages(ctx, path, "latest", Selected, CheckAllowed)
	if err != nil {
		if errors.Is(err, fs.ErrNotExist) {
			// Return "cannot find module providing package […]" instead of whatever
			// low-level error QueryPattern produced.
			return module.Version{}, &ImportMissingError{Path: path, QueryErr: err}
		} else {
			return module.Version{}, err
		}
	}

	candidate0MissingVersion := ""
	for i, c := range candidates {
		cm := c.Mod
		canAdd := true
		for _, bm := range buildList {
			if bm.Path == cm.Path && semver.Compare(bm.Version, cm.Version) > 0 {
				// QueryPattern proposed that we add module cm to provide the package,
				// but we already depend on a newer version of that module (and we don't
				// have the package).
				//
				// This typically happens when a package is present at the "@latest"
				// version (e.g., v1.0.0) of a module, but we have a newer version
				// of the same module in the build list (e.g., v1.0.1-beta), and
				// the package is not present there.
				canAdd = false
				if i == 0 {
					candidate0MissingVersion = bm.Version
				}
				break
			}
		}
		if canAdd {
			return cm, nil
		}
	}
	return module.Version{}, &ImportMissingError{
		Path:              path,
		Module:            candidates[0].Mod,
		newMissingVersion: candidate0MissingVersion,
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
	haveGoModCache   par.Cache // dir → bool
	haveGoFilesCache par.Cache // dir → goFilesEntry
)

type goFilesEntry struct {
	haveGoFiles bool
	err         error
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
			haveGoMod := haveGoModCache.Do(d, func() interface{} {
				fi, err := fsys.Stat(filepath.Join(d, "go.mod"))
				return err == nil && !fi.IsDir()
			}).(bool)

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
	// We don't care about build tags, not even "+build ignore".
	// We're just looking for a plausible directory.
	res := haveGoFilesCache.Do(dir, func() interface{} {
		ok, err := fsys.IsDirWithGoFiles(dir)
		return goFilesEntry{haveGoFiles: ok, err: err}
	}).(goFilesEntry)

	return dir, res.haveGoFiles, res.err
}

// fetch downloads the given module (or its replacement)
// and returns its location.
//
// needSum indicates whether the module may be downloaded in readonly mode
// without a go.sum entry. It should only be false for modules fetched
// speculatively (for example, for incompatible version filtering). The sum
// will still be verified normally.
//
// The isLocal return value reports whether the replacement,
// if any, is local to the filesystem.
func fetch(ctx context.Context, mod module.Version, needSum bool) (dir string, isLocal bool, err error) {
	if mod == Target {
		return ModRoot(), true, nil
	}
	if r := Replacement(mod); r.Path != "" {
		if r.Version == "" {
			dir = r.Path
			if !filepath.IsAbs(dir) {
				dir = filepath.Join(ModRoot(), dir)
			}
			// Ensure that the replacement directory actually exists:
			// dirInModule does not report errors for missing modules,
			// so if we don't report the error now, later failures will be
			// very mysterious.
			if _, err := fsys.Stat(dir); err != nil {
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

	if HasModRoot() && cfg.BuildMod == "readonly" && needSum && !modfetch.HaveSum(mod) {
		return "", false, module.VersionError(mod, &sumMissingError{})
	}

	dir, err = modfetch.Download(ctx, mod)
	return dir, false, err
}

type sumMissingError struct {
	suggestion string
}

func (e *sumMissingError) Error() string {
	return "missing go.sum entry" + e.suggestion
}
