// Copyright 2018 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

package modload

import (
	"context"
	"errors"
	"fmt"
	"io/fs"
	"os"
	"path"
	"path/filepath"
	"runtime"
	"sort"
	"strings"
	"sync"

	"cmd/go/internal/cfg"
	"cmd/go/internal/fsys"
	"cmd/go/internal/gover"
	"cmd/go/internal/imports"
	"cmd/go/internal/modfetch"
	"cmd/go/internal/modindex"
	"cmd/go/internal/search"
	"cmd/go/internal/str"
	"cmd/go/internal/trace"
	"cmd/internal/par"
	"cmd/internal/pkgpattern"

	"golang.org/x/mod/module"
)

type stdFilter int8

const (
	omitStd = stdFilter(iota)
	includeStd
)

// matchPackages is like m.MatchPackages, but uses a local variable (rather than
// a global) for tags, can include or exclude packages in the standard library,
// and is restricted to the given list of modules.
func matchPackages(loaderstate *State, ctx context.Context, m *search.Match, tags map[string]bool, filter stdFilter, modules []module.Version) {
	ctx, span := trace.StartSpan(ctx, "modload.matchPackages")
	defer span.Done()

	m.Pkgs = []string{}

	isMatch := func(string) bool { return true }
	treeCanMatch := func(string) bool { return true }
	if !m.IsMeta() {
		isMatch = pkgpattern.MatchPattern(m.Pattern())
		treeCanMatch = pkgpattern.TreeCanMatchPattern(m.Pattern())
	}

	var mu sync.Mutex
	have := map[string]bool{
		"builtin": true, // ignore pseudo-package that exists only for documentation
	}
	addPkg := func(p string) {
		mu.Lock()
		m.Pkgs = append(m.Pkgs, p)
		mu.Unlock()
	}
	if !cfg.BuildContext.CgoEnabled {
		have["runtime/cgo"] = true // ignore during walk
	}

	type pruning int8
	const (
		pruneVendor = pruning(1 << iota)
		pruneGoMod
	)

	q := par.NewQueue(runtime.GOMAXPROCS(0))
	ignorePatternsMap := parseIgnorePatterns(loaderstate, ctx, treeCanMatch, modules)
	walkPkgs := func(root, importPathRoot string, prune pruning) {
		_, span := trace.StartSpan(ctx, "walkPkgs "+root)
		defer span.Done()

		// If the root itself is a symlink to a directory,
		// we want to follow it (see https://go.dev/issue/50807).
		// Add a trailing separator to force that to happen.
		cleanRoot := filepath.Clean(root)
		root = str.WithFilePathSeparator(cleanRoot)
		err := fsys.WalkDir(root, func(pkgDir string, d fs.DirEntry, err error) error {
			if err != nil {
				m.AddError(err)
				return nil
			}

			want := true
			elem := ""
			relPkgDir := filepath.ToSlash(pkgDir[len(root):])

			// Don't use GOROOT/src but do walk down into it.
			if pkgDir == root {
				if importPathRoot == "" {
					return nil
				}
			} else {
				// Avoid .foo, _foo, and testdata subdirectory trees.
				_, elem = filepath.Split(pkgDir)
				if strings.HasPrefix(elem, ".") || strings.HasPrefix(elem, "_") || elem == "testdata" {
					want = false
				} else if ignorePatternsMap[cleanRoot] != nil && ignorePatternsMap[cleanRoot].ShouldIgnore(relPkgDir) {
					if cfg.BuildX {
						fmt.Fprintf(os.Stderr, "# ignoring directory %s\n", pkgDir)
					}
					want = false
				}
			}

			name := path.Join(importPathRoot, relPkgDir)
			if !treeCanMatch(name) {
				want = false
			}

			if !d.IsDir() {
				if d.Type()&fs.ModeSymlink != 0 && want && strings.Contains(m.Pattern(), "...") {
					if target, err := fsys.Stat(pkgDir); err == nil && target.IsDir() {
						fmt.Fprintf(os.Stderr, "warning: ignoring symlink %s\n", pkgDir)
					}
				}
				return nil
			}

			if !want {
				return filepath.SkipDir
			}
			// Stop at module boundaries.
			if (prune&pruneGoMod != 0) && pkgDir != root {
				if info, err := os.Stat(filepath.Join(pkgDir, "go.mod")); err == nil && !info.IsDir() {
					return filepath.SkipDir
				}
			}

			if !have[name] {
				have[name] = true
				if isMatch(name) {
					q.Add(func() {
						if _, _, err := scanDir(root, pkgDir, tags); err != imports.ErrNoGo {
							addPkg(name)
						}
					})
				}
			}

			if elem == "vendor" && (prune&pruneVendor != 0) {
				return filepath.SkipDir
			}
			return nil
		})
		if err != nil {
			m.AddError(err)
		}
	}

	// Wait for all in-flight operations to complete before returning.
	defer func() {
		<-q.Idle()
		sort.Strings(m.Pkgs) // sort everything we added for determinism
	}()

	if filter == includeStd {
		walkPkgs(cfg.GOROOTsrc, "", pruneGoMod)
		if treeCanMatch("cmd") {
			walkPkgs(filepath.Join(cfg.GOROOTsrc, "cmd"), "cmd", pruneGoMod)
		}
	}

	if cfg.BuildMod == "vendor" {
		for _, mod := range loaderstate.MainModules.Versions() {
			if modRoot := loaderstate.MainModules.ModRoot(mod); modRoot != "" {
				walkPkgs(modRoot, loaderstate.MainModules.PathPrefix(mod), pruneGoMod|pruneVendor)
			}
		}
		if loaderstate.HasModRoot() {
			walkPkgs(VendorDir(loaderstate), "", pruneVendor)
		}
		return
	}

	for _, mod := range modules {
		if gover.IsToolchain(mod.Path) || !treeCanMatch(mod.Path) {
			continue
		}

		var (
			root, modPrefix string
			isLocal         bool
		)
		if loaderstate.MainModules.Contains(mod.Path) {
			if loaderstate.MainModules.ModRoot(mod) == "" {
				continue // If there is no main module, we can't search in it.
			}
			root = loaderstate.MainModules.ModRoot(mod)
			modPrefix = loaderstate.MainModules.PathPrefix(mod)
			isLocal = true
		} else {
			var err error
			root, isLocal, err = fetch(loaderstate, ctx, mod)
			if err != nil {
				m.AddError(err)
				continue
			}
			modPrefix = mod.Path
		}
		if mi, err := modindex.GetModule(root); err == nil {
			walkFromIndex(mi, modPrefix, isMatch, treeCanMatch, tags, have, addPkg, ignorePatternsMap[root], root)
			continue
		} else if !errors.Is(err, modindex.ErrNotIndexed) {
			m.AddError(err)
		}

		prune := pruneVendor
		if isLocal {
			prune |= pruneGoMod
		}
		walkPkgs(root, modPrefix, prune)
	}
}

// walkFromIndex matches packages in a module using the module index. modroot
// is the module's root directory on disk, index is the modindex.Module for the
// module, and importPathRoot is the module's path prefix.
func walkFromIndex(index *modindex.Module, importPathRoot string, isMatch, treeCanMatch func(string) bool, tags, have map[string]bool, addPkg func(string), ignorePatterns *search.IgnorePatterns, modRoot string) {
	index.Walk(func(reldir string) {
		// Avoid .foo, _foo, and testdata subdirectory trees.
		p := reldir
		for {
			elem, rest, found := strings.Cut(p, string(filepath.Separator))
			if strings.HasPrefix(elem, ".") || strings.HasPrefix(elem, "_") || elem == "testdata" {
				return
			}
			if found && elem == "vendor" {
				// Ignore this path if it contains the element "vendor" anywhere
				// except for the last element (packages named vendor are allowed
				// for historical reasons). Note that found is true when this
				// isn't the last path element.
				return
			}
			if !found {
				// Didn't find the separator, so we're considering the last element.
				break
			}
			p = rest
		}

		if ignorePatterns != nil && ignorePatterns.ShouldIgnore(reldir) {
			if cfg.BuildX {
				absPath := filepath.Join(modRoot, reldir)
				fmt.Fprintf(os.Stderr, "# ignoring directory %s\n", absPath)
			}
			return
		}

		// Don't use GOROOT/src.
		if reldir == "" && importPathRoot == "" {
			return
		}

		name := path.Join(importPathRoot, filepath.ToSlash(reldir))
		if !treeCanMatch(name) {
			return
		}

		if !have[name] {
			have[name] = true
			if isMatch(name) {
				if _, _, err := index.Package(reldir).ScanDir(tags); err != imports.ErrNoGo {
					addPkg(name)
				}
			}
		}
	})
}

// MatchInModule identifies the packages matching the given pattern within the
// given module version, which does not need to be in the build list or module
// requirement graph.
//
// If m is the zero module.Version, MatchInModule matches the pattern
// against the standard library (std and cmd) in GOROOT/src.
func MatchInModule(loaderstate *State, ctx context.Context, pattern string, m module.Version, tags map[string]bool) *search.Match {
	match := search.NewMatch(pattern)
	if m == (module.Version{}) {
		matchPackages(loaderstate, ctx, match, tags, includeStd, nil)
	}

	LoadModFile(loaderstate, ctx) // Sets Target, needed by fetch and matchPackages.

	if !match.IsLiteral() {
		matchPackages(loaderstate, ctx, match, tags, omitStd, []module.Version{m})
		return match
	}

	root, isLocal, err := fetch(loaderstate, ctx, m)
	if err != nil {
		match.Errs = []error{err}
		return match
	}

	dir, haveGoFiles, err := dirInModule(pattern, m.Path, root, isLocal)
	if err != nil {
		match.Errs = []error{err}
		return match
	}
	if haveGoFiles {
		if _, _, err := scanDir(root, dir, tags); err != imports.ErrNoGo {
			// ErrNoGo indicates that the directory is not actually a Go package,
			// perhaps due to the tags in use. Any other non-nil error indicates a
			// problem with one or more of the Go source files, but such an error does
			// not stop the package from existing, so it has no impact on matching.
			match.Pkgs = []string{pattern}
		}
	}
	return match
}

// parseIgnorePatterns collects all ignore patterns associated with the
// provided list of modules.
// It returns a map of module root -> *search.IgnorePatterns.
func parseIgnorePatterns(loaderstate *State, ctx context.Context, treeCanMatch func(string) bool, modules []module.Version) map[string]*search.IgnorePatterns {
	ignorePatternsMap := make(map[string]*search.IgnorePatterns)
	for _, mod := range modules {
		if gover.IsToolchain(mod.Path) || !treeCanMatch(mod.Path) {
			continue
		}
		var modRoot string
		var ignorePatterns []string
		if loaderstate.MainModules.Contains(mod.Path) {
			modRoot = loaderstate.MainModules.ModRoot(mod)
			if modRoot == "" {
				continue
			}
			modIndex := loaderstate.MainModules.Index(mod)
			if modIndex == nil {
				continue
			}
			ignorePatterns = modIndex.ignore
		} else if cfg.BuildMod != "vendor" {
			// Skip getting ignore patterns for vendored modules because they
			// do not have go.mod files.
			var err error
			modRoot, _, err = fetch(loaderstate, ctx, mod)
			if err != nil {
				continue
			}
			summary, err := goModSummary(modfetch.Fetcher_, loaderstate, mod)
			if err != nil {
				continue
			}
			ignorePatterns = summary.ignore
		}
		ignorePatternsMap[modRoot] = search.NewIgnorePatterns(ignorePatterns)
	}
	return ignorePatternsMap
}
