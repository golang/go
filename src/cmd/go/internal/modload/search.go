// Copyright 2018 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

package modload

import (
	"fmt"
	"os"
	"path/filepath"
	"strings"

	"cmd/go/internal/cfg"
	"cmd/go/internal/imports"
	"cmd/go/internal/search"

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
func matchPackages(m *search.Match, tags map[string]bool, filter stdFilter, modules []module.Version) {
	m.Pkgs = []string{}

	isMatch := func(string) bool { return true }
	treeCanMatch := func(string) bool { return true }
	if !m.IsMeta() {
		isMatch = search.MatchPattern(m.Pattern())
		treeCanMatch = search.TreeCanMatchPattern(m.Pattern())
	}

	have := map[string]bool{
		"builtin": true, // ignore pseudo-package that exists only for documentation
	}
	if !cfg.BuildContext.CgoEnabled {
		have["runtime/cgo"] = true // ignore during walk
	}

	type pruning int8
	const (
		pruneVendor = pruning(1 << iota)
		pruneGoMod
	)

	walkPkgs := func(root, importPathRoot string, prune pruning) {
		root = filepath.Clean(root)
		err := filepath.Walk(root, func(path string, fi os.FileInfo, err error) error {
			if err != nil {
				m.AddError(err)
				return nil
			}

			want := true
			elem := ""

			// Don't use GOROOT/src but do walk down into it.
			if path == root {
				if importPathRoot == "" {
					return nil
				}
			} else {
				// Avoid .foo, _foo, and testdata subdirectory trees.
				_, elem = filepath.Split(path)
				if strings.HasPrefix(elem, ".") || strings.HasPrefix(elem, "_") || elem == "testdata" {
					want = false
				}
			}

			name := importPathRoot + filepath.ToSlash(path[len(root):])
			if importPathRoot == "" {
				name = name[1:] // cut leading slash
			}
			if !treeCanMatch(name) {
				want = false
			}

			if !fi.IsDir() {
				if fi.Mode()&os.ModeSymlink != 0 && want {
					if target, err := os.Stat(path); err == nil && target.IsDir() {
						fmt.Fprintf(os.Stderr, "warning: ignoring symlink %s\n", path)
					}
				}
				return nil
			}

			if !want {
				return filepath.SkipDir
			}
			// Stop at module boundaries.
			if (prune&pruneGoMod != 0) && path != root {
				if fi, err := os.Stat(filepath.Join(path, "go.mod")); err == nil && !fi.IsDir() {
					return filepath.SkipDir
				}
			}

			if !have[name] {
				have[name] = true
				if isMatch(name) {
					if _, _, err := scanDir(path, tags); err != imports.ErrNoGo {
						m.Pkgs = append(m.Pkgs, name)
					}
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

	if filter == includeStd {
		walkPkgs(cfg.GOROOTsrc, "", pruneGoMod)
		if treeCanMatch("cmd") {
			walkPkgs(filepath.Join(cfg.GOROOTsrc, "cmd"), "cmd", pruneGoMod)
		}
	}

	if cfg.BuildMod == "vendor" {
		if HasModRoot() {
			walkPkgs(ModRoot(), targetPrefix, pruneGoMod|pruneVendor)
			walkPkgs(filepath.Join(ModRoot(), "vendor"), "", pruneVendor)
		}
		return
	}

	for _, mod := range modules {
		if !treeCanMatch(mod.Path) {
			continue
		}

		var (
			root, modPrefix string
			isLocal         bool
		)
		if mod == Target {
			if !HasModRoot() {
				continue // If there is no main module, we can't search in it.
			}
			root = ModRoot()
			modPrefix = targetPrefix
			isLocal = true
		} else {
			var err error
			root, isLocal, err = fetch(mod)
			if err != nil {
				m.AddError(err)
				continue
			}
			modPrefix = mod.Path
		}

		prune := pruneVendor
		if isLocal {
			prune |= pruneGoMod
		}
		walkPkgs(root, modPrefix, prune)
	}

	return
}
