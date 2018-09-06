// Copyright 2018 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

package modload

import (
	"fmt"
	"os"
	"path/filepath"
	"strings"

	"cmd/go/internal/base"
	"cmd/go/internal/cfg"
	"cmd/go/internal/imports"
	"cmd/go/internal/module"
	"cmd/go/internal/search"
)

// matchPackages returns a list of packages in the list of modules
// matching the pattern. Package loading assumes the given set of tags.
func matchPackages(pattern string, tags map[string]bool, useStd bool, modules []module.Version) []string {
	match := func(string) bool { return true }
	treeCanMatch := func(string) bool { return true }
	if !search.IsMetaPackage(pattern) {
		match = search.MatchPattern(pattern)
		treeCanMatch = search.TreeCanMatchPattern(pattern)
	}

	have := map[string]bool{
		"builtin": true, // ignore pseudo-package that exists only for documentation
	}
	if !cfg.BuildContext.CgoEnabled {
		have["runtime/cgo"] = true // ignore during walk
	}
	var pkgs []string

	walkPkgs := func(root, importPathRoot string) {
		root = filepath.Clean(root)
		var cmd string
		if root == cfg.GOROOTsrc {
			cmd = filepath.Join(root, "cmd")
		}
		filepath.Walk(root, func(path string, fi os.FileInfo, err error) error {
			if err != nil {
				return nil
			}

			// Don't use GOROOT/src but do walk down into it.
			if path == root && importPathRoot == "" {
				return nil
			}

			// GOROOT/src/cmd makes use of GOROOT/src/cmd/vendor,
			// which module mode can't deal with. Eventually we'll stop using
			// that vendor directory, and then we can remove this exclusion.
			// golang.org/issue/26924.
			if path == cmd {
				return filepath.SkipDir
			}

			want := true
			// Avoid .foo, _foo, and testdata directory trees.
			_, elem := filepath.Split(path)
			if strings.HasPrefix(elem, ".") || strings.HasPrefix(elem, "_") || elem == "testdata" {
				want = false
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
			if path != root {
				if _, err := os.Stat(filepath.Join(path, "go.mod")); err == nil {
					return filepath.SkipDir
				}
			}

			if !have[name] {
				have[name] = true
				if match(name) {
					if _, _, err := scanDir(path, tags); err != imports.ErrNoGo {
						pkgs = append(pkgs, name)
					}
				}
			}

			if elem == "vendor" {
				return filepath.SkipDir
			}
			return nil
		})
	}

	if useStd {
		walkPkgs(cfg.GOROOTsrc, "")
	}

	for _, mod := range modules {
		if !treeCanMatch(mod.Path) {
			continue
		}
		var root string
		if mod.Version == "" {
			root = ModRoot
		} else {
			var err error
			root, _, err = fetch(mod)
			if err != nil {
				base.Errorf("go: %v", err)
				continue
			}
		}
		walkPkgs(root, mod.Path)
	}

	return pkgs
}
