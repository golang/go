// Copyright 2026 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

package modload

import (
	"os"
	"path/filepath"
	"strings"

	"cmd/go/internal/cfg"
	"cmd/go/internal/fsys"
	"cmd/go/internal/search"
)

// logosDirHasGoFiles reports whether dir holds a plausible Go-family
// source file: a regular file whose name satisfies cfg.IsGoFile.
// It mirrors fsys.IsGoDir (which stays untouched) for logos mode,
// including the symlink-target check, but admits .lgo and alias
// extensions through the shared cfg predicate.
func logosDirHasGoFiles(dir string) (bool, error) {
	fis, err := fsys.ReadDir(dir)
	if err != nil {
		if os.IsNotExist(err) {
			return false, nil
		}
		// Mirror fsys.IsGoDir: a path that exists but is not a
		// directory holds no Go files.
		if fi, statErr := fsys.Stat(dir); statErr == nil && !fi.IsDir() {
			return false, nil
		}
		return false, err
	}

	var firstErr error
	for _, d := range fis {
		if d.IsDir() || !cfg.IsGoFile(d.Name()) {
			continue
		}
		if d.Type().IsRegular() {
			return true, nil
		}

		// d is a non-directory, non-regular Go-family file.
		// Stat to see if it is a symlink, which we allow.
		if actual := fsys.Actual(filepath.Join(dir, d.Name())); actual != "" {
			fi, err := os.Stat(actual)
			if err == nil && fi.Mode().IsRegular() {
				return true, nil
			}
			if err != nil && firstErr == nil {
				firstErr = err
			}
		}
	}

	// No Go-family files found in directory.
	return false, firstErr
}

// matchLogosWorkspaceRoot handles logos-only wildcard patterns issued
// from the go.work directory itself, which contains workspace modules
// but is not inside any of them. Upstream matchLocalDirs rejects such
// patterns before any scan runs, and the per-directory walk skips
// nested modules, so neither path can enumerate the workspace members.
// In logos mode this helper unions the per-module matches instead.
// Plain go never reaches it.
//
// Each member is walked with an absolute sub-pattern through the
// existing search.Match scanning; nothing is registered anywhere.
// It returns true when member walks found directories (handled),
// false when the caller must fall through to the upstream rejection,
// so an empty union is never a silent zero-package match.
func matchLogosWorkspaceRoot(ld *Loader, m *search.Match, modRoots []string, absDir string) bool {
	if !cfg.Logos {
		return false
	}
	if !strings.Contains(m.Pattern(), "...") {
		return false
	}
	if !ld.inWorkspaceMode() {
		return false
	}
	if filepath.Clean(absDir) != filepath.Clean(filepath.Dir(WorkFilePath(ld))) {
		return false
	}

	_, suffix, _ := strings.Cut(filepath.ToSlash(m.Pattern()), "...")

	seen := make(map[string]bool)
	m.Dirs = []string{}
	for _, mr := range modRoots {
		if mr == "" {
			continue
		}
		mr = filepath.Clean(mr)
		subMatch := search.NewMatch(filepath.ToSlash(mr) + "/..." + suffix)
		subMatch.MatchDirs([]string{mr})
		m.Errs = append(m.Errs, subMatch.Errs...)
		for _, d := range subMatch.Dirs {
			if !seen[d] {
				seen[d] = true
				m.Dirs = append(m.Dirs, d)
			}
		}
	}
	return len(m.Dirs) > 0
}
