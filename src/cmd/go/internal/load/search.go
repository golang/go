// Copyright 2017 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

package load

import (
	"path/filepath"
	"strings"

	"cmd/go/internal/modload"
	"cmd/go/internal/search"
	"cmd/internal/pkgpattern"
)

// MatchPackage(pattern, cwd)(p) reports whether package p matches pattern in the working directory cwd.
func MatchPackage(loaderstate *modload.State, pattern, cwd string) func(*Package) bool {
	switch {
	case search.IsRelativePath(pattern):
		// Split pattern into leading pattern-free directory path
		// (including all . and .. elements) and the final pattern.
		var dir string
		i := strings.Index(pattern, "...")
		if i < 0 {
			dir, pattern = pattern, ""
		} else {
			j := strings.LastIndex(pattern[:i], "/")
			dir, pattern = pattern[:j], pattern[j+1:]
		}
		dir = filepath.Join(cwd, dir)
		if pattern == "" {
			return func(p *Package) bool { return p.Dir == dir }
		}
		matchPath := pkgpattern.MatchPattern(pattern)
		return func(p *Package) bool {
			// Compute relative path to dir and see if it matches the pattern.
			rel, err := filepath.Rel(dir, p.Dir)
			if err != nil {
				// Cannot make relative - e.g. different drive letters on Windows.
				return false
			}
			rel = filepath.ToSlash(rel)
			if rel == ".." || strings.HasPrefix(rel, "../") {
				return false
			}
			return matchPath(rel)
		}
	case pattern == "all":
		// This is slightly inaccurate: it matches every package, which isn't the same
		// as matching the "all" package pattern.
		// TODO(matloob): Should we make this more accurate? Does anyone depend on this behavior?
		return func(p *Package) bool { return true }
	case pattern == "std":
		return func(p *Package) bool { return p.Standard }
	case pattern == "cmd":
		return func(p *Package) bool { return p.Standard && strings.HasPrefix(p.ImportPath, "cmd/") }
	case pattern == "tool" && modload.Enabled(loaderstate):
		return func(p *Package) bool {
			return loaderstate.MainModules.Tools()[p.ImportPath]
		}
	case pattern == "work" && modload.Enabled(loaderstate):
		return func(p *Package) bool {
			return p.Module != nil && loaderstate.MainModules.Contains(p.Module.Path)
		}

	default:
		matchPath := pkgpattern.MatchPattern(pattern)
		return func(p *Package) bool { return matchPath(p.ImportPath) }
	}
}
