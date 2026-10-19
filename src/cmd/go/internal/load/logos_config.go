// Copyright 2026 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

package load

import (
	"fmt"
	"path/filepath"
	"slices"
	"strings"
	"sync"

	"cmd/go/internal/base"
	"cmd/go/internal/cfg"
	"cmd/go/internal/logos/config"
	"cmd/go/internal/modload"
	"golang.org/x/mod/module"
)

// logosConfigCache memoizes one parsed logos.toml per main-module root, so
// each main module's file is read once per process. Entries are write-once
// under the mutex and never mutated after store.
var logosConfigCache = struct {
	sync.Mutex
	byRoot map[string]logosConfigResult
}{byRoot: make(map[string]logosConfigResult)}

// logosConfigReported tracks which module roots already surfaced a parse
// error as a package error, so a broken main module's logos.toml is
// reported once per build, not once per package. Claiming is an idempotent
// once-per-root flag under the mutex.
var logosConfigReported = struct {
	sync.Mutex
	seen map[string]bool
}{seen: make(map[string]bool)}

// logosConfigResult is one cached config load.
type logosConfigResult struct {
	file *config.File
	err  error
}

// checkLogosConfig applies logos.toml to one loaded package in logos mode.
// Plain go never reaches it. Only packages in a main module are checked,
// using that module's own root and its own logos.toml; dependency modules
// keep defaults and are never checked. Parse and validation failures become
// a package error carrying the exact config message on the first package of
// that module only, and a loGo/s source file whose extension is not in that
// module's strict_files fails with [correctness::extension_not_enabled].
// The generated test main and the standard library stay out of the census,
// mirroring checkLogosModules. Project.Name is validated non-empty in Parse;
// it is never passed to the compiler or linker, so builds differing only in
// name stay identical.
func checkLogosConfig(ld *modload.Loader, p *Package) {
	if !cfg.Logos || p.Internal.TestmainGo != nil || p.Standard {
		return
	}
	if p.Module == nil {
		return
	}
	if ld != nil && ld.MainModules != nil {
		if !ld.MainModules.Contains(p.Module.Path) {
			return
		}
	} else if !p.Module.Main {
		return
	}
	root := logosConfigRootForPackage(ld, p)
	cf, err := logosConfigAt(root)
	if err != nil {
		if p.Error == nil && claimConfigError(root) {
			p.Error = &PackageError{Err: err}
			p.Incomplete = true
		}
		return
	}
	for _, files := range [][]string{p.GoFiles, p.CgoFiles, p.TestGoFiles, p.XTestGoFiles} {
		for _, name := range files {
			if !cfg.IsLogosFile(name) {
				continue
			}
			ext := logosConfigExt(name)
			if !slices.Contains(cf.Project.StrictFiles, ext) {
				if p.Error == nil {
					p.Error = &PackageError{Err: fmt.Errorf("[correctness::extension_not_enabled] %s: extension %s is not enabled by logos.toml strict_files", filepath.Base(name), ext)}
					p.Incomplete = true
				}
				return
			}
		}
	}
}

// logosConfig loads the logos.toml of the main module containing the current
// directory, or defaults when there is no such module.
func logosConfig(ld *modload.Loader) (*config.File, error) {
	return logosConfigAt(logosConfigRoot(ld))
}

// logosConfigAt loads the logos.toml at one module root, memoized by root.
func logosConfigAt(root string) (*config.File, error) {
	logosConfigCache.Lock()
	defer logosConfigCache.Unlock()
	if r, ok := logosConfigCache.byRoot[root]; ok {
		return r.file, r.err
	}
	cf, err := config.Load(root)
	logosConfigCache.byRoot[root] = logosConfigResult{file: cf, err: err}
	return cf, err
}

// claimConfigError reports whether root's parse error may be attached to
// this package: true once per root, false once it has been reported.
func claimConfigError(root string) bool {
	logosConfigReported.Lock()
	defer logosConfigReported.Unlock()
	if logosConfigReported.seen[root] {
		return false
	}
	logosConfigReported.seen[root] = true
	return true
}

// logosConfigRootForPackage resolves the module root whose logos.toml
// governs p: the ModRoot of p's own main module. It falls back to the
// module Dir recorded on the package and then to the CWD-based root.
func logosConfigRootForPackage(ld *modload.Loader, p *Package) string {
	if p.Module != nil {
		if ld != nil && ld.MainModules != nil {
			if root := ld.MainModules.ModRoot(module.Version{Path: p.Module.Path, Version: p.Module.Version}); root != "" {
				return root
			}
		}
		if p.Module.Main && p.Module.Dir != "" {
			return p.Module.Dir
		}
	}
	return logosConfigRoot(ld)
}

// logosConfigRoot resolves the module root whose logos.toml governs this
// build: the root of the main module containing the current directory
// (which covers workspace mode with several main modules). With no main
// module, in GOPATH mode, or for command-line-arguments builds the root is
// empty and defaults apply. Before modload.Init the go.mod is located from
// the working directory instead.
func logosConfigRoot(ld *modload.Loader) string {
	if ld != nil && ld.MainModules != nil && ld.MainModules.Len() > 0 {
		return ld.MainModules.ModRoot(ld.MainModules.ModContainingCWD())
	}
	if cfg.ModulesEnabled {
		if f := modload.FindGoMod(base.Cwd()); f != "" {
			return filepath.Dir(f)
		}
	}
	return ""
}

// logosConfigExt returns the loGo/s extension of a source file name,
// stripping trailing variation selectors U+FE0E and U+FE0F the same way
// cfg.IsLogosFile does.
func logosConfigExt(name string) string {
	trimmed := strings.TrimRight(name, "\uFE0E\uFE0F")
	if i := strings.LastIndexByte(trimmed, '.'); i >= 0 {
		return trimmed[i:]
	}
	return trimmed
}
