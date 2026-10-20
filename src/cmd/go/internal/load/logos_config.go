// Copyright 2026 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

package load

import (
	"errors"
	"fmt"
	"go/parser"
	"go/token"
	"io/fs"
	"os"
	"path/filepath"
	"slices"
	"strconv"
	"strings"
	"sync"

	"cmd/go/internal/base"
	"cmd/go/internal/cfg"
	"cmd/go/internal/logos/capabilities"
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
		isNoGo := false
		if p.Error != nil {
			_, isNoGo = errors.AsType[*NoGoError](p.Error.Err)
		}
		if claimConfigError(root) && (p.Error == nil || isNoGo) {
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

// MainModuleLinkPolicy resolves the linker capabilities inputs for one
// build: the logos.toml of the main module containing the current
// directory (the same config the load-phase audit enforces, never the
// linked dependency's own file) and the sorted paths of every modload
// main module. A config load failure returns the error; the load phase
// already reports it before linking.
func MainModuleLinkPolicy(ld *modload.Loader) (*config.File, []string, error) {
	cf, err := logosConfig(ld)
	if err != nil {
		return nil, nil, err
	}
	var mains []string
	if ld != nil && ld.MainModules != nil {
		for _, m := range ld.MainModules.Versions() {
			mains = append(mains, m.Path)
		}
	}
	slices.Sort(mains)
	return cf, mains, nil
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

// checkLogosCgo enforces the logos.toml [build] cgo policy for one loaded
// package in logos mode (M0-S5). It runs in (*Package).load after imports
// resolve, beside checkLogosConfig (p.Module must be set, so it cannot run
// at the earlier UsesCgo site). Plain go never reaches past the logos guard.
// The policy follows the package's module: a main-module package answers to
// its own logos.toml, while a dependency answers to the governing main
// module's, so a restricted allowlist covers every package that imports "C"
// in the build, Go-syntax packages included; the standard library stays
// exempt. Cgo use is the compiled file set (UsesCgo): a bare cgo build tag
// without import "C" never counts, while a cgo-only package hidden by
// disabled cgo is still caught through its ignored files importing "C".
// A denied package fails with [restriction::cgo_usage] instead of the
// upstream "build constraints" text: an enabled denial lands directly, and
// a disabled denial replaces NoGoError. Any other pre-existing error keeps
// priority and is never overwritten. Config load failures are left to
// checkLogosConfig.
func checkLogosCgo(ld *modload.Loader, p *Package) {
	if !cfg.Logos || p.Internal.TestmainGo != nil || p.Standard {
		return
	}
	if p.Module == nil {
		return
	}
	root := logosConfigRootForPackage(ld, p)
	if root == "" || !logosModuleHasLogos(root) {
		return
	}
	if !p.UsesCgo() && !logosCgoHidden(p) {
		return
	}
	cf, err := logosConfigAt(root)
	if err != nil || cf == nil {
		return
	}
	allow, msg := cfg.CgoPolicyAllow(cf, cfg.StartupCgoEnabled(), p.ImportPath)
	if allow {
		capabilities.NoteCgoAllowed(p.ImportPath)
		return
	}
	logosDenyCgo(p, msg)
}

// logosModuleKinds memoizes one loGo/s scan per module root, so the cgo gate
// walks each module once per process. Entries are write-once under the mutex
// and never mutated after store.
var logosModuleKinds = struct {
	sync.Mutex
	byRoot map[string]bool
}{byRoot: make(map[string]bool)}

// logosModuleHasLogos reports whether root holds a loGo/s source file, so
// the cgo gate can tell a loGo/s module (policy applies to every package in
// it, whatever syntax each package uses) from a Go module (upstream
// handling, cgo key ignored). Results are memoized by root.
func logosModuleHasLogos(root string) bool {
	logosModuleKinds.Lock()
	defer logosModuleKinds.Unlock()
	if isLogos, ok := logosModuleKinds.byRoot[root]; ok {
		return isLogos
	}
	isLogos := logosTreeHasLogos(root)
	logosModuleKinds.byRoot[root] = isLogos
	return isLogos
}

// logosTreeHasLogos walks root for a loGo/s source file, skipping nested
// modules and directories that never hold built packages of this module.
func logosTreeHasLogos(root string) bool {
	found := false
	filepath.WalkDir(root, func(path string, d fs.DirEntry, err error) error {
		if found {
			return fs.SkipAll
		}
		if err != nil {
			return nil
		}
		if d.IsDir() {
			if path != root && logosSkipsDir(path, d.Name()) {
				return filepath.SkipDir
			}
			return nil
		}
		if cfg.IsLogosFile(d.Name()) {
			found = true
		}
		return nil
	})
	return found
}

// logosSkipsDir reports whether dir never holds built packages of its parent
// module: a nested module, vendor, testdata, or a dot or underscore prefix.
func logosSkipsDir(path, name string) bool {
	if name == "vendor" || name == "testdata" {
		return true
	}
	if strings.HasPrefix(name, ".") || strings.HasPrefix(name, "_") {
		return true
	}
	if info, err := os.Stat(filepath.Join(path, "go.mod")); err == nil && !info.IsDir() {
		return true
	}
	return false
}

// logosCgoHidden reports whether p is a cgo package hidden by disabled cgo:
// nothing compiled, but an ignored file imports "C". It runs only when the
// package failed with NoGoError, so builds that selected a non-cgo variant
// never scan. A bare cgo build tag without import "C" never counts.
func logosCgoHidden(p *Package) bool {
	if p.UsesCgo() || p.Internal.Build == nil {
		return false
	}
	if p.Error == nil {
		return false
	}
	if _, ok := errors.AsType[*NoGoError](p.Error.Err); !ok {
		return false
	}
	if !slices.Contains(p.Internal.Build.AllTags, "cgo") {
		return false
	}
	for _, name := range p.IgnoredGoFiles {
		if logosFileImportsC(p.Dir, name) {
			return true
		}
	}
	return false
}

// logosFileImportsC reports whether the file imports "C", parsed
// imports-only. Files that fail to parse never count: a cgo file must use
// Go syntax, so an unparseable file cannot be one.
func logosFileImportsC(dir, name string) bool {
	parsed, err := parser.ParseFile(token.NewFileSet(), filepath.Join(dir, name), nil, parser.ImportsOnly)
	if err != nil || parsed == nil {
		return false
	}
	for _, imp := range parsed.Imports {
		if path, err := strconv.Unquote(imp.Path.Value); err == nil && path == "C" {
			return true
		}
	}
	return false
}

// logosDenyCgo fails p with the policy denial, replacing NoGoError so the
// scripts see [restriction::cgo_usage] instead of the upstream text. Any
// other pre-existing error keeps priority and is never overwritten.
func logosDenyCgo(p *Package, msg string) {
	if p.Error != nil {
		if _, ok := errors.AsType[*NoGoError](p.Error.Err); !ok {
			return
		}
	}
	p.Error = &PackageError{Err: errors.New(msg)}
	p.Incomplete = true
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
