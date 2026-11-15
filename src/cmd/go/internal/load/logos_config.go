// Copyright 2026 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

package load

import (
	"errors"
	"fmt"
	"go/parser"
	"go/token"
	"internal/logosfile"
	"path/filepath"
	"slices"
	"strconv"

	"cmd/go/internal/base"
	"cmd/go/internal/cfg"
	"cmd/go/internal/logos/capabilities"
	"cmd/go/internal/logos/config"
	"cmd/go/internal/logos/settings"
	"cmd/go/internal/modload"
	"golang.org/x/mod/module"
)

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
	m := settings.ModuleAt(root)
	if m.Err != nil {
		isNoGo := false
		if p.Error != nil {
			_, isNoGo = errors.AsType[*NoGoError](p.Error.Err)
		}
		if m.ClaimError() && (p.Error == nil || isNoGo) {
			p.Error = &PackageError{Err: m.Err}
			p.Incomplete = true
		}
		return
	}
	for _, files := range [][]string{p.GoFiles, p.CgoFiles, p.TestGoFiles, p.XTestGoFiles} {
		for _, name := range files {
			ext := logosfile.Ext(name)
			if ext == "" {
				continue
			}
			if !slices.Contains(m.File.Project.StrictFiles, ext) {
				if p.Error == nil {
					p.Error = &PackageError{Err: fmt.Errorf("[correctness::extension_not_enabled] %s: extension %s is not enabled by logos.toml strict_files", filepath.Base(name), ext)}
					p.Incomplete = true
				}
				return
			}
		}
	}
}

func logosConfig(ld *modload.Loader) (*config.File, error) {
	m := settings.ModuleAt(logosConfigRoot(ld))
	return m.File, m.Err
}

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

func checkLogosCgo(ld *modload.Loader, p *Package) {
	if !cfg.Logos || p.Internal.TestmainGo != nil || p.Standard {
		return
	}
	if p.Module == nil {
		return
	}
	root := logosConfigRootForPackage(ld, p)
	if root == "" || !settings.ModuleAt(root).Logos() {
		return
	}
	if !p.UsesCgo() && !logosCgoHidden(p) {
		return
	}
	m := settings.ModuleAt(root)
	if m.Err != nil || m.File == nil {
		return
	}
	allow, msg := m.CgoAllow(cfg.StartupCgoEnabled(), p.ImportPath)
	if allow {
		capabilities.NoteCgoAllowed(p.ImportPath)
		return
	}
	logosDenyCgo(p, msg)
}

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

func logosDenyCgo(p *Package, msg string) {
	if p.Error != nil {
		if _, ok := errors.AsType[*NoGoError](p.Error.Err); !ok {
			return
		}
	}
	p.Error = &PackageError{Err: errors.New(msg)}
	p.Incomplete = true
}
