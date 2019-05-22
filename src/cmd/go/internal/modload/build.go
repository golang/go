// Copyright 2018 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

package modload

import (
	"bytes"
	"cmd/go/internal/base"
	"cmd/go/internal/cfg"
	"cmd/go/internal/modfetch"
	"cmd/go/internal/modinfo"
	"cmd/go/internal/module"
	"cmd/go/internal/search"
	"cmd/go/internal/semver"
	"encoding/hex"
	"fmt"
	"internal/goroot"
	"os"
	"path/filepath"
	"runtime/debug"
	"strings"
)

var (
	infoStart, _ = hex.DecodeString("3077af0c9274080241e1c107e6d618e6")
	infoEnd, _   = hex.DecodeString("f932433186182072008242104116d8f2")
)

func isStandardImportPath(path string) bool {
	return findStandardImportPath(path) != ""
}

func findStandardImportPath(path string) string {
	if path == "" {
		panic("findStandardImportPath called with empty path")
	}
	if search.IsStandardImportPath(path) {
		if goroot.IsStandardPackage(cfg.GOROOT, cfg.BuildContext.Compiler, path) {
			return filepath.Join(cfg.GOROOT, "src", path)
		}
	}
	return ""
}

func PackageModuleInfo(pkgpath string) *modinfo.ModulePublic {
	if isStandardImportPath(pkgpath) || !Enabled() {
		return nil
	}
	return moduleInfo(findModule(pkgpath, pkgpath), true)
}

func ModuleInfo(path string) *modinfo.ModulePublic {
	if !Enabled() {
		return nil
	}

	if i := strings.Index(path, "@"); i >= 0 {
		return moduleInfo(module.Version{Path: path[:i], Version: path[i+1:]}, false)
	}

	for _, m := range BuildList() {
		if m.Path == path {
			return moduleInfo(m, true)
		}
	}

	return &modinfo.ModulePublic{
		Path: path,
		Error: &modinfo.ModuleError{
			Err: "module not in current build",
		},
	}
}

// addUpdate fills in m.Update if an updated version is available.
func addUpdate(m *modinfo.ModulePublic) {
	if m.Version == "" {
		return
	}

	if info, err := Query(m.Path, "latest", Allowed); err == nil && semver.Compare(info.Version, m.Version) > 0 {
		m.Update = &modinfo.ModulePublic{
			Path:    m.Path,
			Version: info.Version,
			Time:    &info.Time,
		}
	}
}

// addVersions fills in m.Versions with the list of known versions.
func addVersions(m *modinfo.ModulePublic) {
	m.Versions, _ = versions(m.Path)
}

func moduleInfo(m module.Version, fromBuildList bool) *modinfo.ModulePublic {
	if m == Target {
		info := &modinfo.ModulePublic{
			Path:    m.Path,
			Version: m.Version,
			Main:    true,
		}
		if HasModRoot() {
			info.Dir = ModRoot()
			info.GoMod = filepath.Join(info.Dir, "go.mod")
			if modFile.Go != nil {
				info.GoVersion = modFile.Go.Version
			}
		}
		return info
	}

	info := &modinfo.ModulePublic{
		Path:     m.Path,
		Version:  m.Version,
		Indirect: fromBuildList && loaded != nil && !loaded.direct[m.Path],
	}
	if loaded != nil {
		info.GoVersion = loaded.goVersion[m.Path]
	}

	if cfg.BuildMod == "vendor" {
		info.Dir = filepath.Join(ModRoot(), "vendor", m.Path)
		return info
	}

	// complete fills in the extra fields in m.
	complete := func(m *modinfo.ModulePublic) {
		if m.Version != "" {
			if q, err := Query(m.Path, m.Version, nil); err != nil {
				m.Error = &modinfo.ModuleError{Err: err.Error()}
			} else {
				m.Version = q.Version
				m.Time = &q.Time
			}

			mod := module.Version{Path: m.Path, Version: m.Version}
			gomod, err := modfetch.CachePath(mod, "mod")
			if err == nil {
				if info, err := os.Stat(gomod); err == nil && info.Mode().IsRegular() {
					m.GoMod = gomod
				}
			}
			dir, err := modfetch.DownloadDir(mod)
			if err == nil {
				if info, err := os.Stat(dir); err == nil && info.IsDir() {
					m.Dir = dir
				}
			}
		}
	}

	if !fromBuildList {
		complete(info)
		return info
	}

	r := Replacement(m)
	if r.Path == "" {
		complete(info)
		return info
	}

	// Don't hit the network to fill in extra data for replaced modules.
	// The original resolved Version and Time don't matter enough to be
	// worth the cost, and we're going to overwrite the GoMod and Dir from the
	// replacement anyway. See https://golang.org/issue/27859.
	info.Replace = &modinfo.ModulePublic{
		Path:      r.Path,
		Version:   r.Version,
		GoVersion: info.GoVersion,
	}
	if r.Version == "" {
		if filepath.IsAbs(r.Path) {
			info.Replace.Dir = r.Path
		} else {
			info.Replace.Dir = filepath.Join(ModRoot(), r.Path)
		}
	}
	complete(info.Replace)
	info.Dir = info.Replace.Dir
	info.GoMod = filepath.Join(info.Dir, "go.mod")
	return info
}

func PackageBuildInfo(path string, deps []string) string {
	if isStandardImportPath(path) || !Enabled() {
		return ""
	}

	target := findModule(path, path)
	mdeps := make(map[module.Version]bool)
	for _, dep := range deps {
		if !isStandardImportPath(dep) {
			mdeps[findModule(path, dep)] = true
		}
	}
	var mods []module.Version
	delete(mdeps, target)
	for mod := range mdeps {
		mods = append(mods, mod)
	}
	module.Sort(mods)

	var buf bytes.Buffer
	fmt.Fprintf(&buf, "path\t%s\n", path)
	tv := target.Version
	if tv == "" {
		tv = "(devel)"
	}
	fmt.Fprintf(&buf, "mod\t%s\t%s\t%s\n", target.Path, tv, modfetch.Sum(target))
	for _, mod := range mods {
		mv := mod.Version
		if mv == "" {
			mv = "(devel)"
		}
		r := Replacement(mod)
		h := ""
		if r.Path == "" {
			h = "\t" + modfetch.Sum(mod)
		}
		fmt.Fprintf(&buf, "dep\t%s\t%s%s\n", mod.Path, mv, h)
		if r.Path != "" {
			fmt.Fprintf(&buf, "=>\t%s\t%s\t%s\n", r.Path, r.Version, modfetch.Sum(r))
		}
	}
	return buf.String()
}

// findModule returns the module containing the package at path,
// needed to build the package at target.
func findModule(target, path string) module.Version {
	pkg, ok := loaded.pkgCache.Get(path).(*loadPkg)
	if ok {
		if pkg.err != nil {
			base.Fatalf("build %v: cannot load %v: %v", target, path, pkg.err)
		}
		return pkg.mod
	}

	if path == "command-line-arguments" {
		return Target
	}

	if printStackInDie {
		debug.PrintStack()
	}
	base.Fatalf("build %v: cannot find module for path %v", target, path)
	panic("unreachable")
}

func ModInfoProg(info string) []byte {
	// Inject a variable with the debug information as runtime.modinfo,
	// but compile it in package main so that it is specific to the binary.
	// The variable must be a literal so that it will have the correct value
	// before the initializer for package main runs.
	//
	// The runtime startup code refers to the variable, which keeps it live in all binaries.
	return []byte(fmt.Sprintf(`package main
import _ "unsafe"
//go:linkname __debug_modinfo__ runtime.modinfo
var __debug_modinfo__ = %q
	`, string(infoStart)+info+string(infoEnd)))
}
