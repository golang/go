// Copyright 2018 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

package modload

import (
	"bytes"
	"context"
	"encoding/hex"
	"errors"
	"fmt"
	"internal/goroot"
	"os"
	"path/filepath"
	"strings"

	"cmd/go/internal/base"
	"cmd/go/internal/cfg"
	"cmd/go/internal/modfetch"
	"cmd/go/internal/modinfo"
	"cmd/go/internal/search"

	"golang.org/x/mod/module"
	"golang.org/x/mod/semver"
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

// PackageModuleInfo returns information about the module that provides
// a given package. If modules are not enabled or if the package is in the
// standard library or if the package was not successfully loaded with
// LoadPackages or ImportFromFiles, nil is returned.
func PackageModuleInfo(pkgpath string) *modinfo.ModulePublic {
	if isStandardImportPath(pkgpath) || !Enabled() {
		return nil
	}
	m, ok := findModule(pkgpath)
	if !ok {
		return nil
	}
	fromBuildList := true
	listRetracted := false
	return moduleInfo(context.TODO(), m, fromBuildList, listRetracted)
}

func ModuleInfo(ctx context.Context, path string) *modinfo.ModulePublic {
	if !Enabled() {
		return nil
	}

	listRetracted := false
	if i := strings.Index(path, "@"); i >= 0 {
		m := module.Version{Path: path[:i], Version: path[i+1:]}
		fromBuildList := false
		return moduleInfo(ctx, m, fromBuildList, listRetracted)
	}

	for _, m := range buildList {
		if m.Path == path {
			fromBuildList := true
			return moduleInfo(ctx, m, fromBuildList, listRetracted)
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
func addUpdate(ctx context.Context, m *modinfo.ModulePublic) {
	if m.Version == "" {
		return
	}

	if info, err := Query(ctx, m.Path, "upgrade", m.Version, CheckAllowed); err == nil && semver.Compare(info.Version, m.Version) > 0 {
		m.Update = &modinfo.ModulePublic{
			Path:    m.Path,
			Version: info.Version,
			Time:    &info.Time,
		}
	}
}

// addVersions fills in m.Versions with the list of known versions.
// Excluded versions will be omitted. If listRetracted is false, retracted
// versions will also be omitted.
func addVersions(ctx context.Context, m *modinfo.ModulePublic, listRetracted bool) {
	allowed := CheckAllowed
	if listRetracted {
		allowed = CheckExclusions
	}
	m.Versions, _ = versions(ctx, m.Path, allowed)
}

// addRetraction fills in m.Retracted if the module was retracted by its author.
// m.Error is set if there's an error loading retraction information.
func addRetraction(ctx context.Context, m *modinfo.ModulePublic) {
	if m.Version == "" {
		return
	}

	err := CheckRetractions(ctx, module.Version{Path: m.Path, Version: m.Version})
	var rerr *ModuleRetractedError
	if errors.As(err, &rerr) {
		if len(rerr.Rationale) == 0 {
			m.Retracted = []string{"retracted by module author"}
		} else {
			m.Retracted = rerr.Rationale
		}
	} else if err != nil && m.Error == nil {
		m.Error = &modinfo.ModuleError{Err: err.Error()}
	}
}

func moduleInfo(ctx context.Context, m module.Version, fromBuildList, listRetracted bool) *modinfo.ModulePublic {
	if m == Target {
		info := &modinfo.ModulePublic{
			Path:    m.Path,
			Version: m.Version,
			Main:    true,
		}
		if HasModRoot() {
			info.Dir = ModRoot()
			info.GoMod = ModFilePath()
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
	if v, ok := rawGoVersion.Load(m); ok {
		info.GoVersion = v.(string)
	}

	// completeFromModCache fills in the extra fields in m using the module cache.
	completeFromModCache := func(m *modinfo.ModulePublic) {
		mod := module.Version{Path: m.Path, Version: m.Version}

		if m.Version != "" {
			if q, err := Query(ctx, m.Path, m.Version, "", nil); err != nil {
				m.Error = &modinfo.ModuleError{Err: err.Error()}
			} else {
				m.Version = q.Version
				m.Time = &q.Time
			}

			gomod, err := modfetch.CachePath(mod, "mod")
			if err == nil {
				if info, err := os.Stat(gomod); err == nil && info.Mode().IsRegular() {
					m.GoMod = gomod
				}
			}
			dir, err := modfetch.DownloadDir(mod)
			if err == nil {
				m.Dir = dir
			}

			if listRetracted {
				addRetraction(ctx, m)
			}
		}

		if m.GoVersion == "" {
			if summary, err := rawGoModSummary(mod); err == nil && summary.goVersionV != "" {
				m.GoVersion = summary.goVersionV[1:]
			}
		}
	}

	if !fromBuildList {
		// If this was an explicitly-versioned argument to 'go mod download' or
		// 'go list -m', report the actual requested version, not its replacement.
		completeFromModCache(info) // Will set m.Error in vendor mode.
		return info
	}

	r := Replacement(m)
	if r.Path == "" {
		if cfg.BuildMod == "vendor" {
			// It's tempting to fill in the "Dir" field to point within the vendor
			// directory, but that would be misleading: the vendor directory contains
			// a flattened package tree, not complete modules, and it can even
			// interleave packages from different modules if one module path is a
			// prefix of the other.
		} else {
			completeFromModCache(info)
		}
		return info
	}

	// Don't hit the network to fill in extra data for replaced modules.
	// The original resolved Version and Time don't matter enough to be
	// worth the cost, and we're going to overwrite the GoMod and Dir from the
	// replacement anyway. See https://golang.org/issue/27859.
	info.Replace = &modinfo.ModulePublic{
		Path:    r.Path,
		Version: r.Version,
	}
	if v, ok := rawGoVersion.Load(m); ok {
		info.Replace.GoVersion = v.(string)
	}
	if r.Version == "" {
		if filepath.IsAbs(r.Path) {
			info.Replace.Dir = r.Path
		} else {
			info.Replace.Dir = filepath.Join(ModRoot(), r.Path)
		}
		info.Replace.GoMod = filepath.Join(info.Replace.Dir, "go.mod")
	}
	if cfg.BuildMod != "vendor" {
		completeFromModCache(info.Replace)
		info.Dir = info.Replace.Dir
		info.GoMod = info.Replace.GoMod
		info.Retracted = info.Replace.Retracted
	}
	info.GoVersion = info.Replace.GoVersion
	return info
}

// PackageBuildInfo returns a string containing module version information
// for modules providing packages named by path and deps. path and deps must
// name packages that were resolved successfully with LoadPackages.
func PackageBuildInfo(path string, deps []string) string {
	if isStandardImportPath(path) || !Enabled() {
		return ""
	}

	target := mustFindModule(path, path)
	mdeps := make(map[module.Version]bool)
	for _, dep := range deps {
		if !isStandardImportPath(dep) {
			mdeps[mustFindModule(path, dep)] = true
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

	writeEntry := func(token string, m module.Version) {
		mv := m.Version
		if mv == "" {
			mv = "(devel)"
		}
		fmt.Fprintf(&buf, "%s\t%s\t%s", token, m.Path, mv)
		if r := Replacement(m); r.Path == "" {
			fmt.Fprintf(&buf, "\t%s\n", modfetch.Sum(m))
		} else {
			fmt.Fprintf(&buf, "\n=>\t%s\t%s\t%s\n", r.Path, r.Version, modfetch.Sum(r))
		}
	}

	writeEntry("mod", target)
	for _, mod := range mods {
		writeEntry("dep", mod)
	}

	return buf.String()
}

// mustFindModule is like findModule, but it calls base.Fatalf if the
// module can't be found.
//
// TODO(jayconrod): remove this. Callers should use findModule and return
// errors instead of relying on base.Fatalf.
func mustFindModule(target, path string) module.Version {
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

	base.Fatalf("build %v: cannot find module for path %v", target, path)
	panic("unreachable")
}

// findModule searches for the module that contains the package at path.
// If the package was loaded, its containing module and true are returned.
// Otherwise, module.Version{} and false are returend.
func findModule(path string) (module.Version, bool) {
	if pkg, ok := loaded.pkgCache.Get(path).(*loadPkg); ok {
		return pkg.mod, pkg.mod != module.Version{}
	}
	if path == "command-line-arguments" {
		return Target, true
	}
	return module.Version{}, false
}

func ModInfoProg(info string, isgccgo bool) []byte {
	// Inject a variable with the debug information as runtime.modinfo,
	// but compile it in package main so that it is specific to the binary.
	// The variable must be a literal so that it will have the correct value
	// before the initializer for package main runs.
	//
	// The runtime startup code refers to the variable, which keeps it live
	// in all binaries.
	//
	// Note: we use an alternate recipe below for gccgo (based on an
	// init function) due to the fact that gccgo does not support
	// applying a "//go:linkname" directive to a variable. This has
	// drawbacks in that other packages may want to look at the module
	// info in their init functions (see issue 29628), which won't
	// work for gccgo. See also issue 30344.

	if !isgccgo {
		return []byte(fmt.Sprintf(`package main
import _ "unsafe"
//go:linkname __debug_modinfo__ runtime.modinfo
var __debug_modinfo__ = %q
`, string(infoStart)+info+string(infoEnd)))
	} else {
		return []byte(fmt.Sprintf(`package main
import _ "unsafe"
//go:linkname __set_debug_modinfo__ runtime.setmodinfo
func __set_debug_modinfo__(string)
func init() { __set_debug_modinfo__(%q) }
`, string(infoStart)+info+string(infoEnd)))
	}
}
