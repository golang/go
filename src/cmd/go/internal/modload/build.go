// Copyright 2018 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

package modload

import (
	"context"
	"encoding/hex"
	"errors"
	"fmt"
	"io/fs"
	"os"
	"path/filepath"
	"strings"

	"cmd/go/internal/base"
	"cmd/go/internal/cfg"
	"cmd/go/internal/gover"
	"cmd/go/internal/modfetch"
	"cmd/go/internal/modfetch/codehost"
	"cmd/go/internal/modindex"
	"cmd/go/internal/modinfo"
	"cmd/go/internal/search"

	"golang.org/x/mod/module"
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
		if modindex.IsStandardPackage(cfg.GOROOT, cfg.BuildContext.Compiler, path) {
			return filepath.Join(cfg.GOROOT, "src", path)
		}
	}
	return ""
}

// PackageModuleInfo returns information about the module that provides
// a given package. If modules are not enabled or if the package is in the
// standard library or if the package was not successfully loaded with
// LoadPackages or ImportFromFiles, nil is returned.
func PackageModuleInfo(ctx context.Context, pkgpath string) *modinfo.ModulePublic {
	if isStandardImportPath(pkgpath) || !Enabled() {
		return nil
	}
	m, ok := findModule(loaded, pkgpath)
	if !ok {
		return nil
	}

	rs := LoadModFile(ctx)
	return moduleInfo(ctx, rs, m, 0, nil)
}

// PackageModRoot returns the module root directory for the module that provides
// a given package. If modules are not enabled or if the package is in the
// standard library or if the package was not successfully loaded with
// LoadPackages or ImportFromFiles, the empty string is returned.
func PackageModRoot(ctx context.Context, pkgpath string) string {
	if isStandardImportPath(pkgpath) || !Enabled() || cfg.BuildMod == "vendor" {
		return ""
	}
	m, ok := findModule(loaded, pkgpath)
	if !ok {
		return ""
	}
	root, _, err := fetch(ctx, m)
	if err != nil {
		return ""
	}
	return root
}

func ModuleInfo(ctx context.Context, path string) *modinfo.ModulePublic {
	if !Enabled() {
		return nil
	}

	if path, vers, found := strings.Cut(path, "@"); found {
		m := module.Version{Path: path, Version: vers}
		return moduleInfo(ctx, nil, m, 0, nil)
	}

	rs := LoadModFile(ctx)

	var (
		v  string
		ok bool
	)
	if rs.pruning == pruned {
		v, ok = rs.rootSelected(path)
	}
	if !ok {
		mg, err := rs.Graph(ctx)
		if err != nil {
			base.Fatal(err)
		}
		v = mg.Selected(path)
	}

	if v == "none" {
		return &modinfo.ModulePublic{
			Path: path,
			Error: &modinfo.ModuleError{
				Err: "module not in current build",
			},
		}
	}

	return moduleInfo(ctx, rs, module.Version{Path: path, Version: v}, 0, nil)
}

// addUpdate fills in m.Update if an updated version is available.
func addUpdate(ctx context.Context, m *modinfo.ModulePublic) {
	if m.Version == "" {
		return
	}

	info, err := Query(ctx, m.Path, "upgrade", m.Version, CheckAllowed)
	var noVersionErr *NoMatchingVersionError
	if errors.Is(err, ErrDisallowed) ||
		errors.Is(err, fs.ErrNotExist) ||
		errors.As(err, &noVersionErr) {
		// Ignore "not found" and "no matching version" errors.
		// This means the proxy has no matching version or no versions at all.
		//
		// Ignore "disallowed" errors. This means the current version is
		// excluded or retracted and there are no higher allowed versions.
		//
		// We should report other errors though. An attacker that controls the
		// network shouldn't be able to hide versions by interfering with
		// the HTTPS connection. An attacker that controls the proxy may still
		// hide versions, since the "list" and "latest" endpoints are not
		// authenticated.
		return
	} else if err != nil {
		if m.Error == nil {
			m.Error = &modinfo.ModuleError{Err: err.Error()}
		}
		return
	}

	if gover.ModCompare(m.Path, info.Version, m.Version) > 0 {
		m.Update = &modinfo.ModulePublic{
			Path:    m.Path,
			Version: info.Version,
			Time:    info.Time,
		}
	}
}

// mergeOrigin returns the union of data from two origins,
// returning either a new origin or one of its unmodified arguments.
// If the two origins conflict including if either is nil,
// mergeOrigin returns nil.
func mergeOrigin(m1, m2 *codehost.Origin) *codehost.Origin {
	if m1 == nil || m2 == nil {
		return nil
	}

	if m2.VCS != m1.VCS ||
		m2.URL != m1.URL ||
		m2.Subdir != m1.Subdir {
		return nil
	}

	merged := *m1
	if m2.Hash != "" {
		if m1.Hash != "" && m1.Hash != m2.Hash {
			return nil
		}
		merged.Hash = m2.Hash
	}
	if m2.TagSum != "" {
		if m1.TagSum != "" && (m1.TagSum != m2.TagSum || m1.TagPrefix != m2.TagPrefix) {
			return nil
		}
		merged.TagSum = m2.TagSum
		merged.TagPrefix = m2.TagPrefix
	}
	if m2.Ref != "" {
		if m1.Ref != "" && m1.Ref != m2.Ref {
			return nil
		}
		merged.Ref = m2.Ref
	}

	switch {
	case merged == *m1:
		return m1
	case merged == *m2:
		return m2
	default:
		// Clone the result to avoid an alloc for merged
		// if the result is equal to one of the arguments.
		clone := merged
		return &clone
	}
}

// addVersions fills in m.Versions with the list of known versions.
// Excluded versions will be omitted. If listRetracted is false, retracted
// versions will also be omitted.
func addVersions(ctx context.Context, m *modinfo.ModulePublic, listRetracted bool) {
	// TODO(bcmills): Would it make sense to check for reuse here too?
	// Perhaps that doesn't buy us much, though: we would always have to fetch
	// all of the version tags to list the available versions anyway.

	allowed := CheckAllowed
	if listRetracted {
		allowed = CheckExclusions
	}
	v, origin, err := versions(ctx, m.Path, allowed)
	if err != nil && m.Error == nil {
		m.Error = &modinfo.ModuleError{Err: err.Error()}
	}
	m.Versions = v
	m.Origin = mergeOrigin(m.Origin, origin)
}

// addRetraction fills in m.Retracted if the module was retracted by its author.
// m.Error is set if there's an error loading retraction information.
func addRetraction(ctx context.Context, m *modinfo.ModulePublic) {
	if m.Version == "" {
		return
	}

	err := CheckRetractions(ctx, module.Version{Path: m.Path, Version: m.Version})
	var noVersionErr *NoMatchingVersionError
	var retractErr *ModuleRetractedError
	if err == nil || errors.Is(err, fs.ErrNotExist) || errors.As(err, &noVersionErr) {
		// Ignore "not found" and "no matching version" errors.
		// This means the proxy has no matching version or no versions at all.
		//
		// We should report other errors though. An attacker that controls the
		// network shouldn't be able to hide versions by interfering with
		// the HTTPS connection. An attacker that controls the proxy may still
		// hide versions, since the "list" and "latest" endpoints are not
		// authenticated.
		return
	} else if errors.As(err, &retractErr) {
		if len(retractErr.Rationale) == 0 {
			m.Retracted = []string{"retracted by module author"}
		} else {
			m.Retracted = retractErr.Rationale
		}
	} else if m.Error == nil {
		m.Error = &modinfo.ModuleError{Err: err.Error()}
	}
}

// addDeprecation fills in m.Deprecated if the module was deprecated by its
// author. m.Error is set if there's an error loading deprecation information.
func addDeprecation(ctx context.Context, m *modinfo.ModulePublic) {
	deprecation, err := CheckDeprecation(ctx, module.Version{Path: m.Path, Version: m.Version})
	var noVersionErr *NoMatchingVersionError
	if errors.Is(err, fs.ErrNotExist) || errors.As(err, &noVersionErr) {
		// Ignore "not found" and "no matching version" errors.
		// This means the proxy has no matching version or no versions at all.
		//
		// We should report other errors though. An attacker that controls the
		// network shouldn't be able to hide versions by interfering with
		// the HTTPS connection. An attacker that controls the proxy may still
		// hide versions, since the "list" and "latest" endpoints are not
		// authenticated.
		return
	}
	if err != nil {
		if m.Error == nil {
			m.Error = &modinfo.ModuleError{Err: err.Error()}
		}
		return
	}
	m.Deprecated = deprecation
}

// moduleInfo returns information about module m, loaded from the requirements
// in rs (which may be nil to indicate that m was not loaded from a requirement
// graph).
func moduleInfo(ctx context.Context, rs *Requirements, m module.Version, mode ListMode, reuse map[module.Version]*modinfo.ModulePublic) *modinfo.ModulePublic {
	if m.Version == "" && MainModules.Contains(m.Path) {
		info := &modinfo.ModulePublic{
			Path:    m.Path,
			Version: m.Version,
			Main:    true,
		}
		if v, ok := rawGoVersion.Load(m); ok {
			info.GoVersion = v.(string)
		} else {
			panic("internal error: GoVersion not set for main module")
		}
		if modRoot := MainModules.ModRoot(m); modRoot != "" {
			info.Dir = modRoot
			info.GoMod = modFilePath(modRoot)
		}
		return info
	}

	info := &modinfo.ModulePublic{
		Path:     m.Path,
		Version:  m.Version,
		Indirect: rs != nil && !rs.direct[m.Path],
	}
	if v, ok := rawGoVersion.Load(m); ok {
		info.GoVersion = v.(string)
	}

	// completeFromModCache fills in the extra fields in m using the module cache.
	completeFromModCache := func(m *modinfo.ModulePublic) {
		if gover.IsToolchain(m.Path) {
			return
		}

		checksumOk := func(suffix string) bool {
			return rs == nil || m.Version == "" || !mustHaveSums() ||
				modfetch.HaveSum(module.Version{Path: m.Path, Version: m.Version + suffix})
		}

		mod := module.Version{Path: m.Path, Version: m.Version}

		if m.Version != "" {
			if old := reuse[mod]; old != nil {
				if err := checkReuse(ctx, mod, old.Origin); err == nil {
					*m = *old
					m.Query = ""
					m.Dir = ""
					return
				}
			}

			if q, err := Query(ctx, m.Path, m.Version, "", nil); err != nil {
				m.Error = &modinfo.ModuleError{Err: err.Error()}
			} else {
				m.Version = q.Version
				m.Time = q.Time
			}
		}

		if m.GoVersion == "" && checksumOk("/go.mod") {
			// Load the go.mod file to determine the Go version, since it hasn't
			// already been populated from rawGoVersion.
			if summary, err := rawGoModSummary(mod); err == nil && summary.goVersion != "" {
				m.GoVersion = summary.goVersion
			}
		}

		if m.Version != "" {
			if checksumOk("/go.mod") {
				gomod, err := modfetch.CachePath(ctx, mod, "mod")
				if err == nil {
					if info, err := os.Stat(gomod); err == nil && info.Mode().IsRegular() {
						m.GoMod = gomod
					}
				}
				if gomodsum, ok := modfetch.RecordedSum(modkey(mod)); ok {
					m.GoModSum = gomodsum
				}
			}
			if checksumOk("") {
				dir, err := modfetch.DownloadDir(ctx, mod)
				if err == nil {
					m.Dir = dir
				}
				if sum, ok := modfetch.RecordedSum(mod); ok {
					m.Sum = sum
				}
			}

			if mode&ListRetracted != 0 {
				addRetraction(ctx, m)
			}
		}
	}

	if rs == nil {
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
			info.Replace.Dir = filepath.Join(replaceRelativeTo(), r.Path)
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

// findModule searches for the module that contains the package at path.
// If the package was loaded, its containing module and true are returned.
// Otherwise, module.Version{} and false are returned.
func findModule(ld *loader, path string) (module.Version, bool) {
	if pkg, ok := ld.pkgCache.Get(path); ok {
		return pkg.mod, pkg.mod != module.Version{}
	}
	return module.Version{}, false
}

func ModInfoProg(info string, isgccgo bool) []byte {
	// Inject an init function to set runtime.modinfo.
	// This is only used for gccgo - with gc we hand the info directly to the linker.
	// The init function has the drawback that packages may want to
	// look at the module info in their init functions (see issue 29628),
	// which won't work. See also issue 30344.
	if isgccgo {
		return fmt.Appendf(nil, `package main
import _ "unsafe"
//go:linkname __set_debug_modinfo__ runtime.setmodinfo
func __set_debug_modinfo__(string)
func init() { __set_debug_modinfo__(%q) }
`, ModInfoData(info))
	}
	return nil
}

func ModInfoData(info string) []byte {
	return []byte(string(infoStart) + info + string(infoEnd))
}
