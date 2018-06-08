// Copyright 2018 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

package vgo

import (
	"bytes"
	"cmd/go/internal/base"
	"cmd/go/internal/cfg"
	"cmd/go/internal/modinfo"
	"cmd/go/internal/module"
	"cmd/go/internal/search"
	"encoding/hex"
	"fmt"
	"os"
	"path/filepath"
)

var (
	infoStart, _ = hex.DecodeString("3077af0c9274080241e1c107e6d618e6")
	infoEnd, _   = hex.DecodeString("f932433186182072008242104116d8f2")
)

func isStandardImportPath(path string) bool {
	if search.IsStandardImportPath(path) {
		if _, err := os.Stat(filepath.Join(cfg.GOROOT, "src", path)); err == nil {
			return true
		}
		if _, err := os.Stat(filepath.Join(cfg.GOROOT, "src/vendor", path)); err == nil {
			return true
		}
	}
	return false
}

func PackageModuleInfo(path string) *modinfo.ModulePublic {
	var info modinfo.ModulePublic
	if isStandardImportPath(path) || !Enabled() {
		return nil
	}
	target := findModule(path, path)
	info.Top = target.Path == buildList[0].Path
	info.Path = target.Path
	info.Version = target.Version
	return &info
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
	sortModules(mods)

	var buf bytes.Buffer
	fmt.Fprintf(&buf, "path\t%s\n", path)
	tv := target.Version
	if tv == "" {
		tv = "(devel)"
	}
	fmt.Fprintf(&buf, "mod\t%s\t%s\t%s\n", target.Path, tv, findModHash(target))
	for _, mod := range mods {
		mv := mod.Version
		if mv == "" {
			mv = "(devel)"
		}
		r := replaced(mod)
		h := ""
		if r == nil {
			h = "\t" + findModHash(mod)
		}
		fmt.Fprintf(&buf, "dep\t%s\t%s%s\n", mod.Path, mod.Version, h)
		if r := replaced(mod); r != nil {
			fmt.Fprintf(&buf, "=>\t%s\t%s\t%s\n", r.New.Path, r.New.Version, findModHash(r.New))
		}
	}
	return buf.String()
}

func findModule(target, path string) module.Version {
	if path == "." {
		return buildList[0]
	}
	for _, mod := range buildList {
		if importPathInModule(path, mod.Path) {
			return mod
		}
	}
	base.Fatalf("build %v: cannot find module for path %v", target, path)
	panic("unreachable")
}

func ModInfoProg(info string) []byte {
	return []byte(fmt.Sprintf(`
		package main
		import _ "unsafe"
		//go:linkname __debug_modinfo__ runtime/debug.modinfo
		var __debug_modinfo__ string
		func init() {
			__debug_modinfo__ = %q
		}
	`, string(infoStart)+info+string(infoEnd)))
}
