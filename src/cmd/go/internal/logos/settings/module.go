// Copyright 2026 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

package settings

import (
	"internal/logosfile"
	"io/fs"
	"os"
	"path"
	"path/filepath"
	"strings"
	"sync"
	"sync/atomic"

	"cmd/go/internal/logos/config"
)

type Module struct {
	Root string
	File *config.File
	Err  error

	scanOnce sync.Once
	logos    bool
	reported atomic.Bool
}

var moduleCache = struct {
	sync.Mutex
	byRoot map[string]*Module
}{byRoot: make(map[string]*Module)}

var moduleParseCount int

func ModuleAt(root string) *Module {
	moduleCache.Lock()
	defer moduleCache.Unlock()
	if m, ok := moduleCache.byRoot[root]; ok {
		return m
	}
	file, err := config.Load(root)
	m := &Module{Root: root, File: file, Err: err}
	moduleParseCount++
	moduleCache.byRoot[root] = m
	return m
}

func (m *Module) Logos() bool {
	if m.Root == "" {
		return false
	}
	m.scanOnce.Do(func() { m.logos = treeHasLogos(m.Root) })
	return m.logos
}

func (m *Module) ClaimError() bool {
	return !m.reported.Swap(true)
}

func (m *Module) CgoOpinion(cgoEnv string) (enabled, opinion bool) {
	if m.Err != nil || m.File == nil {
		return false, true
	}
	if m.File.Build.CgoSet {
		if !m.Logos() {
			return false, false
		}
		return m.File.Build.Cgo == "allow" || m.File.Build.Cgo == "restricted", true
	}
	if cgoEnv == "0" || cgoEnv == "1" {
		return cgoEnv == "1", true
	}
	if !m.Logos() {
		return false, false
	}
	return false, true
}

const cgoDeny = "[restriction::cgo_usage] cgo is disabled"

func (m *Module) CgoAllow(cgoEnv, importPath string) (bool, string) {
	if m.File == nil {
		return false, cgoDeny
	}
	if m.File.Build.CgoSet {
		switch m.File.Build.Cgo {
		case "allow":
			return true, ""
		case "restricted":
			if CgoAllowlistAllows(m.File.Build.CgoAllowlist, importPath) {
				return true, ""
			}
			return false, "[restriction::cgo_usage] package " + importPath + " uses cgo and is not in cgo_allowlist"
		default:
			return false, cgoDeny
		}
	}
	if cgoEnv == "1" {
		return true, ""
	}
	return false, cgoDeny
}

func (m *Module) Severity(group, def string) string {
	if m.Err != nil || m.File == nil {
		return def
	}
	if v, ok := m.File.Lints.Groups[group]; ok {
		return v
	}
	return def
}

func CgoAllowlistAllows(allowlist []string, importPath string) bool {
	for _, pattern := range allowlist {
		if cgoPatternAllows(pattern, importPath) {
			return true
		}
	}
	return false
}

func cgoPatternAllows(pattern, importPath string) bool {
	if prefix, ok := strings.CutSuffix(pattern, "/**"); ok {
		return importPath == prefix || strings.HasPrefix(importPath, prefix+"/")
	}
	if prefix, ok := strings.CutSuffix(pattern, "**"); ok {
		return strings.HasPrefix(importPath, prefix)
	}
	ok, _ := path.Match(pattern, importPath)
	return ok
}

func treeHasLogos(root string) bool {
	found := false
	_ = filepath.WalkDir(root, func(p string, d fs.DirEntry, err error) error {
		if found {
			return fs.SkipAll
		}
		if err != nil {
			return nil
		}
		if d.IsDir() {
			if p != root && skipsDir(p, d.Name()) {
				return filepath.SkipDir
			}
			return nil
		}
		if logosfile.Is(d.Name()) {
			found = true
		}
		return nil
	})
	return found
}

func skipsDir(p, name string) bool {
	if name == "vendor" || name == "testdata" {
		return true
	}
	if strings.HasPrefix(name, ".") || strings.HasPrefix(name, "_") {
		return true
	}
	if info, err := os.Stat(filepath.Join(p, "go.mod")); err == nil && !info.IsDir() {
		return true
	}
	return false
}
