// Copyright 2026 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

package capabilities

import (
	"crypto/sha256"
	"encoding/hex"
	"encoding/json"
	"maps"
	"os"
	"path/filepath"
	"slices"
	"sync"

	"cmd/go/internal/cfg"
	"cmd/go/internal/logos/config"
	"cmd/go/internal/modinfo"
)

// LinkPolicy is the JSON policy cmd/go writes for the linker -logos.cap
// flag (M12-S2). Severities follow the [capabilities] table; MainModules
// holds the main module paths (first-party packages under them are never
// blamed); CgoAllowed holds the import paths whose cgo use the M0 policy
// allowed.
//
// The wire format is encoding/json v1 with a plain struct on both sides:
// cmd/go and cmd/link build in every experiment configuration, while
// encoding/json/v2 is gated behind goexperiment.jsonv2.
type LinkPolicy struct {
	TransitiveReflect       string   `json:"transitive_reflect"`
	TransitiveExec          string   `json:"transitive_exec"`
	TransitiveNet           string   `json:"transitive_net"`
	ReflectAllowlist        []string `json:"reflect_allowlist"`
	TransitiveExecAllowlist []string `json:"transitive_exec_allowlist"`
	TransitiveNetAllowlist  []string `json:"transitive_net_allowlist"`
	MainModules             []string `json:"main_modules"`
	CgoAllowed              []string `json:"cgo_allowed"`
}

// PolicyForLink builds the linker policy from a parsed logos.toml, the
// main module, and the recorded cgo allowances.
func PolicyForLink(cf *config.File, mod *modinfo.ModulePublic) *LinkPolicy {
	p := &LinkPolicy{CgoAllowed: CgoAllowed()}
	if mod != nil && mod.Path != "" {
		p.MainModules = []string{mod.Path}
	}
	if cf == nil {
		p.TransitiveReflect = string(Warn)
		p.TransitiveExec = string(Deny)
		p.TransitiveNet = string(Deny)
		return p
	}
	p.TransitiveReflect = cf.Capabilities.TransitiveReflect
	p.TransitiveExec = cf.Capabilities.TransitiveExec
	p.TransitiveNet = cf.Capabilities.TransitiveNet
	p.ReflectAllowlist = cf.Capabilities.ReflectAllowlist
	p.TransitiveExecAllowlist = cf.Capabilities.TransitiveExecAllowlist
	p.TransitiveNetAllowlist = cf.Capabilities.TransitiveNetAllowlist
	return p
}

// MarshalPolicy encodes a linker policy as JSON.
func MarshalPolicy(p *LinkPolicy) ([]byte, error) {
	return json.Marshal(p)
}

// UnmarshalPolicy decodes a linker policy from JSON.
func UnmarshalPolicy(data []byte) (*LinkPolicy, error) {
	var p LinkPolicy
	if err := json.Unmarshal(data, &p); err != nil {
		return nil, err
	}
	return &p, nil
}

// cgoAllowedSet records the import paths whose cgo use the M0 policy
// allowed during load. The set is process-global because one cmd/go
// invocation loads once and links after; entries are write-once.
var cgoAllowedSet = struct {
	sync.Mutex
	byPath map[string]bool
}{byPath: make(map[string]bool)}

// NoteCgoAllowed records importPath as allowed cgo use for the linker
// audit. Callers: load.checkLogosCgo records every package it allows;
// Go-module packages never reach it, so their cgo use stays unrecorded
// and the linker flags runtime.cgocall from them.
func NoteCgoAllowed(importPath string) {
	cgoAllowedSet.Lock()
	defer cgoAllowedSet.Unlock()
	cgoAllowedSet.byPath[importPath] = true
}

// CgoAllowed returns the recorded cgo-allowed import paths, sorted.
func CgoAllowed() []string {
	cgoAllowedSet.Lock()
	defer cgoAllowedSet.Unlock()
	return slices.Sorted(maps.Keys(cgoAllowedSet.byPath))
}

// AppendLinkCapFlag appends the linker flag carrying the capabilities
// policy for one linked binary, or returns ldflags unchanged outside
// logos mode. The policy file is content-addressed under the temporary
// directory, so identical policies reuse one path (stable action IDs) and
// changed policies force a relink. A config load failure appends nothing;
// the load phase already failed the build before linking.
func AppendLinkCapFlag(ldflags []string, mod *modinfo.ModulePublic) []string {
	if !cfg.Logos {
		return ldflags
	}
	return append(ldflags, linkCapFlag(mod)...)
}

// linkCapFlag builds the -logos.cap flag for one linked binary.
func linkCapFlag(mod *modinfo.ModulePublic) []string {
	modRoot := ""
	if mod != nil {
		modRoot = mod.Dir
	}
	cf, err := config.Load(modRoot)
	if err != nil {
		return nil
	}
	data, err := MarshalPolicy(PolicyForLink(cf, mod))
	if err != nil {
		return nil
	}
	sum := sha256.Sum256(data)
	name := "logos-cap-" + hex.EncodeToString(sum[:8]) + ".json"
	path := filepath.Join(os.TempDir(), name)
	f, err := os.OpenFile(path, os.O_WRONLY|os.O_CREATE|os.O_EXCL, 0644)
	if err == nil {
		if _, err := f.Write(data); err != nil {
			f.Close()
			return nil
		}
		f.Close()
	}
	return []string{"-logos.cap=" + path}
}
