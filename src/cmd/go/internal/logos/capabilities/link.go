// Copyright 2026 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

package capabilities

import (
	"crypto/sha256"
	"encoding/json"
	"fmt"
	"io"
	"maps"
	"os"
	"path/filepath"
	"slices"
	"strings"
	"sync"

	"cmd/go/internal/cfg"
	"cmd/go/internal/logos/config"
)

type LinkPolicy struct {
	TransitiveReflect       string   `json:"transitive_reflect"`
	TransitiveExec          string   `json:"transitive_exec"`
	TransitiveNet           string   `json:"transitive_net"`
	ReflectAllowlist        []string `json:"reflect_allowlist"`
	TransitiveExecAllowlist []string `json:"transitive_exec_allowlist"`
	TransitiveNetAllowlist  []string `json:"transitive_net_allowlist"`
	MainModules             []string `json:"main_modules"`
	NestedModules           []string `json:"nested_modules"`
	CgoAllowed              []string `json:"cgo_allowed"`
}

func PolicyForLink(cf *config.File, mainModules []string) *LinkPolicy {
	p := &LinkPolicy{CgoAllowed: CgoAllowed(), MainModules: mainModules, NestedModules: NestedModuleRoots(mainModules, BuildModules())}
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

var cgoAllowedSet = struct {
	sync.Mutex
	byPath map[string]bool
}{byPath: make(map[string]bool)}

func NoteCgoAllowed(importPath string) {
	cgoAllowedSet.Lock()
	defer cgoAllowedSet.Unlock()
	cgoAllowedSet.byPath[importPath] = true
}

func CgoAllowed() []string {
	cgoAllowedSet.Lock()
	defer cgoAllowedSet.Unlock()
	return slices.Sorted(maps.Keys(cgoAllowedSet.byPath))
}

func NestedModuleRoots(mainModules, buildModules []string) []string {
	isMain := make(map[string]bool, len(mainModules))
	for _, m := range mainModules {
		if m != "" {
			isMain[m] = true
		}
	}
	seen := make(map[string]bool)
	var out []string
	for _, b := range buildModules {
		if b == "" || isMain[b] || seen[b] {
			continue
		}
		for _, m := range mainModules {
			if m != "" && strings.HasPrefix(b, m+"/") {
				out = append(out, b)
				seen[b] = true
				break
			}
		}
	}
	slices.Sort(out)
	return out
}

var buildModulesSet = struct {
	sync.Mutex
	byPath map[string]bool
}{byPath: make(map[string]bool)}

func NoteBuildModule(modulePath string) {
	if modulePath == "" {
		return
	}
	buildModulesSet.Lock()
	defer buildModulesSet.Unlock()
	buildModulesSet.byPath[modulePath] = true
}

func BuildModules() []string {
	buildModulesSet.Lock()
	defer buildModulesSet.Unlock()
	return slices.Sorted(maps.Keys(buildModulesSet.byPath))
}

var linkPolicySnapshot = struct {
	sync.Mutex
	set  bool
	data []byte
}{}

func SnapshotLinkPolicy(cf *config.File, mainModules []string, err error) {
	linkPolicySnapshot.Lock()
	defer linkPolicySnapshot.Unlock()
	linkPolicySnapshot.set = false
	linkPolicySnapshot.data = nil
	if err != nil || cf == nil {
		return
	}
	data, merr := json.Marshal(PolicyForLink(cf, mainModules))
	if merr != nil {
		return
	}
	linkPolicySnapshot.set = true
	linkPolicySnapshot.data = data
}

func HashLinkPolicy(h io.Writer) error {
	if !cfg.Logos {
		return nil
	}
	linkPolicySnapshot.Lock()
	data := linkPolicySnapshot.data
	set := linkPolicySnapshot.set
	linkPolicySnapshot.Unlock()
	if !set {
		return nil
	}
	sum := sha256.Sum256(data)
	_, err := fmt.Fprintf(h, "logoslinkcap %x\n", sum)
	return err
}

func AppendLinkCapFlag(ldflags []string, objdir string) []string {
	if !cfg.Logos {
		return ldflags
	}
	linkPolicySnapshot.Lock()
	data := linkPolicySnapshot.data
	set := linkPolicySnapshot.set
	linkPolicySnapshot.Unlock()
	if !set {
		return ldflags
	}
	path := filepath.Join(objdir, "logos-cap.json")
	if err := os.WriteFile(path, data, 0644); err != nil {
		return ldflags
	}
	return append(ldflags, "-logos.cap="+path)
}
