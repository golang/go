// Copyright 2026 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

package capabilities

import (
	"bytes"
	"encoding/json"
	"errors"
	"maps"
	"os"
	"path/filepath"
	"slices"
	"strings"
	"testing"

	"cmd/go/internal/cfg"
	"cmd/go/internal/logos/config"
)

func setLogosMode(t *testing.T) {
	t.Helper()
	old := cfg.Logos
	cfg.Logos = true
	t.Cleanup(func() { cfg.Logos = old })
}

func testConfigFile() *config.File {
	return &config.File{Capabilities: config.Capabilities{
		TransitiveReflect:       "warn",
		TransitiveExec:          "deny",
		TransitiveNet:           "allow",
		ReflectAllowlist:        []string{"example.com/r/**"},
		TransitiveExecAllowlist: []string{"example.com/x/**"},
	}}
}

func TestPolicyForLinkMainModules(t *testing.T) {
	mains := []string{"example.com/b", "example.com/a"}
	p := PolicyForLink(testConfigFile(), mains)
	if !slices.Equal(p.MainModules, mains) {
		t.Errorf("MainModules = %q, want %q", p.MainModules, mains)
	}
	if p.TransitiveExec != "deny" || p.TransitiveReflect != "warn" || p.TransitiveNet != "allow" {
		t.Errorf("severities = %q %q %q, want deny warn allow", p.TransitiveExec, p.TransitiveReflect, p.TransitiveNet)
	}
	if !slices.Equal(p.TransitiveExecAllowlist, []string{"example.com/x/**"}) {
		t.Errorf("TransitiveExecAllowlist = %q", p.TransitiveExecAllowlist)
	}
	if !slices.Equal(p.ReflectAllowlist, []string{"example.com/r/**"}) {
		t.Errorf("ReflectAllowlist = %q", p.ReflectAllowlist)
	}
}

func TestPolicyForLinkNilConfigFailsClosed(t *testing.T) {
	p := PolicyForLink(nil, []string{"example.com/m"})
	if p.TransitiveExec != "deny" || p.TransitiveNet != "deny" || p.TransitiveReflect != "warn" {
		t.Errorf("nil config severities = %q %q %q, want deny deny warn", p.TransitiveExec, p.TransitiveNet, p.TransitiveReflect)
	}
}

func TestNestedModuleRoots(t *testing.T) {
	tests := []struct {
		name   string
		mains  []string
		builds []string
		want   []string
	}{
		{
			name:   "nested under main",
			mains:  []string{"example.com/main"},
			builds: []string{"example.com/main", "example.com/main/nested", "example.com/dep"},
			want:   []string{"example.com/main/nested"},
		},
		{
			name:   "prefix without boundary is not nested",
			mains:  []string{"example.com/app"},
			builds: []string{"example.com/app", "example.com/application"},
			want:   nil,
		},
		{
			name:   "nested main stays exempt",
			mains:  []string{"example.com/app", "example.com/app/nested"},
			builds: []string{"example.com/app", "example.com/app/nested", "example.com/app/nested/sub"},
			want:   []string{"example.com/app/nested/sub"},
		},
		{
			name:   "no mains emits nothing",
			mains:  nil,
			builds: []string{"example.com/main/nested"},
			want:   nil,
		},
		{
			name:   "dedupes and sorts",
			mains:  []string{"example.com/main"},
			builds: []string{"example.com/main/z", "example.com/main/a", "example.com/main/z"},
			want:   []string{"example.com/main/a", "example.com/main/z"},
		},
	}
	for _, tc := range tests {
		t.Run(tc.name, func(t *testing.T) {
			if got := NestedModuleRoots(tc.mains, tc.builds); !slices.Equal(got, tc.want) {
				t.Errorf("NestedModuleRoots(%q, %q) = %q, want %q", tc.mains, tc.builds, got, tc.want)
			}
		})
	}
}

func TestPolicyForLinkEmitsNestedModules(t *testing.T) {
	buildModulesSet.Lock()
	saved := maps.Clone(buildModulesSet.byPath)
	buildModulesSet.byPath = make(map[string]bool)
	buildModulesSet.Unlock()
	t.Cleanup(func() {
		buildModulesSet.Lock()
		defer buildModulesSet.Unlock()
		buildModulesSet.byPath = saved
	})
	NoteBuildModule("example.com/nestemit")
	NoteBuildModule("example.com/nestemit/nested")
	NoteBuildModule("example.com/nestemit/nested/sub")
	p := PolicyForLink(testConfigFile(), []string{"example.com/nestemit"})
	for _, want := range []string{"example.com/nestemit/nested", "example.com/nestemit/nested/sub"} {
		if !slices.Contains(p.NestedModules, want) {
			t.Errorf("NestedModules = %q, missing %q", p.NestedModules, want)
		}
	}
	data, err := json.Marshal(p)
	if err != nil {
		t.Fatal(err)
	}
	if !strings.Contains(string(data), `"nested_modules":["example.com/nestemit/nested"`) {
		t.Errorf("marshaled policy missing nested_modules key: %s", data)
	}
	var back LinkPolicy
	if err := json.Unmarshal(data, &back); err != nil {
		t.Fatal(err)
	}
	if !slices.Equal(back.NestedModules, p.NestedModules) {
		t.Errorf("round trip NestedModules = %q, want %q", back.NestedModules, p.NestedModules)
	}
}

func TestSnapshotHashFlagRoundTrip(t *testing.T) {
	setLogosMode(t)
	mains := []string{"example.com/m"}
	SnapshotLinkPolicy(testConfigFile(), mains, nil)
	var h bytes.Buffer
	if err := HashLinkPolicy(&h); err != nil {
		t.Fatal(err)
	}
	if !strings.HasPrefix(h.String(), "logoslinkcap ") {
		t.Fatalf("HashLinkPolicy wrote %q, want logoslinkcap prefix", h.String())
	}
	objdir := t.TempDir()
	got := AppendLinkCapFlag([]string{"-s"}, objdir)
	if len(got) != 2 || got[0] != "-s" {
		t.Fatalf("AppendLinkCapFlag = %q, want [-s -logos.cap=...]", got)
	}
	path := strings.TrimPrefix(got[1], "-logos.cap=")
	if filepath.Dir(path) != objdir {
		t.Fatalf("policy path %q outside objdir %q", path, objdir)
	}
	data, err := os.ReadFile(path)
	if err != nil {
		t.Fatal(err)
	}
	var back LinkPolicy
	if err := json.Unmarshal(data, &back); err != nil {
		t.Fatal(err)
	}
	if !slices.Equal(back.MainModules, mains) || back.TransitiveExec != "deny" {
		t.Errorf("round trip = %+v", back)
	}
}

func TestSnapshotErrorAppendsNothing(t *testing.T) {
	setLogosMode(t)
	SnapshotLinkPolicy(nil, nil, errors.New("broken"))
	var h bytes.Buffer
	if err := HashLinkPolicy(&h); err != nil {
		t.Fatal(err)
	}
	if h.Len() != 0 {
		t.Errorf("HashLinkPolicy wrote %q after error snapshot", h.String())
	}
	got := AppendLinkCapFlag([]string{"-s"}, t.TempDir())
	if !slices.Equal(got, []string{"-s"}) {
		t.Errorf("AppendLinkCapFlag = %q, want unchanged", got)
	}
	SnapshotLinkPolicy(testConfigFile(), []string{"example.com/m"}, nil)
}

func TestSnapshotRecordsCgoAllowed(t *testing.T) {
	setLogosMode(t)
	const path = "example.com/cgo-probe"
	NoteCgoAllowed(path)
	SnapshotLinkPolicy(testConfigFile(), []string{"example.com/m"}, nil)
	objdir := t.TempDir()
	got := AppendLinkCapFlag(nil, objdir)
	if len(got) != 1 {
		t.Fatalf("AppendLinkCapFlag = %q", got)
	}
	data, err := os.ReadFile(strings.TrimPrefix(got[0], "-logos.cap="))
	if err != nil {
		t.Fatal(err)
	}
	var back LinkPolicy
	if err := json.Unmarshal(data, &back); err != nil {
		t.Fatal(err)
	}
	if !slices.Contains(back.CgoAllowed, path) {
		t.Errorf("CgoAllowed = %q, missing %q", back.CgoAllowed, path)
	}
}

func TestPolicyForLinkSharedPluginInternal(t *testing.T) {
	buildModulesSet.Lock()
	savedBuilds := maps.Clone(buildModulesSet.byPath)
	buildModulesSet.Unlock()
	cgoAllowedSet.Lock()
	savedCgo := maps.Clone(cgoAllowedSet.byPath)
	cgoAllowedSet.Unlock()
	t.Cleanup(func() {
		buildModulesSet.Lock()
		buildModulesSet.byPath = savedBuilds
		buildModulesSet.Unlock()
		cgoAllowedSet.Lock()
		cgoAllowedSet.byPath = savedCgo
		cgoAllowedSet.Unlock()
	})
	tests := []struct {
		name       string
		mains      []string
		builds     []string
		cgo        []string
		wantNested []string
	}{
		{
			name:       "shared workspace with nested dep",
			mains:      []string{"example.com/app", "example.com/lib"},
			builds:     []string{"example.com/app", "example.com/app/nested", "example.com/lib", "example.com/dep"},
			cgo:        []string{"example.com/dep"},
			wantNested: []string{"example.com/app/nested"},
		},
		{
			name:       "plugin with nested dep",
			mains:      []string{"example.com/p"},
			builds:     []string{"example.com/p", "example.com/p/nested", "example.com/dep"},
			wantNested: []string{"example.com/p/nested"},
		},
		{
			name:       "internal-link exe with cgo dep",
			mains:      []string{"example.com/m"},
			builds:     []string{"example.com/m", "example.com/dep"},
			cgo:        []string{"example.com/dep"},
			wantNested: nil,
		},
	}
	for _, tc := range tests {
		t.Run(tc.name, func(t *testing.T) {
			buildModulesSet.Lock()
			buildModulesSet.byPath = make(map[string]bool)
			buildModulesSet.Unlock()
			cgoAllowedSet.Lock()
			cgoAllowedSet.byPath = make(map[string]bool)
			cgoAllowedSet.Unlock()
			for _, b := range tc.builds {
				NoteBuildModule(b)
			}
			for _, c := range tc.cgo {
				NoteCgoAllowed(c)
			}
			p := PolicyForLink(testConfigFile(), tc.mains)
			if !slices.Equal(p.MainModules, tc.mains) {
				t.Errorf("MainModules = %q, want %q", p.MainModules, tc.mains)
			}
			if !slices.Equal(p.NestedModules, tc.wantNested) {
				t.Errorf("NestedModules = %q, want %q", p.NestedModules, tc.wantNested)
			}
			if !slices.Equal(p.CgoAllowed, tc.cgo) {
				t.Errorf("CgoAllowed = %q, want %q", p.CgoAllowed, tc.cgo)
			}
			if p.TransitiveReflect != "warn" || p.TransitiveExec != "deny" || p.TransitiveNet != "allow" {
				t.Errorf("severities = %q %q %q, want warn deny allow", p.TransitiveReflect, p.TransitiveExec, p.TransitiveNet)
			}
			data, err := json.Marshal(p)
			if err != nil {
				t.Fatal(err)
			}
			var back LinkPolicy
			if err := json.Unmarshal(data, &back); err != nil {
				t.Fatal(err)
			}
			if !slices.Equal(back.MainModules, tc.mains) {
				t.Errorf("round trip MainModules = %q, want %q", back.MainModules, tc.mains)
			}
			if !slices.Equal(back.NestedModules, tc.wantNested) {
				t.Errorf("round trip NestedModules = %q, want %q", back.NestedModules, tc.wantNested)
			}
			if !slices.Equal(back.CgoAllowed, tc.cgo) {
				t.Errorf("round trip CgoAllowed = %q, want %q", back.CgoAllowed, tc.cgo)
			}
		})
	}
}

func TestLinkPolicyPlainGoUnchanged(t *testing.T) {
	old := cfg.Logos
	cfg.Logos = false
	t.Cleanup(func() { cfg.Logos = old })
	linkPolicySnapshot.Lock()
	savedSet, savedData := linkPolicySnapshot.set, linkPolicySnapshot.data
	linkPolicySnapshot.Unlock()
	t.Cleanup(func() {
		linkPolicySnapshot.Lock()
		linkPolicySnapshot.set, linkPolicySnapshot.data = savedSet, savedData
		linkPolicySnapshot.Unlock()
	})
	SnapshotLinkPolicy(testConfigFile(), []string{"example.com/m"}, nil)
	var h bytes.Buffer
	if err := HashLinkPolicy(&h); err != nil {
		t.Fatal(err)
	}
	if h.Len() != 0 {
		t.Errorf("HashLinkPolicy outside logos mode wrote %q, want nothing", h.String())
	}
	objdir := t.TempDir()
	got := AppendLinkCapFlag([]string{"-s"}, objdir)
	if !slices.Equal(got, []string{"-s"}) {
		t.Errorf("AppendLinkCapFlag outside logos mode = %q, want unchanged", got)
	}
	if entries, err := os.ReadDir(objdir); err != nil || len(entries) != 0 {
		t.Errorf("AppendLinkCapFlag outside logos mode wrote %d files, want none (err=%v)", len(entries), err)
	}
}
