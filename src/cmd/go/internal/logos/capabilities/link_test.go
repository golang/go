// Copyright 2026 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

package capabilities

import (
	"bytes"
	"errors"
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
}

func TestPolicyForLinkNilConfigFailsClosed(t *testing.T) {
	p := PolicyForLink(nil, []string{"example.com/m"})
	if p.TransitiveExec != "deny" || p.TransitiveNet != "deny" || p.TransitiveReflect != "warn" {
		t.Errorf("nil config severities = %q %q %q, want deny deny warn", p.TransitiveExec, p.TransitiveNet, p.TransitiveReflect)
	}
}

func TestSnapshotHashFlagRoundTrip(t *testing.T) {
	setLogosMode(t)
	mains := []string{"example.com/m"}
	SnapshotLinkPolicy(testConfigFile(), mains, nil)
	var h bytes.Buffer
	HashLinkPolicy(&h)
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
	back, err := UnmarshalPolicy(data)
	if err != nil {
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
	HashLinkPolicy(&h)
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
	back, err := UnmarshalPolicy(data)
	if err != nil {
		t.Fatal(err)
	}
	if !slices.Contains(back.CgoAllowed, path) {
		t.Errorf("CgoAllowed = %q, missing %q", back.CgoAllowed, path)
	}
}
