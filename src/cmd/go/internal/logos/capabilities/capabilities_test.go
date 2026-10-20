// Copyright 2026 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

package capabilities

import (
	"strings"
	"testing"
)

func TestParseSeverity(t *testing.T) {
	for _, s := range []string{"deny", "warn", "allow"} {
		got, err := ParseSeverity(s)
		if err != nil {
			t.Errorf("ParseSeverity(%q) error = %v", s, err)
		}
		if got != Severity(s) {
			t.Errorf("ParseSeverity(%q) = %q, want %q", s, got, s)
		}
	}
	if _, err := ParseSeverity("forbid"); err == nil {
		t.Error("ParseSeverity(forbid) = nil error, want error")
	}
}

func TestAllowlisted(t *testing.T) {
	tests := []struct {
		name       string
		allowlist  []string
		path       string
		wantListed bool
	}{
		{"empty allowlist", nil, "example.com/dep", false},
		{"suffix glob matches prefix itself", []string{"example.com/dep/**"}, "example.com/dep", true},
		{"suffix glob matches child", []string{"example.com/dep/**"}, "example.com/dep/sub", true},
		{"suffix glob matches deep child", []string{"example.com/dep/**"}, "example.com/dep/a/b", true},
		{"suffix glob rejects sibling", []string{"example.com/dep/**"}, "example.com/dep2", false},
		{"suffix glob rejects parent", []string{"example.com/dep/sub/**"}, "example.com/dep", false},
		{"star does not cross slash", []string{"example.com/*"}, "example.com/a/b", false},
		{"star matches one segment", []string{"example.com/*"}, "example.com/a", true},
		{"middle doublestar spans segments", []string{"example.com/**/tail"}, "example.com/a/b/tail", true},
		{"middle doublestar spans zero", []string{"example.com/**/tail"}, "example.com/tail", true},
		{"middle doublestar rejects other tail", []string{"example.com/**/tail"}, "example.com/a/head", false},
		{"exact match", []string{"example.com/dep"}, "example.com/dep", true},
		{"exact rejects child", []string{"example.com/dep"}, "example.com/dep/sub", false},
	}
	for _, tc := range tests {
		t.Run(tc.name, func(t *testing.T) {
			if got := Allowlisted(tc.allowlist, tc.path); got != tc.wantListed {
				t.Errorf("Allowlisted(%q, %q) = %v, want %v", tc.allowlist, tc.path, got, tc.wantListed)
			}
		})
	}
}

func TestCheckUnsafe(t *testing.T) {
	nodes := func(extra ...Node) []Node {
		base := []Node{
			{ImportPath: "example.com/main/cmd", Imports: []string{"example.com/dep", "fmt"}, Main: true},
			{ImportPath: "example.com/dep", Imports: []string{"unsafe", "strconv"}},
			{ImportPath: "fmt", Goroot: true},
			{ImportPath: "strconv", Goroot: true},
			{ImportPath: "unsafe", Goroot: true},
		}
		return append(base, extra...)
	}
	t.Run("deny reports shortest chain", func(t *testing.T) {
		got := CheckUnsafe(Deny, nil, nodes(
			Node{ImportPath: "example.com/long", Imports: []string{"example.com/main/cmd"}, Main: true},
		))
		if len(got) != 1 {
			t.Fatalf("CheckUnsafe = %v, want one violation", got)
		}
		v := got[0]
		if v.Dep != "example.com/dep" {
			t.Errorf("Dep = %q, want example.com/dep", v.Dep)
		}
		want := []string{"example.com/main/cmd", "example.com/dep", "unsafe"}
		if strings.Join(v.Chain, "\x00") != strings.Join(want, "\x00") {
			t.Errorf("Chain = %q, want %q", v.Chain, want)
		}
	})
	t.Run("allow skips walk", func(t *testing.T) {
		if got := CheckUnsafe(Allow, nil, nodes()); len(got) != 0 {
			t.Errorf("CheckUnsafe(allow) = %v, want none", got)
		}
	})
	t.Run("allowlisted dep passes", func(t *testing.T) {
		got := CheckUnsafe(Deny, []string{"example.com/dep/**"}, nodes())
		if len(got) != 0 {
			t.Errorf("CheckUnsafe(allowlisted) = %v, want none", got)
		}
	})
	t.Run("main and stdlib never blamed", func(t *testing.T) {
		all := nodes(
			Node{ImportPath: "example.com/main/self", Imports: []string{"unsafe"}, Main: true},
		)
		got := CheckUnsafe(Deny, nil, all)
		for _, v := range got {
			if v.Dep == "example.com/main/self" || v.Dep == "unsafe" {
				t.Errorf("blamed exempt package %q", v.Dep)
			}
		}
		if len(got) != 1 || got[0].Dep != "example.com/dep" {
			t.Errorf("CheckUnsafe = %v, want only the dep violation", got)
		}
	})
	t.Run("unreachable unsafe user ignored", func(t *testing.T) {
		all := nodes(
			Node{ImportPath: "example.com/stray", Imports: []string{"unsafe"}},
		)
		got := CheckUnsafe(Deny, nil, all)
		if len(got) != 1 || got[0].Dep != "example.com/dep" {
			t.Errorf("CheckUnsafe = %v, want only the reachable dep", got)
		}
	})
}

func TestFormatViolation(t *testing.T) {
	v := Violation{Dep: "example.com/dep", Chain: []string{"example.com/main/cmd", "example.com/dep", "unsafe"}}
	got := FormatViolation(v)
	for _, want := range []string{
		"ERROR [capabilities::unauthorized_unsafe]: Transitive dependency 'example.com/dep' imports 'unsafe'",
		"1. example.com/main/cmd",
		"2.   -> example.com/dep",
		"3.   -> unsafe",
		`authorize "example.com/dep/**" under 'transitive_unsafe_allowlist' in logos.toml.`,
	} {
		if !strings.Contains(got, want) {
			t.Errorf("FormatViolation missing %q:\n%s", want, got)
		}
	}
}
