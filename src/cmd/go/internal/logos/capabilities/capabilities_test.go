// Copyright 2026 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

package capabilities

import (
	"strings"
	"testing"
)

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

func TestCheck(t *testing.T) {
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
		got := Check(Deny, nil, nodes(
			Node{ImportPath: "example.com/long", Imports: []string{"example.com/main/cmd"}, Main: true},
		))
		if len(got) != 1 {
			t.Fatalf("Check = %v, want one violation", got)
		}
		v := got[0]
		if v.Dep != "example.com/dep" {
			t.Errorf("Dep = %q, want example.com/dep", v.Dep)
		}
		if v.Rule != RuleUnsafe {
			t.Errorf("Rule = %q, want %q", v.Rule, RuleUnsafe)
		}
		want := []string{"example.com/main/cmd", "example.com/dep", "unsafe"}
		if strings.Join(v.Chain, "\x00") != strings.Join(want, "\x00") {
			t.Errorf("Chain = %q, want %q", v.Chain, want)
		}
	})

	t.Run("diamond reports shortest chain", func(t *testing.T) {
		all := []Node{
			{ImportPath: "example.com/main", Imports: []string{"example.com/near", "example.com/far"}, Main: true},
			{ImportPath: "example.com/near", Imports: []string{"example.com/dep"}},
			{ImportPath: "example.com/far", Imports: []string{"example.com/near"}},
			{ImportPath: "example.com/dep", Imports: []string{"unsafe"}},
			{ImportPath: "unsafe", Goroot: true},
		}
		got := Check(Deny, nil, all)
		if len(got) != 1 {
			t.Fatalf("Check = %v, want one violation", got)
		}
		want := []string{"example.com/main", "example.com/near", "example.com/dep", "unsafe"}
		if strings.Join(got[0].Chain, "\x00") != strings.Join(want, "\x00") {
			t.Errorf("Chain = %q, want %q", got[0].Chain, want)
		}
	})
	t.Run("allow skips walk", func(t *testing.T) {
		if got := Check(Allow, nil, nodes()); len(got) != 0 {
			t.Errorf("Check(allow) = %v, want none", got)
		}
	})
	t.Run("allowlisted dep passes", func(t *testing.T) {
		got := Check(Deny, []string{"example.com/dep/**"}, nodes())
		if len(got) != 0 {
			t.Errorf("Check(allowlisted) = %v, want none", got)
		}
	})
	t.Run("main and stdlib never blamed", func(t *testing.T) {
		all := nodes(
			Node{ImportPath: "example.com/main/self", Imports: []string{"unsafe"}, Main: true},
		)
		got := Check(Deny, nil, all)
		for _, v := range got {
			if v.Dep == "example.com/main/self" || v.Dep == "unsafe" {
				t.Errorf("blamed exempt package %q", v.Dep)
			}
		}
		if len(got) != 1 || got[0].Dep != "example.com/dep" {
			t.Errorf("Check = %v, want only the dep violation", got)
		}
	})
	t.Run("unreachable unsafe user ignored", func(t *testing.T) {
		all := nodes(
			Node{ImportPath: "example.com/stray", Imports: []string{"unsafe"}},
		)
		got := Check(Deny, nil, all)
		if len(got) != 1 || got[0].Dep != "example.com/dep" {
			t.Errorf("Check = %v, want only the reachable dep", got)
		}
	})
}

func TestCheckAssembly(t *testing.T) {
	graph := func(asm Node) []Node {
		return []Node{
			{ImportPath: "example.com/main", Imports: []string{"example.com/mid", "bytes"}, Main: true},
			{ImportPath: "example.com/mid", Imports: []string{"example.com/asm"}},
			asm,
			{ImportPath: "bytes", Goroot: true, Asm: true},
		}
	}
	dep := Node{ImportPath: "example.com/asm", Asm: true}
	t.Run("assembly dep reported with import chain", func(t *testing.T) {
		got := Check(Deny, nil, graph(dep))
		if len(got) != 1 {
			t.Fatalf("Check = %v, want one violation", got)
		}
		v := got[0]
		if v.Dep != "example.com/asm" || v.Rule != RuleAssembly {
			t.Errorf("violation = %+v, want assembly on example.com/asm", v)
		}
		want := []string{"example.com/main", "example.com/mid", "example.com/asm"}
		if strings.Join(v.Chain, "\x00") != strings.Join(want, "\x00") {
			t.Errorf("Chain = %q, want %q", v.Chain, want)
		}
	})
	t.Run("allowlisted assembly passes", func(t *testing.T) {
		if got := Check(Deny, []string{"example.com/asm/**"}, graph(dep)); len(got) != 0 {
			t.Errorf("Check(allowlisted) = %v, want none", got)
		}
	})
	t.Run("allow severity skips assembly", func(t *testing.T) {
		if got := Check(Allow, nil, graph(dep)); len(got) != 0 {
			t.Errorf("Check(allow) = %v, want none", got)
		}
	})
	t.Run("standard library and main assembly never blamed", func(t *testing.T) {
		got := Check(Deny, nil, graph(Node{ImportPath: "example.com/asm"}))
		if len(got) != 0 {
			t.Errorf("Check = %v, want none", got)
		}
	})
	t.Run("unsafe and assembly in one dependency report both rules", func(t *testing.T) {
		both := Node{ImportPath: "example.com/asm", Asm: true, Imports: []string{"unsafe"}}
		all := append(graph(both), Node{ImportPath: "unsafe", Goroot: true})
		got := Check(Deny, nil, all)
		if len(got) != 2 || got[0].Rule == got[1].Rule {
			t.Errorf("Check = %v, want one unsafe and one assembly violation", got)
		}
	})
}
