// Copyright 2026 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

package config

import (
	"strings"
	"testing"
)

func TestBuildArenaKey(t *testing.T) {
	tests := []struct {
		name    string
		src     string
		wantNil bool
		want    bool
	}{
		{"absent", "[build]\ncgo = \"forbid\"\n", true, true},
		{"noBuildTable", "", true, true},
		{"true", "[build]\narena = true\n", false, true},
		{"false", "[build]\narena = false\n", false, false},
		{"commentAfter", "[build]\narena = true # arena on\n", false, true},
	}
	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			f, err := Parse("logos.toml", []byte(tt.src))
			if err != nil {
				t.Fatalf("Parse error: %v", err)
			}
			if (f.Build.Arena == nil) != tt.wantNil {
				t.Fatalf("Build.Arena nil = %v, want nil %v", f.Build.Arena == nil, tt.wantNil)
			}
			eff := f.Build.Arena == nil || *f.Build.Arena
			if eff != tt.want {
				t.Errorf("effective Build.Arena = %v, want %v", eff, tt.want)
			}
			out := f.Format()
			g, err := Parse("logos.toml", out)
			if err != nil {
				t.Fatalf("Parse of Format output: %v", err)
			}
			if (g.Build.Arena == nil) != tt.wantNil {
				t.Fatalf("round trip Build.Arena nil = %v, want nil %v", g.Build.Arena == nil, tt.wantNil)
			}
			eff2 := g.Build.Arena == nil || *g.Build.Arena
			if eff2 != tt.want {
				t.Errorf("round trip effective Build.Arena = %v, want %v", eff2, tt.want)
			}
		})
	}
}

func TestBuildArenaKeyErrors(t *testing.T) {
	tests := []struct {
		name string
		src  string
		want string
	}{
		{"bareYes", "[build]\narena = yes\n", "logos.toml:2:9: want boolean for key \"arena\""},
		{"string", "[build]\narena = \"true\"\n", `want boolean for key "arena"`},
		{"literalString", "[build]\narena = 'true'\n", `want boolean for key "arena"`},
		{"integer", "[build]\narena = 1\n", `want boolean for key "arena"`},
		{"array", "[build]\narena = [true]\n", `want boolean for key "arena"`},
		{"missing", "[build]\narena =\n", `expected value for key "arena"`},
		{"inlineTable", "[build]\narena = {}\n", "inline table not allowed"},
		{"wrongTable", "[project]\narena = true\n", `unknown key "arena"`},
		{"duplicate", "[build]\narena = true\narena = false\n", `duplicate key "arena"`},
		{"unterminatedString", "[build]\narena = \"tru\n", "unterminated string"},
	}
	for _, tt := range tests {
		_, err := Parse("logos.toml", []byte(tt.src))
		if err == nil {
			t.Errorf("%s: Parse accepted %q", tt.name, tt.src)
			continue
		}
		if !strings.Contains(err.Error(), tt.want) {
			t.Errorf("%s: error %q does not contain %q", tt.name, err.Error(), tt.want)
		}
	}
}

func TestBuildArenaFormatRoundTrip(t *testing.T) {
	f := defaultFile()
	if got := string(f.Format()); strings.Contains(got, "arena") {
		t.Errorf("Format of the default file mentions arena:\n%s", got)
	}
	on := true
	f.Build.Arena = &on
	out := f.Format()
	if !strings.Contains(string(out), "arena = true\n") {
		t.Fatalf("Format dropped arena = true:\n%s", out)
	}
	g, err := Parse("logos.toml", out)
	if err != nil {
		t.Fatalf("Parse of Format output: %v", err)
	}
	if g.Build.Arena == nil || !*g.Build.Arena {
		t.Error("Build.Arena lost in the Format round trip")
	}
	off := false
	f.Build.Arena = &off
	out = f.Format()
	if !strings.Contains(string(out), "arena = false\n") {
		t.Fatalf("Format dropped arena = false:\n%s", out)
	}
	h, err := Parse("logos.toml", out)
	if err != nil {
		t.Fatalf("Parse of Format output: %v", err)
	}
	if h.Build.Arena == nil || *h.Build.Arena {
		t.Error("Build.Arena false lost in the Format round trip")
	}
}
