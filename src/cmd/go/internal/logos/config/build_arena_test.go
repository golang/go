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
		name string
		src  string
		want bool
	}{
		{"absent", "[build]\ncgo = \"forbid\"\n", false},
		{"noBuildTable", "", false},
		{"true", "[build]\narena = true\n", true},
		{"false", "[build]\narena = false\n", false},
		{"commentAfter", "[build]\narena = true # arena on\n", true},
	}
	for _, tt := range tests {
		f, err := Parse("logos.toml", []byte(tt.src))
		if err != nil {
			t.Errorf("%s: Parse error: %v", tt.name, err)
			continue
		}
		if f.Build.Arena != tt.want {
			t.Errorf("%s: Build.Arena = %v, want %v", tt.name, f.Build.Arena, tt.want)
		}
	}
}

func TestBuildArenaKeyErrors(t *testing.T) {
	tests := []struct {
		name string
		src  string
		want string
	}{
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
	f.Build.Arena = true
	out := f.Format()
	if !strings.Contains(string(out), "arena = true\n") {
		t.Fatalf("Format dropped arena = true:\n%s", out)
	}
	g, err := Parse("logos.toml", out)
	if err != nil {
		t.Fatalf("Parse of Format output: %v", err)
	}
	if !g.Build.Arena {
		t.Error("Build.Arena lost in the Format round trip")
	}
}
