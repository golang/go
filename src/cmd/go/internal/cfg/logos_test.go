// Copyright 2026 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

package cfg

import (
	"strings"
	"testing"

	"cmd/go/internal/logos/config"
)

func TestCgoAllowlistAllows(t *testing.T) {
	tests := []struct {
		name string
		list []string
		path string
		want bool
	}{
		{"exact", []string{"example.com/m/ok"}, "example.com/m/ok", true},
		{"exactMiss", []string{"example.com/m/ok"}, "example.com/m/other", false},
		{"starStarMatchesPrefix", []string{"example.com/m/ok/**"}, "example.com/m/ok", true},
		{"starStarMatchesBelow", []string{"example.com/m/ok/**"}, "example.com/m/ok/sub/deep", true},
		{"starStarSibling", []string{"example.com/m/ok/**"}, "example.com/m/okay", false},
		{"doubleStarSuffixAcrossSlash", []string{"example.com/a/o**"}, "example.com/a/okay/x", true},
		{"doubleStarSuffixSameDir", []string{"example.com/a/o**"}, "example.com/a/okay", true},
		{"doubleStarSuffixMiss", []string{"example.com/a/o**"}, "example.com/a/nope", false},
		{"bareDoubleStar", []string{"**"}, "example.com/a/b", true},
		{"starOneLevel", []string{"example.com/m/*"}, "example.com/m/a", true},
		{"starNoSlash", []string{"example.com/m/*"}, "example.com/m/a/b", false},
		{"question", []string{"example.com/m/o?"}, "example.com/m/ok", true},
	}
	for _, tt := range tests {
		if got := CgoAllowlistAllows(tt.list, tt.path); got != tt.want {
			t.Errorf("%s: CgoAllowlistAllows(%q, %q) = %v, want %v", tt.name, tt.list, tt.path, got, tt.want)
		}
	}
}

func TestParseRejectsInvalidCgoAllowlist(t *testing.T) {
	data := "[build]\ncgo = \"restricted\"\ncgo_allowlist = [\"[\"]\n"
	_, err := config.Parse("logos.toml", []byte(data))
	if err == nil {
		t.Fatal("Parse accepted invalid cgo_allowlist pattern \"[\", want [configuration::invalid_cgo_allowlist]")
	}
	const want = `[configuration::invalid_cgo_allowlist] cgo_allowlist pattern "[": syntax error in pattern`
	if !strings.Contains(err.Error(), want) {
		t.Errorf("Parse error = %q, want substring %q", err.Error(), want)
	}
}
