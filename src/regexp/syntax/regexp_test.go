// Copyright 2026 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

package syntax

import (
	"testing"
)

func TestRunesInFoldableRange(t *testing.T) {
	for _, tt := range []struct {
		regexp string
		want   int
	}{
		{`[a-z]`, 26},
		{`[a-bd-eg-h]`, 6},
		{`[\x00-A]`, 1},              // end at minFold
		{`[\x{1e943}-\x{1e949}]`, 1}, // start at maxFold
	} {
		re, err := Parse(tt.regexp, MatchNL|Perl&^OneLine)
		if err != nil {
			t.Errorf("Parse(%#q) = error %v", tt.regexp, err)
			continue
		}
		if got := runesInFoldableRange(re); got != tt.want {
			t.Errorf("runesInFoldableRange(%#q) = %d, want %d", tt.regexp, got, tt.want)
		}
	}
}

func TestIncludesMostFoldableRunes(t *testing.T) {
	for _, tt := range []struct {
		regexp string
		want   bool
	}{
		{``, false},
		{`[A-z]`, false},
		{`[A-\x{10000}]`, true},
		{`\W`, true},
		{`[\x{1e900}-\x{3e949}]`, false}, // large, but mostly non-foldable
	} {
		re, err := Parse(tt.regexp, MatchNL|Perl&^OneLine)
		if err != nil {
			t.Errorf("Parse(%#q) = error %v", tt.regexp, err)
			continue
		}
		if got := includesMostFoldableRunes(re); got != tt.want {
			t.Errorf("includesMostFoldableRunes(%#q) = %t, want %t", tt.regexp, got, tt.want)
		}
	}
}

func TestCharClassIsFoldSensitive(t *testing.T) {
	for _, tt := range []struct {
		regexp string
		want   bool
	}{
		{`\d`, false},   // no foldables -> safe to use ?i
		{`[A-B]`, true}, // doesn't have lowercase -> unsafe
		{`[kK]`, true},  // doesn't have Kelvin rune
		{`[A-Ba-b]`, false},
		{`[^Aa]`, false},
		{`\W`, true},             // some non-word runes fold to A-Za-z
		{`[A-\x{1e943}]`, false}, // entire foldable range
		{`[B-\x{1e943}]`, true},  // missing first foldable rune
		{`[A-\x{1e942}]`, true},  // missing last foldable rune
		{`\D`, false},            // excluded runes (digits) don't fold
	} {
		re, err := Parse(tt.regexp, MatchNL|Perl&^OneLine)
		if err != nil {
			t.Errorf("Parse(%#q) = error %v", tt.regexp, err)
			continue
		}
		if got := charClassIsFoldSensitive(re); got != tt.want {
			t.Errorf("foldSensitiveCharClass(%#q) = %t, want %t", tt.regexp, got, tt.want)
		}
	}
}

func BenchmarkIssue69303(b *testing.B) {
	for _, tt := range []struct {
		name string
		re   string
	}{
		{name: "negated", re: `.[^.]\w\W`},
		{name: "first-rune-sensitive", re: `[\x{13a0}\x{b000}-\x{10ffff}]`},
	} {
		b.Run(tt.name, func(b *testing.B) {
			re, err := Parse(tt.re, MatchNL|Perl&^OneLine)
			if err != nil {
				b.Fatal(err)
			}
			for b.Loop() {
				_ = re.String()
			}
		})
	}
}
