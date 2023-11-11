// Copyright 2023 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

package http

import (
	"fmt"
	"slices"
	"sort"
	"strings"
	"testing"
)

func TestIndex(t *testing.T) {
	// Generate every kind of pattern up to some number of segments,
	// and compare conflicts found during indexing with those found
	// by exhaustive comparison.
	patterns := generatePatterns()
	var idx routingIndex
	for i, pat := range patterns {
		got := indexConflicts(pat, &idx)
		want := trueConflicts(pat, patterns[:i])
		if !slices.Equal(got, want) {
			t.Fatalf("%q:\ngot  %q\nwant %q", pat, got, want)
		}
		idx.addPattern(pat)
	}
}

func trueConflicts(pat *pattern, pats []*pattern) []string {
	var s []string
	for _, p := range pats {
		if pat.conflictsWith(p) {
			s = append(s, p.String())
		}
	}
	sort.Strings(s)
	return s
}

func indexConflicts(pat *pattern, idx *routingIndex) []string {
	var s []string
	idx.possiblyConflictingPatterns(pat, func(p *pattern) error {
		if pat.conflictsWith(p) {
			s = append(s, p.String())
		}
		return nil
	})
	sort.Strings(s)
	return slices.Compact(s)
}

// generatePatterns generates all possible patterns using a representative
// sample of parts.
func generatePatterns() []*pattern {
	var pats []*pattern

	collect := func(s string) {
		// Replace duplicate wildcards with unique ones.
		var b strings.Builder
		wc := 0
		for {
			i := strings.Index(s, "{x}")
			if i < 0 {
				b.WriteString(s)
				break
			}
			b.WriteString(s[:i])
			fmt.Fprintf(&b, "{x%d}", wc)
			wc++
			s = s[i+3:]
		}
		pat, err := parsePattern(b.String())
		if err != nil {
			panic(err)
		}
		pats = append(pats, pat)
	}

	var (
		methods   = []string{"", "GET ", "HEAD ", "POST "}
		hosts     = []string{"", "h1", "h2"}
		segs      = []string{"/a", "/b", "/{x}"}
		finalSegs = []string{"/a", "/b", "/{f}", "/{m...}", "/{$}"}
	)

	g := genConcat(
		genChoice(methods),
		genChoice(hosts),
		genStar(3, genChoice(segs)),
		genChoice(finalSegs))
	g(collect)
	return pats
}

// A generator is a function that calls its argument with the strings that it
// generates.
type generator func(collect func(string))

// genConst generates a single constant string.
func genConst(s string) generator {
	return func(collect func(string)) {
		collect(s)
	}
}

// genChoice generates all the strings in its argument.
func genChoice(choices []string) generator {
	return func(collect func(string)) {
		for _, c := range choices {
			collect(c)
		}
	}
}

// genConcat2 generates the cross product of the strings of g1 concatenated
// with those of g2.
func genConcat2(g1, g2 generator) generator {
	return func(collect func(string)) {
		g1(func(s1 string) {
			g2(func(s2 string) {
				collect(s1 + s2)
			})
		})
	}
}

// genConcat generalizes genConcat2 to any number of generators.
func genConcat(gs ...generator) generator {
	if len(gs) == 0 {
		return genConst("")
	}
	return genConcat2(gs[0], genConcat(gs[1:]...))
}

// genRepeat generates strings of exactly n copies of g's strings.
func genRepeat(n int, g generator) generator {
	if n == 0 {
		return genConst("")
	}
	return genConcat(g, genRepeat(n-1, g))
}

// genStar (named after the Kleene star) generates 0, 1, 2, ..., max
// copies of the strings of g.
func genStar(max int, g generator) generator {
	return func(collect func(string)) {
		for i := 0; i <= max; i++ {
			genRepeat(i, g)(collect)
		}
	}
}

func BenchmarkMultiConflicts(b *testing.B) {
	// How fast is indexing if the corpus is all multis?
	const nMultis = 1000
	var pats []*pattern
	for i := 0; i < nMultis; i++ {
		pats = append(pats, mustParsePattern(b, fmt.Sprintf("/a/b/{x}/d%d/", i)))
	}
	b.ResetTimer()
	for i := 0; i < b.N; i++ {
		var idx routingIndex
		for _, p := range pats {
			got := indexConflicts(p, &idx)
			if len(got) != 0 {
				b.Fatalf("got %d conflicts, want 0", len(got))
			}
			idx.addPattern(p)
		}
		if i == 0 {
			// Confirm that all the multis ended up where they belong.
			if g, w := len(idx.multis), nMultis; g != w {
				b.Fatalf("got %d multis, want %d", g, w)
			}
		}
	}
}
