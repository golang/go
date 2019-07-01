// Copyright 2019 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

// Benchmark results:
//
// BenchmarkMatcher-12    	 1000000	      1615 ns/op	  30.95 MB/s	       0 B/op	       0 allocs/op
//
package fuzzy_test

import (
	"bytes"
	"fmt"
	"math"
	"testing"

	"golang.org/x/tools/internal/lsp/fuzzy"
)

func ExampleFuzzyMatcher() {
	pattern := "TEdit"
	candidates := []string{"fuzzy.TextEdit", "ArtEdit", "TED talks about IT"}

	// Create a fuzzy matcher for the pattern.
	matcher := fuzzy.NewMatcher(pattern, fuzzy.Text)

	for _, candidate := range candidates {
		// Compute candidate's score against the matcher.
		score := matcher.Score(candidate)

		if score > -1 {
			// Get the substrings in the candidate matching the pattern.
			ranges := matcher.MatchedRanges()

			fmt.Println(ranges) // Do something with the ranges.
		}
	}
}

type comparator struct {
	f     func(val, ref float32) bool
	descr string
}

var (
	eq = comparator{
		f: func(val, ref float32) bool {
			return val == ref
		},
		descr: "==",
	}
	ge = comparator{
		f: func(val, ref float32) bool {
			return val >= ref
		},
		descr: ">=",
	}
)

func (c comparator) eval(val, ref float32) bool {
	return c.f(val, ref)
}

func (c comparator) String() string {
	return c.descr
}

type scoreTest struct {
	candidate string
	comparator
	ref float32
}

var matcherTests = []struct {
	pattern string
	input   fuzzy.Input
	tests   []scoreTest
}{
	{
		pattern: "",
		input:   fuzzy.Text,
		tests: []scoreTest{
			{"def", eq, 1},
			{"Ab stuff c", eq, 1},
		},
	},
	{
		pattern: "abc",
		input:   fuzzy.Text,
		tests: []scoreTest{
			{"def", eq, -1},
			{"abd", eq, -1},
			{"abc", ge, 0},
			{"Abc", ge, 0},
			{"Ab stuff c", ge, 0},
		},
	},
	{
		pattern: "Abc",
		input:   fuzzy.Text,
		tests: []scoreTest{
			{"def", eq, -1},
			{"abd", eq, -1},
			{"abc", ge, 0},
			{"Abc", ge, 0},
			{"Ab stuff c", ge, 0},
		},
	},
	{
		pattern: "subs",
		input:   fuzzy.Filename,
		tests: []scoreTest{
			{"sub/seq", ge, 0},
			{"sub/seq/end", eq, -1},
			{"sub/seq/base", ge, 0},
		},
	},
	{
		pattern: "subs",
		input:   fuzzy.Filename,
		tests: []scoreTest{
			{"//sub/seq", ge, 0},
			{"//sub/seq/end", eq, -1},
			{"//sub/seq/base", ge, 0},
		},
	},
}

func TestScore(t *testing.T) {
	for _, tc := range matcherTests {
		m := fuzzy.NewMatcher(tc.pattern, tc.input)
		for _, sct := range tc.tests {
			score := m.Score(sct.candidate)
			if !sct.comparator.eval(score, sct.ref) {
				t.Errorf("not true that m.Score(%s)[=%v] %s %v", sct.candidate, score, sct.comparator, sct.ref)
			}
		}
	}
}

type candidateCompTest struct {
	c1         string
	comparator comparator
	c2         string
}

var compareCandidatesTestCases = []struct {
	pattern           string
	input             fuzzy.Input
	orderedCandidates []string
}{
	{
		pattern: "aa",
		input:   fuzzy.Filename,
		orderedCandidates: []string{
			"baab",
			"bb_aa",
			"a/a/a",
			"aa_bb",
			"aa_b",
			"aabb",
			"aab",
			"b/aa",
		},
	},
	{
		pattern: "Foo",
		input:   fuzzy.Text,
		orderedCandidates: []string{
			"Barfoo",
			"F_o_o",
			"Faoo",
			"F__oo",
			"F_oo",
			"FaoFooa",
			"BarFoo",
			"FooA",
			"FooBar",
			"Foo",
		},
	},
}

func TestCompareCandidateScores(t *testing.T) {
	for _, tc := range compareCandidatesTestCases {
		m := fuzzy.NewMatcher(tc.pattern, tc.input)

		var prevScore float32
		prevCand := "MIN_SCORE"
		for _, cand := range tc.orderedCandidates {
			score := m.Score(cand)
			if prevScore > score {
				t.Errorf("%s[=%v] is scored lower than %s[=%v]", cand, score, prevCand, prevScore)
			}
			if score < -1 || score > 1 {
				t.Errorf("%s score is %v; want value between [-1, 1]", cand, score)
			}
			prevScore = score
			prevCand = cand
		}
	}
}

var fuzzyMatcherTestCases = []struct {
	p     string
	str   string
	want  string
	input fuzzy.Input
}{
	// fuzzy.Filename
	{p: "aa", str: "a_a/a_a", want: "[a]_a/[a]_a", input: fuzzy.Filename},
	{p: "aaaa", str: "a_a/a_a", want: "[a]_[a]/[a]_[a]", input: fuzzy.Filename},
	{p: "aaaa", str: "aaaa", want: "[aaaa]", input: fuzzy.Filename},
	{p: "aaaa", str: "a_a/a_aaaa", want: "a_a/[a]_[aaa]a", input: fuzzy.Filename},
	{p: "aaaa", str: "a_a/aaaaa", want: "a_a/[aaaa]a", input: fuzzy.Filename},
	{p: "aaaa", str: "aabaaa", want: "[aa]b[aa]a", input: fuzzy.Filename},
	{p: "aaaa", str: "a/baaa", want: "[a]/b[aaa]", input: fuzzy.Filename},
	{p: "abcxz", str: "d/abc/abcd/oxz", want: "d/[abc]/abcd/o[xz]", input: fuzzy.Filename},
	{p: "abcxz", str: "d/abcd/abc/oxz", want: "d/[abc]d/abc/o[xz]", input: fuzzy.Filename},

	// fuzzy.Symbol
	{p: "foo", str: "abc::foo", want: "abc::[foo]", input: fuzzy.Symbol},
	{p: "foo", str: "foo.foo", want: "foo.[foo]", input: fuzzy.Symbol},
	{p: "foo", str: "fo_oo.o_oo", want: "[fo]_oo.[o]_oo", input: fuzzy.Symbol},
	{p: "foo", str: "fo_oo.fo_oo", want: "fo_oo.[fo]_[o]o", input: fuzzy.Symbol},
	{p: "fo_o", str: "fo_oo.o_oo", want: "[f]o_oo.[o_o]o", input: fuzzy.Symbol},
	{p: "fOO", str: "fo_oo.o_oo", want: "[f]o_oo.[o]_[o]o", input: fuzzy.Symbol},
	{p: "tedit", str: "foo.TextEdit", want: "foo.[T]ext[Edit]", input: fuzzy.Symbol},
	{p: "TEdit", str: "foo.TextEdit", want: "foo.[T]ext[Edit]", input: fuzzy.Symbol},
	{p: "Tedit", str: "foo.TextEdit", want: "foo.[T]ext[Edit]", input: fuzzy.Symbol},
	{p: "Tedit", str: "foo.Textedit", want: "foo.[Te]xte[dit]", input: fuzzy.Symbol},
	{p: "TEdit", str: "foo.Textedit", want: "", input: fuzzy.Symbol},
	{p: "te", str: "foo.Textedit", want: "foo.[Te]xtedit", input: fuzzy.Symbol},
	{p: "ee", str: "foo.Textedit", want: "", input: fuzzy.Symbol}, // short middle of the word match
	{p: "ex", str: "foo.Textedit", want: "foo.T[ex]tedit", input: fuzzy.Symbol},
	{p: "exdi", str: "foo.Textedit", want: "", input: fuzzy.Symbol},  // short middle of the word match
	{p: "exdit", str: "foo.Textedit", want: "", input: fuzzy.Symbol}, // short middle of the word match
	{p: "extdit", str: "foo.Textedit", want: "foo.T[ext]e[dit]", input: fuzzy.Symbol},
	{p: "e", str: "foo.Textedit", want: "foo.T[e]xtedit", input: fuzzy.Symbol},
	{p: "E", str: "foo.Textedit", want: "foo.T[e]xtedit", input: fuzzy.Symbol},
	{p: "ed", str: "foo.Textedit", want: "foo.Text[ed]it", input: fuzzy.Symbol},
	{p: "edt", str: "foo.Textedit", want: "", input: fuzzy.Symbol}, // short middle of the word match
	{p: "edit", str: "foo.Textedit", want: "foo.Text[edit]", input: fuzzy.Symbol},
	{p: "edin", str: "foo.TexteditNum", want: "foo.Text[edi]t[N]um", input: fuzzy.Symbol},
	{p: "n", str: "node.GoNodeMax", want: "node.Go[N]odeMax", input: fuzzy.Symbol},
	{p: "N", str: "node.GoNodeMax", want: "node.Go[N]odeMax", input: fuzzy.Symbol},
	{p: "completio", str: "completion", want: "[completio]n", input: fuzzy.Symbol},
	{p: "completio", str: "completion.None", want: "[completi]on.N[o]ne", input: fuzzy.Symbol},
}

func TestFuzzyMatcherRanges(t *testing.T) {
	for _, tc := range fuzzyMatcherTestCases {
		matcher := fuzzy.NewMatcher(tc.p, tc.input)
		score := matcher.Score(tc.str)
		if tc.want == "" {
			if score >= 0 {
				t.Errorf("Score(%s, %s) = %v; want: <= 0", tc.p, tc.str, score)
			}
			continue
		}
		if score < 0 {
			t.Errorf("Score(%s, %s) = %v, want: > 0", tc.p, tc.str, score)
			continue
		}
		got := highlightMatches(tc.str, matcher)
		if tc.want != got {
			t.Errorf("highlightMatches(%s, %s) = %v, want: %v", tc.p, tc.str, got, tc.want)
		}
	}
}

var scoreTestCases = []struct {
	p    string
	str  string
	want float64
}{
	// Score precision up to five digits. Modify if changing the score, but make sure the new values
	// are reasonable.
	{p: "abc", str: "abc", want: 1},
	{p: "abc", str: "Abc", want: 1},
	{p: "abc", str: "Abcdef", want: 1},
	{p: "strc", str: "StrCat", want: 1},
	{p: "abc_def", str: "abc_def_xyz", want: 1},
	{p: "abcdef", str: "abc_def_xyz", want: 0.91667},
	{p: "abcxyz", str: "abc_def_xyz", want: 0.875},
	{p: "sc", str: "StrCat", want: 0.75},
	{p: "abc", str: "AbstrBasicCtor", want: 0.75},
	{p: "foo", str: "abc::foo", want: 1},
	{p: "afoo", str: "abc::foo", want: 0.9375},
	{p: "abr", str: "abc::bar", want: 0.5},
	{p: "br", str: "abc::bar", want: 0.375},
	{p: "aar", str: "abc::bar", want: 0.16667},
	{p: "edin", str: "foo.TexteditNum", want: 0},
	{p: "ediu", str: "foo.TexteditNum", want: 0},
	// We want the next two items to have roughly similar scores.
	{p: "up", str: "unique_ptr", want: 0.75},
	{p: "up", str: "upper_bound", want: 1},
}

func TestScores(t *testing.T) {
	for _, tc := range scoreTestCases {
		matcher := fuzzy.NewMatcher(tc.p, fuzzy.Symbol)
		got := math.Round(float64(matcher.Score(tc.str))*1e5) / 1e5
		if got != tc.want {
			t.Errorf("Score(%s, %s) = %v, want: %v", tc.p, tc.str, got, tc.want)
		}
	}
}

func highlightMatches(str string, matcher *fuzzy.Matcher) string {
	matches := matcher.MatchedRanges()

	var buf bytes.Buffer
	index := 0
	for i := 0; i < len(matches)-1; i += 2 {
		s, e := matches[i], matches[i+1]
		fmt.Fprintf(&buf, "%s[%s]", str[index:s], str[s:e])
		index = e
	}
	buf.WriteString(str[index:])
	return buf.String()
}

func BenchmarkMatcher(b *testing.B) {
	pattern := "Foo"
	candidates := []string{
		"F_o_o",
		"Barfoo",
		"Faoo",
		"F__oo",
		"F_oo",
		"FaoFooa",
		"BarFoo",
		"FooA",
		"FooBar",
		"Foo",
	}

	matcher := fuzzy.NewMatcher(pattern, fuzzy.Text)

	b.ResetTimer()
	for i := 0; i < b.N; i++ {
		for _, c := range candidates {
			matcher.Score(c)
		}
	}
	var numBytes int
	for _, c := range candidates {
		numBytes += len(c)
	}
	b.SetBytes(int64(numBytes))
}
