// Copyright 2021 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

package fuzzy_test

import (
	"go/ast"
	"go/token"
	"sort"
	"testing"

	"golang.org/x/tools/go/packages"
	. "golang.org/x/tools/internal/fuzzy"
)

func TestSymbolMatchIndex(t *testing.T) {
	tests := []struct {
		pattern, input string
		want           int
	}{
		{"test", "foo.TestFoo", 4},
		{"test", "test", 0},
		{"test", "Test", 0},
		{"test", "est", -1},
		{"t", "shortest", 7},
		{"", "foo", -1},
		{"", string([]rune{0}), -1}, // verify that we don't default to an empty pattern.
		{"anything", "", -1},
	}

	for _, test := range tests {
		matcher := NewSymbolMatcher(test.pattern)
		if got, _ := matcher.Match([]string{test.input}); got != test.want {
			t.Errorf("NewSymbolMatcher(%q).Match(%q) = %v, _, want %v, _", test.pattern, test.input, got, test.want)
		}
	}
}

func TestSymbolRanking(t *testing.T) {

	// query -> symbols to match, in ascending order of score
	queryRanks := map[string][]string{
		"test": {
			"this.is.better.than.most",
			"test.foo.bar",
			"thebest",
			"atest",
			"test.foo",
			"testage",
			"tTest",
			"foo.test",
		},
		"parseside": { // golang/go#60201
			"yaml_PARSE_FLOW_SEQUENCE_ENTRY_MAPPING_END_STATE",
			"parseContext.parse_sidebyside",
		},
		"cvb": {
			"filecache_test.testIPCValueB",
			"cover.Boundary",
		},
		"dho": {
			"gocommand.DebugHangingGoCommands",
			"protocol.DocumentHighlightOptions",
		},
		"flg": {
			"completion.FALLTHROUGH",
			"main.flagGoCmd",
		},
		"fvi": {
			"godoc.fileIndexVersion",
			"macho.FlagSubsectionsViaSymbols",
		},
	}

	for query, symbols := range queryRanks {
		t.Run(query, func(t *testing.T) {
			matcher := NewSymbolMatcher(query)
			prev := 0.0
			for _, sym := range symbols {
				_, score := matcher.Match([]string{sym})
				t.Logf("Match(%q) = %v", sym, score)
				if score <= prev {
					t.Errorf("Match(%q) = _, %v, want > %v", sym, score, prev)
				}
				prev = score
			}
		})
	}
}

func TestMatcherSimilarities(t *testing.T) {
	// This test compares the fuzzy matcher with the symbol matcher on a corpus
	// of qualified identifiers extracted from x/tools.
	//
	// These two matchers are not expected to agree, but inspecting differences
	// can be useful for finding interesting ranking edge cases.
	t.Skip("unskip this test to compare matchers")

	idents := collectIdentifiers(t)
	t.Logf("collected %d unique identifiers", len(idents))

	// TODO: use go1.21 slices.MaxFunc.
	topMatch := func(score func(string) float64) string {
		top := ""
		topScore := 0.0
		for _, cand := range idents {
			if s := score(cand); s > topScore {
				top = cand
				topScore = s
			}
		}
		return top
	}

	agreed := 0
	total := 0
	bad := 0
	patterns := generatePatterns()
	for _, pattern := range patterns {
		total++

		fm := NewMatcher(pattern)
		topFuzzy := topMatch(func(input string) float64 {
			return float64(fm.Score(input))
		})
		sm := NewSymbolMatcher(pattern)
		topSymbol := topMatch(func(input string) float64 {
			_, score := sm.Match([]string{input})
			return score
		})
		switch {
		case topFuzzy == "" && topSymbol != "":
			if false {
				// The fuzzy matcher has a bug where it misses some matches; for this
				// test we only care about the symbol matcher.
				t.Logf("%q matched %q but no fuzzy match", pattern, topSymbol)
			}
			total--
			bad++
		case topFuzzy != "" && topSymbol == "":
			t.Fatalf("%q matched %q but no symbol match", pattern, topFuzzy)
		case topFuzzy == topSymbol:
			agreed++
		default:
			// Enable this log to see mismatches.
			if false {
				t.Logf("mismatch for %q: fuzzy: %q, symbol: %q", pattern, topFuzzy, topSymbol)
			}
		}
	}
	t.Logf("fuzzy matchers agreed on %d out of %d queries (%d bad)", agreed, total, bad)
}

func collectIdentifiers(tb testing.TB) []string {
	cfg := &packages.Config{
		Mode:  packages.NeedName | packages.NeedSyntax | packages.NeedFiles,
		Tests: true,
	}
	pkgs, err := packages.Load(cfg, "golang.org/x/tools/...")
	if err != nil {
		tb.Fatal(err)
	}
	uniqueIdents := make(map[string]bool)
	decls := 0
	for _, pkg := range pkgs {
		for _, f := range pkg.Syntax {
			for _, decl := range f.Decls {
				decls++
				switch decl := decl.(type) {
				case *ast.GenDecl:
					for _, spec := range decl.Specs {
						switch decl.Tok {
						case token.IMPORT:
						case token.TYPE:
							name := spec.(*ast.TypeSpec).Name.Name
							qualified := pkg.Name + "." + name
							uniqueIdents[qualified] = true
						case token.CONST, token.VAR:
							for _, n := range spec.(*ast.ValueSpec).Names {
								qualified := pkg.Name + "." + n.Name
								uniqueIdents[qualified] = true
							}
						}
					}
				}
			}
		}
	}
	var idents []string
	for k := range uniqueIdents {
		idents = append(idents, k)
	}
	sort.Strings(idents)
	return idents
}

func generatePatterns() []string {
	var patterns []string
	for x := 'a'; x <= 'z'; x++ {
		for y := 'a'; y <= 'z'; y++ {
			for z := 'a'; z <= 'z'; z++ {
				patterns = append(patterns, string(x)+string(y)+string(z))
			}
		}
	}
	return patterns
}

// Test that we strongly prefer exact matches.
//
// In golang/go#60027, we preferred "Runner" for the query "rune" over several
// results containing the word "rune" exactly. Following this observation,
// scoring was tweaked to more strongly emphasize sequential characters and
// exact matches.
func TestSymbolRanking_Issue60027(t *testing.T) {
	matcher := NewSymbolMatcher("rune")

	// symbols to match, in ascending order of ranking.
	symbols := []string{
		"Runner",
		"singleRuneParam",
		"Config.ifsRune",
		"Parser.rune",
	}
	prev := 0.0
	for _, sym := range symbols {
		_, score := matcher.Match([]string{sym})
		t.Logf("Match(%q) = %v", sym, score)
		if score < prev {
			t.Errorf("Match(%q) = _, %v, want > %v", sym, score, prev)
		}
		prev = score
	}
}

func TestChunkedMatch(t *testing.T) {
	matcher := NewSymbolMatcher("test")
	_, want := matcher.Match([]string{"test"})
	chunked := [][]string{
		{"", "test"},
		{"test", ""},
		{"te", "st"},
	}

	for _, chunks := range chunked {
		offset, score := matcher.Match(chunks)
		if offset != 0 || score != want {
			t.Errorf("Match(%v) = %v, %v, want 0, 1.0", chunks, offset, score)
		}
	}
}
