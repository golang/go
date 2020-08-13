// Copyright 2020 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

package source

import (
	"strings"
	"testing"
)

func TestParseQuery(t *testing.T) {
	tests := []struct {
		query, s  string
		wantMatch bool
	}{
		{"", "anything", false},
		{"any", "anything", true},
		{"any$", "anything", false},
		{"ing$", "anything", true},
		{"ing$", "anythinG", true},
		{"inG$", "anything", false},
		{"^any", "anything", true},
		{"^any", "Anything", true},
		{"^Any", "anything", false},
		{"at", "anything", true},
		// TODO: this appears to be a bug in the fuzzy matching algorithm. 'At'
		// should cause a case-sensitive match.
		// {"At", "anything", false},
		{"At", "Anything", true},
		{"'yth", "Anything", true},
		{"'yti", "Anything", false},
		{"'any 'thing", "Anything", true},
		{"anythn nythg", "Anything", true},
		{"ntx", "Anything", false},
		{"anythn", "anything", true},
		{"ing", "anything", true},
		{"anythn nythgx", "anything", false},
	}

	for _, test := range tests {
		matcher := parseQuery(test.query)
		if score := matcher(test.s); score > 0 != test.wantMatch {
			t.Errorf("parseQuery(%q) match for %q: %.2g, want match: %t", test.query, test.s, score, test.wantMatch)
		}
	}
}

func TestBestMatch(t *testing.T) {
	tests := []struct {
		desc      string
		symbol    string
		matcher   matcherFunc
		wantMatch string
		wantScore float64
	}{
		{
			desc:      "shortest match",
			symbol:    "foo/bar/baz.quux",
			matcher:   func(string) float64 { return 1.0 },
			wantMatch: "quux",
			wantScore: 1.0,
		},
		{
			desc:   "partial match",
			symbol: "foo/bar/baz.quux",
			matcher: func(s string) float64 {
				if strings.HasPrefix(s, "bar") {
					return 1.0
				}
				return 0.0
			},
			wantMatch: "bar/baz.quux",
			wantScore: 1.0,
		},
		{
			desc:   "longest match",
			symbol: "foo/bar/baz.quux",
			matcher: func(s string) float64 {
				parts := strings.Split(s, "/")
				return float64(len(parts))
			},
			wantMatch: "foo/bar/baz.quux",
			wantScore: 3.0,
		},
	}

	for _, test := range tests {
		test := test
		t.Run(test.desc, func(t *testing.T) {
			gotMatch, gotScore := bestMatch(test.symbol, test.matcher)
			if gotMatch != test.wantMatch || gotScore != test.wantScore {
				t.Errorf("bestMatch(%q, matcher) = (%q, %.2g), want (%q, %.2g)", test.symbol, gotMatch, gotScore, test.wantMatch, test.wantScore)
			}
		})
	}
}
