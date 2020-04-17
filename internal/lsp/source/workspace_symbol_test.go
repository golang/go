// Copyright 2020 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

package source

import (
	"strings"
	"testing"
)

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
