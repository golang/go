// Copyright 2020 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

package source

import (
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
		if _, score := matcher([]string{test.s}); score > 0 != test.wantMatch {
			t.Errorf("parseQuery(%q) match for %q: %.2g, want match: %t", test.query, test.s, score, test.wantMatch)
		}
	}
}
