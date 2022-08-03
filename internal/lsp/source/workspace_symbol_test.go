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
		matcher := parseQuery(test.query, newFuzzyMatcher)
		if _, score := matcher([]string{test.s}); score > 0 != test.wantMatch {
			t.Errorf("parseQuery(%q) match for %q: %.2g, want match: %t", test.query, test.s, score, test.wantMatch)
		}
	}
}

func TestFiltererDisallow(t *testing.T) {
	tests := []struct {
		filters  []string
		included []string
		excluded []string
	}{
		{
			[]string{"+**/c.go"},
			[]string{"a/c.go", "a/b/c.go"},
			[]string{},
		},
		{
			[]string{"+a/**/c.go"},
			[]string{"a/b/c.go", "a/b/d/c.go", "a/c.go"},
			[]string{},
		},
		{
			[]string{"-a/c.go", "+a/**"},
			[]string{"a/c.go"},
			[]string{},
		},
		{
			[]string{"+a/**/c.go", "-**/c.go"},
			[]string{},
			[]string{"a/b/c.go"},
		},
		{
			[]string{"+a/**/c.go", "-a/**"},
			[]string{},
			[]string{"a/b/c.go"},
		},
		{
			[]string{"+**/c.go", "-a/**/c.go"},
			[]string{},
			[]string{"a/b/c.go"},
		},
		{
			[]string{"+foobar", "-foo"},
			[]string{"foobar", "foobar/a"},
			[]string{"foo", "foo/a"},
		},
		{
			[]string{"+", "-"},
			[]string{},
			[]string{"foobar", "foobar/a", "foo", "foo/a"},
		},
		{
			[]string{"-", "+"},
			[]string{"foobar", "foobar/a", "foo", "foo/a"},
			[]string{},
		},
		{
			[]string{"-a/**/b/**/c.go"},
			[]string{},
			[]string{"a/x/y/z/b/f/g/h/c.go"},
		},
		// tests for unsupported glob operators
		{
			[]string{"+**/c.go", "-a/*/c.go"},
			[]string{"a/b/c.go"},
			[]string{},
		},
		{
			[]string{"+**/c.go", "-a/?/c.go"},
			[]string{"a/b/c.go"},
			[]string{},
		},
		{
			[]string{"-b"}, // should only filter paths prefixed with the "b" directory
			[]string{"a/b/c.go", "bb"},
			[]string{"b/c/d.go", "b"},
		},
	}

	for _, test := range tests {
		filterer := NewFilterer(test.filters)
		for _, inc := range test.included {
			if filterer.Disallow(inc) {
				t.Errorf("Filters %v excluded %v, wanted included", test.filters, inc)
			}
		}

		for _, exc := range test.excluded {
			if !filterer.Disallow(exc) {
				t.Errorf("Filters %v included %v, wanted excluded", test.filters, exc)
			}
		}
	}
}
