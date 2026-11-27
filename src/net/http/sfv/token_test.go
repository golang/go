// Copyright 2020 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

package sfv

import (
	"strings"
	"testing"
)

func TestTokenMarshalSFV(t *testing.T) {
	data := []struct {
		in    string
		valid bool
	}{
		{"abc'!#$%*+-.^_|~:/`", true},
		{"H3lLo", true},
		{"a*foo", true},
		{"a!1", true},
		{"a#1", true},
		{"a$1", true},
		{"a%1", true},
		{"a&1", true},
		{"a'1", true},
		{"a*1", true},
		{"a+1", true},
		{"a-1", true},
		{"a.1", true},
		{"a^1", true},
		{"a_1", true},
		{"a`1", true},
		{"a|1", true},
		{"a~1", true},
		{"a:1", true},
		{"a/1", true},
		{`0foo`, false},
		{`!foo`, false},
		{"1abc", false},
		{"", false},
		{"hel\tlo", false},
		{"hel\x1flo", false},
		{"hel\x7flo", false},
		{"Kévin", false},
	}

	var b strings.Builder

	for _, d := range data {
		b.Reset()

		err := Token(d.in).marshalSFV(&b)
		if d.valid && err != nil {
			t.Errorf("error not expected for %v, got %v", d.in, err)
		} else if !d.valid && err == nil {
			t.Errorf("error expected for %v, got %v", d.in, err)
		}

		if d.valid && b.String() != d.in {
			t.Errorf("got %v; want %v", b.String(), d.in)
		}
	}
}

func TestParseToken(t *testing.T) {
	data := []struct {
		in  string
		out Token
		err bool
	}{
		{"t", Token("t"), false},
		{"tok", Token("tok"), false},
		{"*t!o&k", Token("*t!o&k"), false},
		{"t=", Token("t"), false},
		{"", Token(""), true},
		{"é", Token(""), true},
	}

	for _, d := range data {
		s := &scanner{data: d.in}

		i, err := parseToken(s)
		if d.err && err == nil {
			t.Errorf("parseToken(%s): error expected", d.in)
		}

		if !d.err && d.out != i {
			t.Errorf("parseToken(%s) = %v, %v; %v, <nil> expected", d.in, i, err, d.out)
		}
	}
}
