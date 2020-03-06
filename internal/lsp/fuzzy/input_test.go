// Copyright 2019 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

package fuzzy_test

import (
	"bytes"
	"sort"
	"testing"

	"golang.org/x/tools/internal/lsp/fuzzy"
)

var rolesTests = []struct {
	str  string
	want string
}{
	{str: "abc::def::goo", want: "Ccc//Ccc//Ccc"},
	{str: "proto::Message", want: "Ccccc//Ccccccc"},
	{str: "AbstractSWTFactory", want: "CcccccccCuuCcccccc"},
	{str: "Abs012", want: "Cccccc"},
	{str: "/", want: " "},
	{str: "fOO", want: "CCu"},
	{str: "fo_oo.o_oo", want: "Cc Cc/C Cc"},
}

func rolesString(roles []fuzzy.RuneRole) string {
	var buf bytes.Buffer
	for _, r := range roles {
		buf.WriteByte(" /cuC"[int(r)])
	}
	return buf.String()
}

func TestRoles(t *testing.T) {
	for _, tc := range rolesTests {
		gotRoles := make([]fuzzy.RuneRole, len(tc.str))
		fuzzy.RuneRoles(tc.str, gotRoles)
		got := rolesString(gotRoles)
		if got != tc.want {
			t.Errorf("roles(%s) = %v; want %v", tc.str, got, tc.want)
		}
	}
}

var wordSplitTests = []struct {
	input string
	want  []string
}{
	{
		input: "foo bar baz",
		want:  []string{"foo", "bar", "baz"},
	},
	{
		input: "fooBarBaz",
		want:  []string{"foo", "Bar", "Baz"},
	},
	{
		input: "FOOBarBAZ",
		want:  []string{"FOO", "Bar", "BAZ"},
	},
	{
		input: "foo123_bar2Baz3",
		want:  []string{"foo123", "bar2", "Baz3"},
	},
}

func TestWordSplit(t *testing.T) {
	for _, tc := range wordSplitTests {
		roles := fuzzy.RuneRoles(tc.input, nil)

		var got []string
		consumer := func(i, j int) {
			got = append(got, tc.input[i:j])
		}
		fuzzy.Words(roles, consumer)

		if eq := diffStringLists(tc.want, got); !eq {
			t.Errorf("input %v: (want %v -> got %v)", tc.input, tc.want, got)
		}
	}
}

func diffStringLists(a, b []string) bool {
	if len(a) != len(b) {
		return false
	}
	sort.Strings(a)
	sort.Strings(b)
	for i := range a {
		if a[i] != b[i] {
			return false
		}
	}
	return true
}

var lastSegmentSplitTests = []struct {
	str  string
	want string
}{
	{
		str:  "identifier",
		want: "identifier",
	},
	{
		str:  "two_words",
		want: "two_words",
	},
	{
		str:  "first::second",
		want: "second",
	},
	{
		str:  "foo.bar.FOOBar_buz123_test",
		want: "FOOBar_buz123_test",
	},
}

func TestLastSegment(t *testing.T) {
	for _, tc := range lastSegmentSplitTests {
		roles := fuzzy.RuneRoles(tc.str, nil)

		got := fuzzy.LastSegment(tc.str, roles)

		if got != tc.want {
			t.Errorf("str %v: want %v; got %v", tc.str, tc.want, got)
		}
	}
}

func BenchmarkRoles(b *testing.B) {
	str := "AbstractSWTFactory"
	out := make([]fuzzy.RuneRole, len(str))

	for i := 0; i < b.N; i++ {
		fuzzy.RuneRoles(str, out)
	}
	b.SetBytes(int64(len(str)))
}
