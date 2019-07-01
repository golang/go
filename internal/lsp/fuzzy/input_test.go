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
	str   string
	input fuzzy.Input
	want  string
}{
	{str: "abc", want: "Ccc", input: fuzzy.Text},
	{str: ".abc", want: " Ccc", input: fuzzy.Text},
	{str: "abc def", want: "Ccc Ccc", input: fuzzy.Text},
	{str: "SWT MyID", want: "Cuu CcCu", input: fuzzy.Text},
	{str: "ID", want: "Cu", input: fuzzy.Text},
	{str: "IDD", want: "Cuu", input: fuzzy.Text},
	{str: " ID ", want: " Cu ", input: fuzzy.Text},
	{str: "IDSome", want: "CuCccc", input: fuzzy.Text},
	{str: "0123456789", want: "Cccccccccc", input: fuzzy.Text},
	{str: "abcdefghigklmnopqrstuvwxyz", want: "Cccccccccccccccccccccccccc", input: fuzzy.Text},
	{str: "ABCDEFGHIGKLMNOPQRSTUVWXYZ", want: "Cuuuuuuuuuuuuuuuuuuuuuuuuu", input: fuzzy.Text},
	{str: "こんにちは", want: "Ccccccccccccccc", input: fuzzy.Text}, // We don't parse unicode
	{str: ":/.", want: "   ", input: fuzzy.Text},

	// Filenames
	{str: "abc/def", want: "Ccc/Ccc", input: fuzzy.Filename},
	{str: " abc_def", want: " Ccc Ccc", input: fuzzy.Filename},
	{str: " abc_DDf", want: " Ccc CCc", input: fuzzy.Filename},
	{str: ":.", want: "  ", input: fuzzy.Filename},

	// Symbols
	{str: "abc::def::goo", want: "Ccc//Ccc//Ccc", input: fuzzy.Symbol},
	{str: "proto::Message", want: "Ccccc//Ccccccc", input: fuzzy.Symbol},
	{str: "AbstractSWTFactory", want: "CcccccccCuuCcccccc", input: fuzzy.Symbol},
	{str: "Abs012", want: "Cccccc", input: fuzzy.Symbol},
	{str: "/", want: " ", input: fuzzy.Symbol},
	{str: "fOO", want: "CCu", input: fuzzy.Symbol},
	{str: "fo_oo.o_oo", want: "Cc Cc/C Cc", input: fuzzy.Symbol},
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
		fuzzy.RuneRoles(tc.str, tc.input, gotRoles)
		got := rolesString(gotRoles)
		if got != tc.want {
			t.Errorf("roles(%s) = %v; want %v", tc.str, got, tc.want)
		}
	}
}

func words(strWords ...string) [][]byte {
	var ret [][]byte
	for _, w := range strWords {
		ret = append(ret, []byte(w))
	}
	return ret
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
		roles := fuzzy.RuneRoles(tc.input, fuzzy.Symbol, nil)

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
	str   string
	input fuzzy.Input
	want  string
}{
	{
		str:   "identifier",
		input: fuzzy.Symbol,
		want:  "identifier",
	},
	{
		str:   "two_words",
		input: fuzzy.Symbol,
		want:  "two_words",
	},
	{
		str:   "first::second",
		input: fuzzy.Symbol,
		want:  "second",
	},
	{
		str:   "foo.bar.FOOBar_buz123_test",
		input: fuzzy.Symbol,
		want:  "FOOBar_buz123_test",
	},
	{
		str:   "golang.org/x/tools/internal/lsp/fuzzy_matcher.go",
		input: fuzzy.Filename,
		want:  "fuzzy_matcher.go",
	},
	{
		str:   "golang.org/x/tools/internal/lsp/fuzzy_matcher.go",
		input: fuzzy.Text,
		want:  "golang.org/x/tools/internal/lsp/fuzzy_matcher.go",
	},
}

func TestLastSegment(t *testing.T) {
	for _, tc := range lastSegmentSplitTests {
		roles := fuzzy.RuneRoles(tc.str, tc.input, nil)

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
		fuzzy.RuneRoles(str, fuzzy.Symbol, out)
	}
	b.SetBytes(int64(len(str)))
}
