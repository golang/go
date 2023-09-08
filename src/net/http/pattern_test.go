// Copyright 2023 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

package http

import (
	"slices"
	"strings"
	"testing"
)

func TestParsePattern(t *testing.T) {
	lit := func(name string) segment {
		return segment{s: name}
	}

	wild := func(name string) segment {
		return segment{s: name, wild: true}
	}

	multi := func(name string) segment {
		s := wild(name)
		s.multi = true
		return s
	}

	for _, test := range []struct {
		in   string
		want pattern
	}{
		{"/", pattern{segments: []segment{multi("")}}},
		{"/a", pattern{segments: []segment{lit("a")}}},
		{
			"/a/",
			pattern{segments: []segment{lit("a"), multi("")}},
		},
		{"/path/to/something", pattern{segments: []segment{
			lit("path"), lit("to"), lit("something"),
		}}},
		{
			"/{w1}/lit/{w2}",
			pattern{
				segments: []segment{wild("w1"), lit("lit"), wild("w2")},
			},
		},
		{
			"/{w1}/lit/{w2}/",
			pattern{
				segments: []segment{wild("w1"), lit("lit"), wild("w2"), multi("")},
			},
		},
		{
			"example.com/",
			pattern{host: "example.com", segments: []segment{multi("")}},
		},
		{
			"GET /",
			pattern{method: "GET", segments: []segment{multi("")}},
		},
		{
			"POST example.com/foo/{w}",
			pattern{
				method:   "POST",
				host:     "example.com",
				segments: []segment{lit("foo"), wild("w")},
			},
		},
		{
			"/{$}",
			pattern{segments: []segment{lit("/")}},
		},
		{
			"DELETE example.com/a/{foo12}/{$}",
			pattern{method: "DELETE", host: "example.com", segments: []segment{lit("a"), wild("foo12"), lit("/")}},
		},
		{
			"/foo/{$}",
			pattern{segments: []segment{lit("foo"), lit("/")}},
		},
		{
			"/{a}/foo/{rest...}",
			pattern{segments: []segment{wild("a"), lit("foo"), multi("rest")}},
		},
		{
			"//",
			pattern{segments: []segment{lit(""), multi("")}},
		},
		{
			"/foo///./../bar",
			pattern{segments: []segment{lit("foo"), lit(""), lit(""), lit("."), lit(".."), lit("bar")}},
		},
		{
			"a.com/foo//",
			pattern{host: "a.com", segments: []segment{lit("foo"), lit(""), multi("")}},
		},
	} {
		got := mustParsePattern(t, test.in)
		if !got.equal(&test.want) {
			t.Errorf("%q:\ngot  %#v\nwant %#v", test.in, got, &test.want)
		}
	}
}

func TestParsePatternError(t *testing.T) {
	for _, test := range []struct {
		in       string
		contains string
	}{
		{"", "empty pattern"},
		{"A=B /", "invalid method"},
		{" ", "missing /"},
		{"/{w}x", "bad wildcard segment"},
		{"/x{w}", "bad wildcard segment"},
		{"/{wx", "bad wildcard segment"},
		{"/{a$}", "bad wildcard name"},
		{"/{}", "empty wildcard"},
		{"/{...}", "empty wildcard"},
		{"/{$...}", "bad wildcard"},
		{"/{$}/", "{$} not at end"},
		{"/{$}/x", "{$} not at end"},
		{"/{a...}/", "not at end"},
		{"/{a...}/x", "not at end"},
		{"{a}/b", "missing initial '/'"},
		{"/a/{x}/b/{x...}", "duplicate wildcard name"},
		{"GET //", "unclean path"},
	} {
		_, err := parsePattern(test.in)
		if err == nil || !strings.Contains(err.Error(), test.contains) {
			t.Errorf("%q:\ngot %v, want error containing %q", test.in, err, test.contains)
		}
	}
}

func (p1 *pattern) equal(p2 *pattern) bool {
	return p1.method == p2.method && p1.host == p2.host &&
		slices.Equal(p1.segments, p2.segments)
}

func TestIsValidHTTPToken(t *testing.T) {
	for _, test := range []struct {
		in   string
		want bool
	}{
		{"", false},
		{"GET", true},
		{"get", true},
		{"white space", false},
		{"#!~", true},
		{"a-b1_2", true},
		{"notok)", false},
	} {
		got := isValidHTTPToken(test.in)
		if g, w := got, test.want; g != w {
			t.Errorf("%q: got %t, want %t", test.in, g, w)
		}
	}
}

func mustParsePattern(t *testing.T, s string) *pattern {
	t.Helper()
	p, err := parsePattern(s)
	if err != nil {
		t.Fatal(err)
	}
	return p
}

func TestCompareMethods(t *testing.T) {
	for _, test := range []struct {
		p1, p2 string
		want   relationship
	}{
		{"/", "/", equivalent},
		{"GET /", "GET /", equivalent},
		{"HEAD /", "HEAD /", equivalent},
		{"POST /", "POST /", equivalent},
		{"GET /", "POST /", disjoint},
		{"GET /", "/", moreSpecific},
		{"HEAD /", "/", moreSpecific},
		{"GET /", "HEAD /", moreGeneral},
	} {
		pat1 := mustParsePattern(t, test.p1)
		pat2 := mustParsePattern(t, test.p2)
		got := pat1.compareMethods(pat2)
		if got != test.want {
			t.Errorf("%s vs %s: got %s, want %s", test.p1, test.p2, got, test.want)
		}
		got2 := pat2.compareMethods(pat1)
		want2 := inverseRelationship(test.want)
		if got2 != want2 {
			t.Errorf("%s vs %s: got %s, want %s", test.p2, test.p1, got2, want2)
		}
	}
}

func TestComparePaths(t *testing.T) {
	for _, test := range []struct {
		p1, p2 string
		want   relationship
	}{
		// A non-final pattern segment can have one of two values: literal or
		// single wildcard. A final pattern segment can have one of 5: empty
		// (trailing slash), literal, dollar, single wildcard, or multi
		// wildcard. Trailing slash and multi wildcard are the same.

		// A literal should be more specific than anything it overlaps, except itself.
		{"/a", "/a", equivalent},
		{"/a", "/b", disjoint},
		{"/a", "/", moreSpecific},
		{"/a", "/{$}", disjoint},
		{"/a", "/{x}", moreSpecific},
		{"/a", "/{x...}", moreSpecific},

		// Adding a segment doesn't change that.
		{"/b/a", "/b/a", equivalent},
		{"/b/a", "/b/b", disjoint},
		{"/b/a", "/b/", moreSpecific},
		{"/b/a", "/b/{$}", disjoint},
		{"/b/a", "/b/{x}", moreSpecific},
		{"/b/a", "/b/{x...}", moreSpecific},
		{"/{z}/a", "/{z}/a", equivalent},
		{"/{z}/a", "/{z}/b", disjoint},
		{"/{z}/a", "/{z}/", moreSpecific},
		{"/{z}/a", "/{z}/{$}", disjoint},
		{"/{z}/a", "/{z}/{x}", moreSpecific},
		{"/{z}/a", "/{z}/{x...}", moreSpecific},

		// Single wildcard on left.
		{"/{z}", "/a", moreGeneral},
		{"/{z}", "/a/b", disjoint},
		{"/{z}", "/{$}", disjoint},
		{"/{z}", "/{x}", equivalent},
		{"/{z}", "/", moreSpecific},
		{"/{z}", "/{x...}", moreSpecific},
		{"/b/{z}", "/b/a", moreGeneral},
		{"/b/{z}", "/b/a/b", disjoint},
		{"/b/{z}", "/b/{$}", disjoint},
		{"/b/{z}", "/b/{x}", equivalent},
		{"/b/{z}", "/b/", moreSpecific},
		{"/b/{z}", "/b/{x...}", moreSpecific},

		// Trailing slash on left.
		{"/", "/a", moreGeneral},
		{"/", "/a/b", moreGeneral},
		{"/", "/{$}", moreGeneral},
		{"/", "/{x}", moreGeneral},
		{"/", "/", equivalent},
		{"/", "/{x...}", equivalent},

		{"/b/", "/b/a", moreGeneral},
		{"/b/", "/b/a/b", moreGeneral},
		{"/b/", "/b/{$}", moreGeneral},
		{"/b/", "/b/{x}", moreGeneral},
		{"/b/", "/b/", equivalent},
		{"/b/", "/b/{x...}", equivalent},

		{"/{z}/", "/{z}/a", moreGeneral},
		{"/{z}/", "/{z}/a/b", moreGeneral},
		{"/{z}/", "/{z}/{$}", moreGeneral},
		{"/{z}/", "/{z}/{x}", moreGeneral},
		{"/{z}/", "/{z}/", equivalent},
		{"/{z}/", "/a/", moreGeneral},
		{"/{z}/", "/{z}/{x...}", equivalent},
		{"/{z}/", "/a/{x...}", moreGeneral},
		{"/a/{z}/", "/{z}/a/", overlaps},
		{"/a/{z}/b/", "/{x}/c/{y...}", overlaps},

		// Multi wildcard on left.
		{"/{m...}", "/a", moreGeneral},
		{"/{m...}", "/a/b", moreGeneral},
		{"/{m...}", "/{$}", moreGeneral},
		{"/{m...}", "/{x}", moreGeneral},
		{"/{m...}", "/", equivalent},
		{"/{m...}", "/{x...}", equivalent},

		{"/b/{m...}", "/b/a", moreGeneral},
		{"/b/{m...}", "/b/a/b", moreGeneral},
		{"/b/{m...}", "/b/{$}", moreGeneral},
		{"/b/{m...}", "/b/{x}", moreGeneral},
		{"/b/{m...}", "/b/", equivalent},
		{"/b/{m...}", "/b/{x...}", equivalent},
		{"/b/{m...}", "/a/{x...}", disjoint},

		{"/{z}/{m...}", "/{z}/a", moreGeneral},
		{"/{z}/{m...}", "/{z}/a/b", moreGeneral},
		{"/{z}/{m...}", "/{z}/{$}", moreGeneral},
		{"/{z}/{m...}", "/{z}/{x}", moreGeneral},
		{"/{z}/{m...}", "/{w}/", equivalent},
		{"/{z}/{m...}", "/a/", moreGeneral},
		{"/{z}/{m...}", "/{z}/{x...}", equivalent},
		{"/{z}/{m...}", "/a/{x...}", moreGeneral},
		{"/a/{m...}", "/a/b/{y...}", moreGeneral},
		{"/a/{m...}", "/a/{x}/{y...}", moreGeneral},
		{"/a/{z}/{m...}", "/a/b/{y...}", moreGeneral},
		{"/a/{z}/{m...}", "/{z}/a/", overlaps},
		{"/a/{z}/{m...}", "/{z}/b/{y...}", overlaps},
		{"/a/{z}/b/{m...}", "/{x}/c/{y...}", overlaps},

		// Dollar on left.
		{"/{$}", "/a", disjoint},
		{"/{$}", "/a/b", disjoint},
		{"/{$}", "/{$}", equivalent},
		{"/{$}", "/{x}", disjoint},
		{"/{$}", "/", moreSpecific},
		{"/{$}", "/{x...}", moreSpecific},

		{"/b/{$}", "/b", disjoint},
		{"/b/{$}", "/b/a", disjoint},
		{"/b/{$}", "/b/a/b", disjoint},
		{"/b/{$}", "/b/{$}", equivalent},
		{"/b/{$}", "/b/{x}", disjoint},
		{"/b/{$}", "/b/", moreSpecific},
		{"/b/{$}", "/b/{x...}", moreSpecific},
		{"/b/{$}", "/b/c/{x...}", disjoint},
		{"/b/{x}/a/{$}", "/{x}/c/{y...}", overlaps},

		{"/{z}/{$}", "/{z}/a", disjoint},
		{"/{z}/{$}", "/{z}/a/b", disjoint},
		{"/{z}/{$}", "/{z}/{$}", equivalent},
		{"/{z}/{$}", "/{z}/{x}", disjoint},
		{"/{z}/{$}", "/{z}/", moreSpecific},
		{"/{z}/{$}", "/a/", overlaps},
		{"/{z}/{$}", "/a/{x...}", overlaps},
		{"/{z}/{$}", "/{z}/{x...}", moreSpecific},
		{"/a/{z}/{$}", "/{z}/a/", overlaps},
	} {
		pat1 := mustParsePattern(t, test.p1)
		pat2 := mustParsePattern(t, test.p2)
		if g := pat1.comparePaths(pat1); g != equivalent {
			t.Errorf("%s does not match itself; got %s", pat1, g)
		}
		if g := pat2.comparePaths(pat2); g != equivalent {
			t.Errorf("%s does not match itself; got %s", pat2, g)
		}
		got := pat1.comparePaths(pat2)
		if got != test.want {
			t.Errorf("%s vs %s: got %s, want %s", test.p1, test.p2, got, test.want)
			t.Logf("pat1: %+v\n", pat1.segments)
			t.Logf("pat2: %+v\n", pat2.segments)
		}
		want2 := inverseRelationship(test.want)
		got2 := pat2.comparePaths(pat1)
		if got2 != want2 {
			t.Errorf("%s vs %s: got %s, want %s", test.p2, test.p1, got2, want2)
		}
	}
}

func TestConflictsWith(t *testing.T) {
	for _, test := range []struct {
		p1, p2 string
		want   bool
	}{
		{"/a", "/a", true},
		{"/a", "/ab", false},
		{"/a/b/cd", "/a/b/cd", true},
		{"/a/b/cd", "/a/b/c", false},
		{"/a/b/c", "/a/c/c", false},
		{"/{x}", "/{y}", true},
		{"/{x}", "/a", false}, // more specific
		{"/{x}/{y}", "/{x}/a", false},
		{"/{x}/{y}", "/{x}/a/b", false},
		{"/{x}", "/a/{y}", false},
		{"/{x}/{y}", "/{x}/a/", false},
		{"/{x}", "/a/{y...}", false},           // more specific
		{"/{x}/a/{y}", "/{x}/a/{y...}", false}, // more specific
		{"/{x}/{y}", "/{x}/a/{$}", false},      // more specific
		{"/{x}/{y}/{$}", "/{x}/a/{$}", false},
		{"/a/{x}", "/{x}/b", true},
		{"/", "GET /", false},
		{"/", "GET /foo", false},
		{"GET /", "GET /foo", false},
		{"GET /", "/foo", true},
		{"GET /foo", "HEAD /", true},
	} {
		pat1 := mustParsePattern(t, test.p1)
		pat2 := mustParsePattern(t, test.p2)
		got := pat1.conflictsWith(pat2)
		if got != test.want {
			t.Errorf("%q.ConflictsWith(%q) = %t, want %t",
				test.p1, test.p2, got, test.want)
		}
		// conflictsWith should be commutative.
		got = pat2.conflictsWith(pat1)
		if got != test.want {
			t.Errorf("%q.ConflictsWith(%q) = %t, want %t",
				test.p2, test.p1, got, test.want)
		}
	}
}
