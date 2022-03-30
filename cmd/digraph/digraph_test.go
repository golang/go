// Copyright 2019 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.
package main

import (
	"bytes"
	"fmt"
	"reflect"
	"sort"
	"strings"
	"testing"
)

func TestDigraph(t *testing.T) {
	const g1 = `
socks shoes
shorts pants
pants belt shoes
shirt tie sweater
sweater jacket
hat
`

	const g2 = `
a b c
b d
c d
d c
e e
`

	for _, test := range []struct {
		name  string
		input string
		cmd   string
		args  []string
		want  string
	}{
		{"nodes", g1, "nodes", nil, "belt\nhat\njacket\npants\nshirt\nshoes\nshorts\nsocks\nsweater\ntie\n"},
		{"reverse", g1, "reverse", []string{"jacket"}, "jacket\nshirt\nsweater\n"},
		{"transpose", g1, "transpose", nil, "belt pants\njacket sweater\npants shorts\nshoes pants\nshoes socks\nsweater shirt\ntie shirt\n"},
		{"forward", g1, "forward", []string{"socks"}, "shoes\nsocks\n"},
		{"forward multiple args", g1, "forward", []string{"socks", "sweater"}, "jacket\nshoes\nsocks\nsweater\n"},
		{"scss", g2, "sccs", nil, "c d\ne\n"},
		{"scc", g2, "scc", []string{"d"}, "c\nd\n"},
		{"succs", g2, "succs", []string{"a"}, "b\nc\n"},
		{"preds", g2, "preds", []string{"c"}, "a\nd\n"},
		{"preds multiple args", g2, "preds", []string{"c", "d"}, "a\nb\nc\nd\n"},
	} {
		t.Run(test.name, func(t *testing.T) {
			stdin = strings.NewReader(test.input)
			stdout = new(bytes.Buffer)
			if err := digraph(test.cmd, test.args); err != nil {
				t.Fatal(err)
			}

			got := stdout.(fmt.Stringer).String()
			if got != test.want {
				t.Errorf("digraph(%s, %s) = got %q, want %q", test.cmd, test.args, got, test.want)
			}
		})
	}

	// TODO(adonovan):
	// - test somepath (it's nondeterministic).
	// - test errors
}

func TestAllpaths(t *testing.T) {
	for _, test := range []struct {
		name string
		in   string
		to   string // from is always "A"
		want string
	}{
		{
			name: "Basic",
			in:   "A B\nB C",
			to:   "B",
			want: "A B\n",
		},
		{
			name: "Long",
			in:   "A B\nB C\n",
			to:   "C",
			want: "A B\nB C\n",
		},
		{
			name: "Cycle Basic",
			in:   "A B\nB A",
			to:   "B",
			want: "A B\nB A\n",
		},
		{
			name: "Cycle Path Out",
			// A <-> B -> C -> D
			in:   "A B\nB A\nB C\nC D",
			to:   "C",
			want: "A B\nB A\nB C\n",
		},
		{
			name: "Cycle Path Out Further Out",
			// A -> B <-> C -> D -> E
			in:   "A B\nB C\nC D\nC B\nD E",
			to:   "D",
			want: "A B\nB C\nC B\nC D\n",
		},
		{
			name: "Two Paths Basic",
			//           /-> C --\
			// A -> B --          -> E -> F
			//           \-> D --/
			in:   "A B\nB C\nC E\nB D\nD E\nE F",
			to:   "E",
			want: "A B\nB C\nB D\nC E\nD E\n",
		},
		{
			name: "Two Paths With One Immediately From Start",
			//      /-> B -+ -> D
			// A --        |
			//      \-> C <+
			in:   "A B\nA C\nB C\nB D",
			to:   "C",
			want: "A B\nA C\nB C\n",
		},
		{
			name: "Two Paths Further Up",
			//      /-> B --\
			// A --          -> D -> E -> F
			//      \-> C --/
			in:   "A B\nA C\nB D\nC D\nD E\nE F",
			to:   "E",
			want: "A B\nA C\nB D\nC D\nD E\n",
		},
		{
			// We should include A - C  - D even though it's further up the
			// second path than D (which would already be in the graph by
			// the time we get around to integrating the second path).
			name: "Two Splits",
			//      /-> B --\         /-> E --\
			// A --           -> D --          -> G -> H
			//      \-> C --/         \-> F --/
			in:   "A B\nA C\nB D\nC D\nD E\nD F\nE G\nF G\nG H",
			to:   "G",
			want: "A B\nA C\nB D\nC D\nD E\nD F\nE G\nF G\n",
		},
		{
			// D - E should not be duplicated.
			name: "Two Paths - Two Splits With Gap",
			//      /-> B --\              /-> F --\
			// A --           -> D -> E --          -> H -> I
			//      \-> C --/              \-> G --/
			in:   "A B\nA C\nB D\nC D\nD E\nE F\nE G\nF H\nG H\nH I",
			to:   "H",
			want: "A B\nA C\nB D\nC D\nD E\nE F\nE G\nF H\nG H\n",
		},
	} {
		t.Run(test.name, func(t *testing.T) {
			stdin = strings.NewReader(test.in)
			stdout = new(bytes.Buffer)
			if err := digraph("allpaths", []string{"A", test.to}); err != nil {
				t.Fatal(err)
			}

			got := stdout.(fmt.Stringer).String()
			if got != test.want {
				t.Errorf("digraph(allpaths, A, %s) = got %q, want %q", test.to, got, test.want)
			}
		})
	}
}

func TestSomepath(t *testing.T) {
	for _, test := range []struct {
		name string
		in   string
		to   string
		// somepath is non-deterministic, so we have to provide all the
		// possible options. Each option is separated with |.
		wantAnyOf string
	}{
		{
			name:      "Basic",
			in:        "A B\n",
			to:        "B",
			wantAnyOf: "A B",
		},
		{
			name:      "Basic With Cycle",
			in:        "A B\nB A",
			to:        "B",
			wantAnyOf: "A B",
		},
		{
			name: "Two Paths",
			//      /-> B --\
			// A --          -> D
			//      \-> C --/
			in:        "A B\nA C\nB D\nC D",
			to:        "D",
			wantAnyOf: "A B\nB D|A C\nC D",
		},
	} {
		t.Run(test.name, func(t *testing.T) {
			stdin = strings.NewReader(test.in)
			stdout = new(bytes.Buffer)
			if err := digraph("somepath", []string{"A", test.to}); err != nil {
				t.Fatal(err)
			}

			got := stdout.(fmt.Stringer).String()
			lines := strings.Split(got, "\n")
			sort.Strings(lines)
			got = strings.Join(lines[1:], "\n")

			var oneMatch bool
			for _, want := range strings.Split(test.wantAnyOf, "|") {
				if got == want {
					oneMatch = true
				}
			}
			if !oneMatch {
				t.Errorf("digraph(somepath, A, %s) = got %q, want any of\n%s", test.to, got, test.wantAnyOf)
			}
		})
	}
}

func TestSplit(t *testing.T) {
	for _, test := range []struct {
		line string
		want []string
	}{
		{`one "2a 2b" three`, []string{"one", "2a 2b", "three"}},
		{`one tw"\n\x0a\u000a\012"o three`, []string{"one", "tw\n\n\n\no", "three"}},
	} {
		got, err := split(test.line)
		if err != nil {
			t.Errorf("split(%s) failed: %v", test.line, err)
		}
		if !reflect.DeepEqual(got, test.want) {
			t.Errorf("split(%s) = %v, want %v", test.line, got, test.want)
		}
	}
}

func TestQuotedLength(t *testing.T) {
	for _, test := range []struct {
		input string
		want  int
	}{
		{`"abc"`, 5},
		{`"abc"def`, 5},
		{`"abc\"d"ef`, 8}, // "abc\"d" is consumed, ef is residue
		{`"\012\n\x0a\u000a\U0000000a"`, 28},
		{"\"\xff\"", 3}, // bad UTF-8 is ok
		{`"\xff"`, 6},   // hex escape for bad UTF-8 is ok
	} {
		got, ok := quotedLength(test.input)
		if !ok {
			got = 0
		}
		if got != test.want {
			t.Errorf("quotedLength(%s) = %d, want %d", test.input, got, test.want)
		}
	}

	// errors
	for _, input := range []string{
		``,            // not a quotation
		`a`,           // not a quotation
		`'a'`,         // not a quotation
		`"a`,          // not terminated
		`"\0"`,        // short octal escape
		`"\x1"`,       // short hex escape
		`"\u000"`,     // short \u escape
		`"\U0000000"`, // short \U escape
		`"\k"`,        // invalid escape
		"\"ab\nc\"",   // newline
	} {
		if n, ok := quotedLength(input); ok {
			t.Errorf("quotedLength(%s) = %d, want !ok", input, n)
		}
	}
}

func TestFocus(t *testing.T) {
	for _, test := range []struct {
		name  string
		in    string
		focus string
		want  string
	}{
		{
			name:  "Basic",
			in:    "A B",
			focus: "B",
			want:  "A B\n",
		},
		{
			name: "Some Nodes Not Included",
			// C does not have a path involving B, and should not be included
			// in the output.
			in:    "A B\nA C",
			focus: "B",
			want:  "A B\n",
		},
		{
			name: "Cycle In Path",
			// A <-> B -> C
			in:    "A B\nB A\nB C",
			focus: "C",
			want:  "A B\nB A\nB C\n",
		},
		{
			name: "Cycle Out Of Path",
			// C <- A <->B
			in:    "A B\nB A\nB C",
			focus: "C",
			want:  "A B\nB A\nB C\n",
		},
		{
			name: "Complex",
			// Paths in and out from focus.
			//                   /-> F
			//      /-> B -> D --
			// A --              \-> E
			//      \-> C
			in:    "A B\nA C\nB D\nD F\nD E",
			focus: "D",
			want:  "A B\nB D\nD E\nD F\n",
		},
	} {
		t.Run(test.name, func(t *testing.T) {
			stdin = strings.NewReader(test.in)
			stdout = new(bytes.Buffer)
			if err := digraph("focus", []string{test.focus}); err != nil {
				t.Fatal(err)
			}
			got := stdout.(fmt.Stringer).String()
			if got != test.want {
				t.Errorf("digraph(focus, %s) = got %q, want %q", test.focus, got, test.want)
			}
		})
	}
}
