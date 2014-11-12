package main

import (
	"bytes"
	"fmt"
	"reflect"
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
`

	for _, test := range []struct {
		input string
		cmd   string
		args  []string
		want  string
	}{
		{g1, "nodes", nil, "belt\nhat\njacket\npants\nshirt\nshoes\nshorts\nsocks\nsweater\ntie\n"},
		{g1, "reverse", []string{"jacket"}, "jacket\nshirt\nsweater\n"},
		{g1, "forward", []string{"socks"}, "shoes\nsocks\n"},
		{g1, "forward", []string{"socks", "sweater"}, "jacket\nshoes\nsocks\nsweater\n"},

		{g2, "allpaths", []string{"a", "d"}, "a\nb\nc\nd\n"},

		{g2, "sccs", nil, "a\nb\nc d\n"},
		{g2, "scc", []string{"d"}, "c\nd\n"},
		{g2, "succs", []string{"a"}, "b\nc\n"},
		{g2, "preds", []string{"c"}, "a\nd\n"},
		{g2, "preds", []string{"c", "d"}, "a\nb\nc\nd\n"},
	} {
		stdin = strings.NewReader(test.input)
		stdout = new(bytes.Buffer)
		if err := digraph(test.cmd, test.args); err != nil {
			t.Error(err)
			continue
		}

		got := stdout.(fmt.Stringer).String()
		if got != test.want {
			t.Errorf("digraph(%s, %s) = %q, want %q", test.cmd, test.args, got, test.want)
		}
	}

	// TODO(adonovan):
	// - test somepath (it's nondeterministic).
	// - test errors
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
