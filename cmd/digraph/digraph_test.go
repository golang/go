package main

import (
	"bytes"
	"fmt"
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
