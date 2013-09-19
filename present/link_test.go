// Copyright 2012 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

package present

import "testing"

func TestInlineParsing(t *testing.T) {
	var tests = []struct {
		in     string
		link   string
		text   string
		length int
	}{
		{"[[http://golang.org]]", "http://golang.org", "golang.org", 21},
		{"[[http://golang.org][]]", "http://golang.org", "http://golang.org", 23},
		{"[[http://golang.org]] this is ignored", "http://golang.org", "golang.org", 21},
		{"[[http://golang.org][link]]", "http://golang.org", "link", 27},
		{"[[http://golang.org][two words]]", "http://golang.org", "two words", 32},
		{"[[http://golang.org][*link*]]", "http://golang.org", "<b>link</b>", 29},
		{"[[http://bad[url]]", "", "", 0},
		{"[[http://golang.org][a [[link]] ]]", "http://golang.org", "a [[link", 31},
		{"[[http:// *spaces* .com]]", "", "", 0},
		{"[[http://bad`char.com]]", "", "", 0},
		{" [[http://google.com]]", "", "", 0},
		{"[[mailto:gopher@golang.org][Gopher]]", "mailto:gopher@golang.org", "Gopher", 36},
		{"[[mailto:gopher@golang.org]]", "mailto:gopher@golang.org", "gopher@golang.org", 28},
	}

	for i, test := range tests {
		link, length := parseInlineLink(test.in)
		if length == 0 && test.length == 0 {
			continue
		}
		if a := renderLink(test.link, test.text); length != test.length || link != a {
			t.Errorf("#%d: parseInlineLink(%q):\ngot\t%q, %d\nwant\t%q, %d", i, test.in, link, length, a, test.length)
		}
	}
}
