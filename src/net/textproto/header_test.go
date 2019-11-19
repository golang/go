// Copyright 2010 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

package textproto

import "testing"

type canonicalHeaderKeyTest struct {
	in, out string
}

var canonicalHeaderKeyTests = []canonicalHeaderKeyTest{
	{"a-b-c", "A-B-C"},
	{"a-1-c", "A-1-C"},
	{"User-Agent", "User-Agent"},
	{"uSER-aGENT", "User-Agent"},
	{"user-agent", "User-Agent"},
	{"USER-AGENT", "User-Agent"},

	// Other valid tchar bytes in tokens:
	{"foo-bar_baz", "Foo-Bar_baz"},
	{"foo-bar$baz", "Foo-Bar$baz"},
	{"foo-bar~baz", "Foo-Bar~baz"},
	{"foo-bar*baz", "Foo-Bar*baz"},

	// Non-ASCII or anything with spaces or non-token chars is unchanged:
	{"üser-agenT", "üser-agenT"},
	{"a B", "a B"},

	// This caused a panic due to mishandling of a space:
	{"C Ontent-Transfer-Encoding", "C Ontent-Transfer-Encoding"},
	{"foo bar", "foo bar"},
}

func TestCanonicalMIMEHeaderKey(t *testing.T) {
	for _, tt := range canonicalHeaderKeyTests {
		if s := CanonicalMIMEHeaderKey(tt.in); s != tt.out {
			t.Errorf("CanonicalMIMEHeaderKey(%q) = %q, want %q", tt.in, s, tt.out)
		}
	}
}

// Issue #34799 add a Header method to get multiple values []string, with canonicalized key
func TestMIMEHeaderMultipleValues(t *testing.T) {
	testHeader := MIMEHeader{
		"Set-Cookie": {"cookie 1", "cookie 2"},
	}
	values := testHeader.Values("set-cookie")
	n := len(values)
	if n != 2 {
		t.Errorf("count: %d; want 2", n)
	}
}
