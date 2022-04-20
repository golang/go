// Copyright 2011 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

// These tests are carried forward from the old go/doc implementation.

package comment

import "testing"

var oldHeadingTests = []struct {
	line string
	ok   bool
}{
	{"Section", true},
	{"A typical usage", true},
	{"ΔΛΞ is Greek", true},
	{"Foo 42", true},
	{"", false},
	{"section", false},
	{"A typical usage:", false},
	{"This code:", false},
	{"δ is Greek", false},
	{"Foo §", false},
	{"Fermat's Last Sentence", true},
	{"Fermat's", true},
	{"'sX", false},
	{"Ted 'Too' Bar", false},
	{"Use n+m", false},
	{"Scanning:", false},
	{"N:M", false},
}

func TestIsOldHeading(t *testing.T) {
	for _, tt := range oldHeadingTests {
		if isOldHeading(tt.line, []string{"Text.", "", tt.line, "", "Text."}, 2) != tt.ok {
			t.Errorf("isOldHeading(%q) = %v, want %v", tt.line, !tt.ok, tt.ok)
		}
	}
}

var autoURLTests = []struct {
	in, out string
}{
	{"", ""},
	{"http://[::1]:8080/foo.txt", "http://[::1]:8080/foo.txt"},
	{"https://www.google.com) after", "https://www.google.com"},
	{"https://www.google.com:30/x/y/z:b::c. After", "https://www.google.com:30/x/y/z:b::c"},
	{"http://www.google.com/path/:;!-/?query=%34b#093124", "http://www.google.com/path/:;!-/?query=%34b#093124"},
	{"http://www.google.com/path/:;!-/?query=%34bar#093124", "http://www.google.com/path/:;!-/?query=%34bar#093124"},
	{"http://www.google.com/index.html! After", "http://www.google.com/index.html"},
	{"http://www.google.com/", "http://www.google.com/"},
	{"https://www.google.com/", "https://www.google.com/"},
	{"http://www.google.com/path.", "http://www.google.com/path"},
	{"http://en.wikipedia.org/wiki/Camellia_(cipher)", "http://en.wikipedia.org/wiki/Camellia_(cipher)"},
	{"http://www.google.com/)", "http://www.google.com/"},
	{"http://gmail.com)", "http://gmail.com"},
	{"http://gmail.com))", "http://gmail.com"},
	{"http://gmail.com ((http://gmail.com)) ()", "http://gmail.com"},
	{"http://example.com/ quux!", "http://example.com/"},
	{"http://example.com/%2f/ /world.", "http://example.com/%2f/"},
	{"http: ipsum //host/path", ""},
	{"javascript://is/not/linked", ""},
	{"http://foo", "http://foo"},
	{"https://www.example.com/person/][Person Name]]", "https://www.example.com/person/"},
	{"http://golang.org/)", "http://golang.org/"},
	{"http://golang.org/hello())", "http://golang.org/hello()"},
	{"http://git.qemu.org/?p=qemu.git;a=blob;f=qapi-schema.json;hb=HEAD", "http://git.qemu.org/?p=qemu.git;a=blob;f=qapi-schema.json;hb=HEAD"},
	{"https://foo.bar/bal/x(])", "https://foo.bar/bal/x"}, // inner ] causes (]) to be cut off from URL
	{"http://bar(])", "http://bar"},                       // same
}

func TestAutoURL(t *testing.T) {
	for _, tt := range autoURLTests {
		url, ok := autoURL(tt.in)
		if url != tt.out || ok != (tt.out != "") {
			t.Errorf("autoURL(%q) = %q, %v, want %q, %v", tt.in, url, ok, tt.out, tt.out != "")
		}
	}
}
