// Copyright 2013 The Go Authors.  All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

package goobj

import "testing"

var importPathToPrefixTests = []struct {
	in  string
	out string
}{
	{"runtime", "runtime"},
	{"sync/atomic", "sync/atomic"},
	{"code.google.com/p/go.tools/godoc", "code.google.com/p/go.tools/godoc"},
	{"foo.bar/baz.quux", "foo.bar/baz%2equux"},
	{"", ""},
	{"%foo%bar", "%25foo%25bar"},
	{"\x01\x00\x7F☺", "%01%00%7f%e2%98%ba"},
}

func TestImportPathToPrefix(t *testing.T) {
	for _, tt := range importPathToPrefixTests {
		if out := importPathToPrefix(tt.in); out != tt.out {
			t.Errorf("importPathToPrefix(%q) = %q, want %q", tt.in, out, tt.out)
		}
	}
}
