// Copyright 2017 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

package objabi

import "testing"

func TestPathToPrefix(t *testing.T) {
	tests := []struct {
		Path     string
		Expected string
	}{{"foo/bar/v1", "foo/bar/v1"},
		{"foo/bar/v.1", "foo/bar/v%2e1"},
		{"f.o.o/b.a.r/v1", "f.o.o/b.a.r/v1"},
		{"f.o.o/b.a.r/v.1", "f.o.o/b.a.r/v%2e1"},
		{"f.o.o/b.a.r/v..1", "f.o.o/b.a.r/v%2e%2e1"},
		{"f.o.o/b.a.r/v..1.", "f.o.o/b.a.r/v%2e%2e1%2e"},
		{"f.o.o/b.a.r/v%1", "f.o.o/b.a.r/v%251"},
		{"runtime", "runtime"},
		{"sync/atomic", "sync/atomic"},
		{"golang.org/x/tools/godoc", "golang.org/x/tools/godoc"},
		{"foo.bar/baz.quux", "foo.bar/baz%2equux"},
		{"", ""},
		{"%foo%bar", "%25foo%25bar"},
		{"\x01\x00\x7F☺", "%01%00%7f%e2%98%ba"},
	}
	for _, tc := range tests {
		if got := PathToPrefix(tc.Path); got != tc.Expected {
			t.Errorf("expected PathToPrefix(%s) = %s, got %s", tc.Path, tc.Expected, got)
		}
	}
}
