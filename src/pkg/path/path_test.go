// Copyright 2009 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

package path

import (
	"testing"
)

type CleanTest struct {
	path, clean string
}

var cleantests = []CleanTest {
	// Already clean
	CleanTest{"", "."},
	CleanTest{"abc", "abc"},
	CleanTest{"abc/def", "abc/def"},
	CleanTest{"a/b/c", "a/b/c"},
	CleanTest{".", "."},
	CleanTest{"..", ".."},
	CleanTest{"../..", "../.."},
	CleanTest{"../../abc", "../../abc"},
	CleanTest{"/abc", "/abc"},
	CleanTest{"/", "/"},

	// Remove trailing slash
	CleanTest{"abc/", "abc"},
	CleanTest{"abc/def/", "abc/def"},
	CleanTest{"a/b/c/", "a/b/c"},
	CleanTest{"./", "."},
	CleanTest{"../", ".."},
	CleanTest{"../../", "../.."},
	CleanTest{"/abc/", "/abc"},

	// Remove doubled slash
	CleanTest{"abc//def//ghi", "abc/def/ghi"},
	CleanTest{"//abc", "/abc"},
	CleanTest{"///abc", "/abc"},
	CleanTest{"//abc//", "/abc"},
	CleanTest{"abc//", "abc"},

	// Remove . elements
	CleanTest{"abc/./def", "abc/def"},
	CleanTest{"/./abc/def", "/abc/def"},
	CleanTest{"abc/.", "abc"},

	// Remove .. elements
	CleanTest{"abc/def/ghi/../jkl", "abc/def/jkl"},
	CleanTest{"abc/def/../ghi/../jkl", "abc/jkl"},
	CleanTest{"abc/def/..", "abc"},
	CleanTest{"abc/def/../..", "."},
	CleanTest{"/abc/def/../..", "/"},
	CleanTest{"abc/def/../../..", ".."},
	CleanTest{"/abc/def/../../..", "/"},
	CleanTest{"abc/def/../../../ghi/jkl/../../../mno", "../../mno"},

	// Combinations
	CleanTest{"abc/./../def", "def"},
	CleanTest{"abc//./../def", "def"},
	CleanTest{"abc/../../././../def", "../../def"},
}

func TestClean(t *testing.T) {
	for i, test := range cleantests {
		if s := Clean(test.path); s != test.clean {
			t.Errorf("Clean(%q) = %q, want %q", test.path, s, test.clean);
		}
	}
}

type SplitTest struct {
	path, dir, file string
}

var splittests = []SplitTest {
	SplitTest{"a/b", "a/", "b"},
	SplitTest{"a/b/", "a/b/", ""},
	SplitTest{"a/", "a/", ""},
	SplitTest{"a", "", "a"},
	SplitTest{"/", "/", ""},
}

func TestSplit(t *testing.T) {
	for i, test := range splittests {
		if d, f := Split(test.path); d != test.dir || f != test.file {
			t.Errorf("Split(%q) = %q, %q, want %q, %q", test.path, d, f, test.dir, test.file);
		}
	}
}

type JoinTest struct {
	dir, file, path string
}

var jointests = []JoinTest {
	JoinTest{"a", "b", "a/b"},
	JoinTest{"a", "", "a"},
	JoinTest{"", "b", "b"},
	JoinTest{"/", "a", "/a"},
	JoinTest{"/", "", "/"},
	JoinTest{"a/", "b", "a/b"},
	JoinTest{"a/", "", "a"},
}

func TestJoin(t *testing.T) {
	for i, test := range jointests {
		if p := Join(test.dir, test.file); p != test.path {
			t.Errorf("Join(%q, %q) = %q, want %q", test.dir, test.file, p, test.path);
		}
	}
}

type ExtTest struct {
	path, ext string
}

var exttests = []ExtTest {
	ExtTest{"path.go", ".go"},
	ExtTest{"path.pb.go", ".go"},
	ExtTest{"a.dir/b", ""},
	ExtTest{"a.dir/b.go", ".go"},
	ExtTest{"a.dir/", ""},
}

func TestExt(t *testing.T) {
	for i, test := range exttests {
		if x := Ext(test.path); x != test.ext {
			t.Errorf("Ext(%q) = %q, want %q", test.path, x, test.ext);
		}
	}
}

