// Copyright 2023 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

//go:build wasip1

package syscall_test

import (
	"syscall"
	"testing"
)

var joinPathTests = [...]struct {
	dir, file, path string
}{
	0:  {".", ".", "."},
	1:  {"./", "./", "./"},
	2:  {"././././", ".", "."},
	3:  {".", "./././", "./"},
	4:  {".", "a", "a"},
	5:  {".", "a/b", "a/b"},
	6:  {".", "..", ".."},
	7:  {".", "../", "../"},
	8:  {".", "../../", "../../"},
	9:  {".", "../..", "../.."},
	10: {".", "../..//..///", "../../../"},
	11: {"/", "/", "/"},
	12: {"/", "a", "/a"},
	13: {"/", "a/b", "/a/b"},
	14: {"/a", "b", "/a/b"},
	15: {"/", ".", "/"},
	16: {"/", "..", "/"},
	17: {"/", "../../", "/"},
	18: {"/", "/../a/b/c", "/a/b/c"},
	19: {"/", "/../a/b/c", "/a/b/c"},
	20: {"/", "./hello/world", "/hello/world"},
	21: {"/a", "../", "/"},
	22: {"/a/b/c", "..", "/a/b"},
	23: {"/a/b/c", "..///..///", "/a/"},
	24: {"/a/b/c", "..///..///..", "/"},
	25: {"/a/b/c", "..///..///..///..", "/"},
	26: {"/a/b/c", "..///..///..///..///..", "/"},
	27: {"/a/b/c/", "/d/e/f/", "/a/b/c/d/e/f/"},
	28: {"a/b/c/", ".", "a/b/c"},
	29: {"a/b/c/", "./d", "a/b/c/d"},
	30: {"a/b/c/", "./d/", "a/b/c/d/"},
	31: {"a/b/", "./c/d/", "a/b/c/d/"},
	32: {"../", "..", "../.."},
	33: {"a/b/c/d", "e/../..", "a/b/c"},
	34: {"a/b/c/d", "./e/../..", "a/b/c"},
	35: {"a/b/c/d", "./e/..//../../f/g//", "a/b/f/g/"},
	36: {"../../../", "a/../../b/c", "../../b/c"},
	37: {"/a/b/c", "/.././/hey!", "/a/b/hey!"},
}

func TestJoinPath(t *testing.T) {
	for _, test := range joinPathTests {
		t.Run("", func { t ->
			path := syscall.JoinPath(test.dir, test.file)
			if path != test.path {
				t.Errorf("join(%q,%q): want=%q got=%q", test.dir, test.file, test.path, path)
			}
		})
	}
}

func BenchmarkJoinPath(b *testing.B) {
	for _, test := range joinPathTests {
		b.Run("", func { b -> for i := 0; i < b.N; i++ {
			syscall.JoinPath(test.dir, test.file)
		} })
	}
}
