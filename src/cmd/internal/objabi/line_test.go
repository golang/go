// Copyright 2019 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

package objabi

import (
	"path/filepath"
	"runtime"
	"testing"
)

// On Windows, "/foo" is reported as a relative path
// (it is relative to the current drive letter),
// so we need add a drive letter to test absolute path cases.
func drive() string {
	if runtime.GOOS == "windows" {
		return "c:"
	}
	return ""
}

var absFileTests = []struct {
	dir      string
	file     string
	rewrites string
	abs      string
}{
	{"/d", "f", "", "/d/f"},
	{"/d", drive() + "/f", "", drive() + "/f"},
	{"/d", "f/g", "", "/d/f/g"},
	{"/d", drive() + "/f/g", "", drive() + "/f/g"},

	{"/d", "f", "/d/f", "??"},
	{"/d", "f/g", "/d/f", "g"},
	{"/d", "f/g", "/d/f=>h", "h/g"},
	{"/d", "f/g", "/d/f=>/h", "/h/g"},
	{"/d", "f/g", "/d/f=>/h;/d/e=>/i", "/h/g"},
	{"/d", "e/f", "/d/f=>/h;/d/e=>/i", "/i/f"},
}

func TestAbsFile(t *testing.T) {
	for _, tt := range absFileTests {
		abs := filepath.FromSlash(AbsFile(filepath.FromSlash(tt.dir), filepath.FromSlash(tt.file), tt.rewrites))
		want := filepath.FromSlash(tt.abs)
		if abs != want {
			t.Errorf("AbsFile(%q, %q, %q) = %q, want %q", tt.dir, tt.file, tt.rewrites, abs, want)
		}
	}
}
