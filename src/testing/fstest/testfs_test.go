// Copyright 2021 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

package fstest

import (
	"internal/testenv"
	"os"
	"path/filepath"
	"testing"
)

func TestSymlink(t *testing.T) {
	testenv.MustHaveSymlink(t)

	tmp := t.TempDir()
	tmpfs := os.DirFS(tmp)

	if err := os.WriteFile(filepath.Join(tmp, "hello"), []byte("hello, world\n"), 0644); err != nil {
		t.Fatal(err)
	}

	if err := os.Symlink(filepath.Join(tmp, "hello"), filepath.Join(tmp, "hello.link")); err != nil {
		t.Fatal(err)
	}

	if err := TestFS(tmpfs, "hello", "hello.link"); err != nil {
		t.Fatal(err)
	}
}

func TestDash(t *testing.T) {
	m := MapFS{
		"a-b/a": {Data: []byte("a-b/a")},
	}
	if err := TestFS(m, "a-b/a"); err != nil {
		t.Error(err)
	}
}
