// Copyright 2023 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

package os_test

import (
	"internal/testenv"
	"io/fs"
	. "os"
	"path/filepath"
	"testing"
)

func TestDirFSReadLink(t *testing.T) {
	testenv.MustHaveSymlink(t)

	root := t.TempDir()
	subdir := filepath.Join(root, "dir")
	if err := Mkdir(subdir, 0o777); err != nil {
		t.Fatal(err)
	}
	links := map[string]string{
		filepath.Join(root, "parent-link"):        filepath.Join("..", "foo"),
		filepath.Join(root, "sneaky-parent-link"): filepath.Join("dir", "..", "..", "foo"),
		filepath.Join(root, "abs-link"):           filepath.Join(root, "foo"),
		filepath.Join(root, "rel-link"):           "foo",
		filepath.Join(root, "rel-sub-link"):       filepath.Join("dir", "foo"),
		filepath.Join(subdir, "parent-link"):      filepath.Join("..", "foo"),
	}
	for newname, oldname := range links {
		if err := Symlink(oldname, newname); err != nil {
			t.Fatal(err)
		}
	}

	fsys := DirFS(root)
	want := map[string]string{
		"rel-link":           "foo",
		"rel-sub-link":       filepath.Join("dir", "foo"),
		"dir/parent-link":    filepath.Join("..", "foo"),
		"parent-link":        filepath.Join("..", "foo"),
		"sneaky-parent-link": filepath.Join("dir", "..", "..", "foo"),
		"abs-link":           filepath.Join(root, "foo"),
	}
	for name, want := range want {
		got, err := fs.ReadLink(fsys, name)
		if got != want || err != nil {
			t.Errorf("fs.ReadLink(fsys, %q) = %q, %v; want %q, <nil>", name, got, err, want)
		}
	}
}

func TestDirFSLstat(t *testing.T) {
	testenv.MustHaveSymlink(t)

	root := t.TempDir()
	subdir := filepath.Join(root, "dir")
	if err := Mkdir(subdir, 0o777); err != nil {
		t.Fatal(err)
	}
	if err := Symlink("dir", filepath.Join(root, "link")); err != nil {
		t.Fatal(err)
	}

	fsys := DirFS(root)
	want := map[string]fs.FileMode{
		"link": fs.ModeSymlink,
		"dir":  fs.ModeDir,
	}
	for name, want := range want {
		info, err := fs.Lstat(fsys, name)
		var got fs.FileMode
		if info != nil {
			got = info.Mode().Type()
		}
		if got != want || err != nil {
			t.Errorf("fs.Lstat(fsys, %q).Mode().Type() = %v, %v; want %v, <nil>", name, got, err, want)
		}
	}
}

func TestDirFSWalkDir(t *testing.T) {
	testenv.MustHaveSymlink(t)

	root := t.TempDir()
	subdir := filepath.Join(root, "dir")
	if err := Mkdir(subdir, 0o777); err != nil {
		t.Fatal(err)
	}
	if err := Symlink("dir", filepath.Join(root, "link")); err != nil {
		t.Fatal(err)
	}
	if err := WriteFile(filepath.Join(root, "dir", "a"), nil, 0o666); err != nil {
		t.Fatal(err)
	}
	fsys := DirFS(root)

	t.Run("SymlinkRoot", func(t *testing.T) {
		wantTypes := map[string]fs.FileMode{
			"link":   fs.ModeDir,
			"link/a": 0,
		}
		marks := make(map[string]int)
		err := fs.WalkDir(fsys, "link", func(path string, entry fs.DirEntry, err error) error {
			marks[path]++
			if want, ok := wantTypes[path]; !ok {
				t.Errorf("Unexpected path %q in walk", path)
			} else if got := entry.Type(); got != want {
				t.Errorf("%s entry type = %v; want %v", path, got, want)
			}
			if err != nil {
				t.Errorf("%s: %v", path, err)
			}
			return nil
		})
		if err != nil {
			t.Fatal(err)
		}
		for path := range wantTypes {
			if got := marks[path]; got != 1 {
				t.Errorf("%s visited %d times; expected 1", path, got)
			}
		}
	})

	t.Run("SymlinkPresent", func(t *testing.T) {
		wantTypes := map[string]fs.FileMode{
			".":     fs.ModeDir,
			"dir":   fs.ModeDir,
			"dir/a": 0,
			"link":  fs.ModeSymlink,
		}
		marks := make(map[string]int)
		err := fs.WalkDir(fsys, ".", func(path string, entry fs.DirEntry, err error) error {
			marks[path]++
			if want, ok := wantTypes[path]; !ok {
				t.Errorf("Unexpected path %q in walk", path)
			} else if got := entry.Type(); got != want {
				t.Errorf("%s entry type = %v; want %v", path, got, want)
			}
			if err != nil {
				t.Errorf("%s: %v", path, err)
			}
			return nil
		})
		if err != nil {
			t.Fatal(err)
		}
		for path := range wantTypes {
			if got := marks[path]; got != 1 {
				t.Errorf("%s visited %d times; expected 1", path, got)
			}
		}
	})
}
