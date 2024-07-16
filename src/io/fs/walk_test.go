// Copyright 2020 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

package fs_test

import (
	. "io/fs"
	"os"
	pathpkg "path"
	"path/filepath"
	"slices"
	"testing"
	"testing/fstest"
)

type Node struct {
	name    string
	entries []*Node // nil if the entry is a file
	mark    int
}

var tree = &Node{
	"testdata",
	[]*Node{
		{"a", nil, 0},
		{"b", []*Node{}, 0},
		{"c", nil, 0},
		{
			"d",
			[]*Node{
				{"x", nil, 0},
				{"y", []*Node{}, 0},
				{
					"z",
					[]*Node{
						{"u", nil, 0},
						{"v", nil, 0},
					},
					0,
				},
			},
			0,
		},
	},
	0,
}

func walkTree(n *Node, path string, f func(path string, n *Node)) {
	f(path, n)
	for _, e := range n.entries {
		walkTree(e, pathpkg.Join(path, e.name), f)
	}
}

func makeTree() FS {
	fsys := fstest.MapFS{}
	walkTree(tree, tree.name, func(path string, n *Node) {
		if n.entries == nil {
			fsys[path] = &fstest.MapFile{}
		} else {
			fsys[path] = &fstest.MapFile{Mode: ModeDir}
		}
	})
	return fsys
}

// Assumes that each node name is unique. Good enough for a test.
// If clear is true, any incoming error is cleared before return. The errors
// are always accumulated, though.
func mark(entry DirEntry, err error, errors *[]error, clear bool) error {
	name := entry.Name()
	walkTree(tree, tree.name, func(path string, n *Node) {
		if n.name == name {
			n.mark++
		}
	})
	if err != nil {
		*errors = append(*errors, err)
		if clear {
			return nil
		}
		return err
	}
	return nil
}

func TestWalkDir(t *testing.T) {
	t.Chdir(t.TempDir())

	fsys := makeTree()
	errors := make([]error, 0, 10)
	clear := true
	markFn := func(path string, entry DirEntry, err error) error {
		return mark(entry, err, &errors, clear)
	}
	// Expect no errors.
	err := WalkDir(fsys, ".", markFn)
	if err != nil {
		t.Fatalf("no error expected, found: %s", err)
	}
	if len(errors) != 0 {
		t.Fatalf("unexpected errors: %s", errors)
	}
	walkTree(tree, tree.name, func(path string, n *Node) {
		if n.mark != 1 {
			t.Errorf("node %s mark = %d; expected 1", path, n.mark)
		}
		n.mark = 0
	})
}

func TestWalkDirSymlink(t *testing.T) {
	fsys := fstest.MapFS{
		"link":    {Data: []byte("dir"), Mode: ModeSymlink},
		"dir/a":   {},
		"dir/b/c": {},
		"dir/d":   {Data: []byte("b"), Mode: ModeSymlink},
	}

	wantTypes := map[string]FileMode{
		"link":     ModeDir,
		"link/a":   0,
		"link/b":   ModeDir,
		"link/b/c": 0,
		"link/d":   ModeSymlink,
	}
	marks := make(map[string]int)
	walkFn := func(path string, entry DirEntry, err error) error {
		marks[path]++
		if want, ok := wantTypes[path]; !ok {
			t.Errorf("Unexpected path %q in walk", path)
		} else if got := entry.Type(); got != want {
			t.Errorf("%s entry type = %v; want %v", path, got, want)
		}
		if err != nil {
			t.Errorf("Walking %s: %v", path, err)
		}
		return nil
	}

	// Expect no errors.
	err := WalkDir(fsys, "link", walkFn)
	if err != nil {
		t.Fatalf("no error expected, found: %s", err)
	}
	for path := range wantTypes {
		if got := marks[path]; got != 1 {
			t.Errorf("%s visited %d times; expected 1", path, got)
		}
	}
}

func TestIssue51617(t *testing.T) {
	dir := t.TempDir()
	for _, sub := range []string{"a", filepath.Join("a", "bad"), filepath.Join("a", "next")} {
		if err := os.Mkdir(filepath.Join(dir, sub), 0755); err != nil {
			t.Fatal(err)
		}
	}
	bad := filepath.Join(dir, "a", "bad")
	if err := os.Chmod(bad, 0); err != nil {
		t.Fatal(err)
	}
	defer os.Chmod(bad, 0700) // avoid errors on cleanup
	var saw []string
	err := WalkDir(os.DirFS(dir), ".", func(path string, d DirEntry, err error) error {
		if err != nil {
			return filepath.SkipDir
		}
		if d.IsDir() {
			saw = append(saw, path)
		}
		return nil
	})
	if err != nil {
		t.Fatal(err)
	}
	want := []string{".", "a", "a/bad", "a/next"}
	if !slices.Equal(saw, want) {
		t.Errorf("got directories %v, want %v", saw, want)
	}
}
