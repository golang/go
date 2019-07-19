// Copyright 2018 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

package gopathwalk

import (
	"io/ioutil"
	"os"
	"path/filepath"
	"reflect"
	"runtime"
	"strings"
	"sync"
	"testing"
)

func TestShouldTraverse(t *testing.T) {
	switch runtime.GOOS {
	case "windows", "plan9":
		t.Skipf("skipping symlink-requiring test on %s", runtime.GOOS)
	}

	dir, err := ioutil.TempDir("", "goimports-")
	if err != nil {
		t.Fatal(err)
	}
	defer os.RemoveAll(dir)

	// Note: mapToDir prepends "src" to each element, since
	// mapToDir was made for creating GOPATHs.
	if err := mapToDir(dir, map[string]string{
		"foo/foo2/file.txt":        "",
		"foo/foo2/link-to-src":     "LINK:../../",
		"foo/foo2/link-to-src-foo": "LINK:../../foo",
		"foo/foo2/link-to-dot":     "LINK:.",
		"bar/bar2/file.txt":        "",
		"bar/bar2/link-to-src-foo": "LINK:../../foo",

		"a/b/c": "LINK:../../a/d",
		"a/d/e": "LINK:../../a/b",
	}); err != nil {
		t.Fatal(err)
	}
	tests := []struct {
		dir  string
		file string
		want bool
	}{
		{
			dir:  "src/foo/foo2",
			file: "link-to-src-foo",
			want: false, // loop
		},
		{
			dir:  "src/foo/foo2",
			file: "link-to-src",
			want: false, // loop
		},
		{
			dir:  "src/foo/foo2",
			file: "link-to-dot",
			want: false, // loop
		},
		{
			dir:  "src/bar/bar2",
			file: "link-to-src-foo",
			want: true, // not a loop
		},
		{
			dir:  "src/a/b/c",
			file: "e",
			want: false, // loop: "e" is the same as "b".
		},
	}
	for i, tt := range tests {
		fi, err := os.Stat(filepath.Join(dir, tt.dir, tt.file))
		if err != nil {
			t.Errorf("%d. Stat = %v", i, err)
			continue
		}
		var w walker
		got := w.shouldTraverse(filepath.Join(dir, tt.dir), fi)
		if got != tt.want {
			t.Errorf("%d. shouldTraverse(%q, %q) = %v; want %v", i, tt.dir, tt.file, got, tt.want)
		}
	}
}

// TestSkip tests that various goimports rules are followed in non-modules mode.
func TestSkip(t *testing.T) {
	dir, err := ioutil.TempDir("", "goimports-")
	if err != nil {
		t.Fatal(err)
	}
	defer os.RemoveAll(dir)

	if err := mapToDir(dir, map[string]string{
		"ignoreme/f.go":     "package ignoreme",     // ignored by .goimportsignore
		"node_modules/f.go": "package nodemodules;", // ignored by hardcoded node_modules filter
		"v/f.go":            "package v;",           // ignored by hardcoded vgo cache rule
		"mod/f.go":          "package mod;",         // ignored by hardcoded vgo cache rule
		"shouldfind/f.go":   "package shouldfind;",  // not ignored

		".goimportsignore": "ignoreme\n",
	}); err != nil {
		t.Fatal(err)
	}

	var found []string
	var mu sync.Mutex
	walkDir(Root{filepath.Join(dir, "src"), RootGOPATH},
		func(root Root, dir string) {
			mu.Lock()
			defer mu.Unlock()
			found = append(found, dir[len(root.Path)+1:])
		}, func(root Root, dir string) bool {
			return false
		}, Options{ModulesEnabled: false, Debug: true})
	if want := []string{"shouldfind"}; !reflect.DeepEqual(found, want) {
		t.Errorf("expected to find only %v, got %v", want, found)
	}
}

// TestSkipFunction tests that scan successfully skips directories from user callback.
func TestSkipFunction(t *testing.T) {
	dir, err := ioutil.TempDir("", "goimports-")
	if err != nil {
		t.Fatal(err)
	}
	defer os.RemoveAll(dir)

	if err := mapToDir(dir, map[string]string{
		"ignoreme/f.go":           "package ignoreme",    // ignored by skip
		"ignoreme/subignore/f.go": "package subignore",   // also ignored by skip
		"shouldfind/f.go":         "package shouldfind;", // not ignored
	}); err != nil {
		t.Fatal(err)
	}

	var found []string
	var mu sync.Mutex
	walkDir(Root{filepath.Join(dir, "src"), RootGOPATH},
		func(root Root, dir string) {
			mu.Lock()
			defer mu.Unlock()
			found = append(found, dir[len(root.Path)+1:])
		}, func(root Root, dir string) bool {
			return strings.HasSuffix(dir, "ignoreme")
		},
		Options{ModulesEnabled: false})
	if want := []string{"shouldfind"}; !reflect.DeepEqual(found, want) {
		t.Errorf("expected to find only %v, got %v", want, found)
	}
}

func mapToDir(destDir string, files map[string]string) error {
	for path, contents := range files {
		file := filepath.Join(destDir, "src", path)
		if err := os.MkdirAll(filepath.Dir(file), 0755); err != nil {
			return err
		}
		var err error
		if strings.HasPrefix(contents, "LINK:") {
			err = os.Symlink(strings.TrimPrefix(contents, "LINK:"), file)
		} else {
			err = ioutil.WriteFile(file, []byte(contents), 0644)
		}
		if err != nil {
			return err
		}
	}
	return nil
}
