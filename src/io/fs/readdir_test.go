// Copyright 2020 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

package fs_test

import (
	"errors"
	. "io/fs"
	"os"
	"testing"
	"testing/fstest"
	"time"
)

type readDirOnly struct{ ReadDirFS }

func (readDirOnly) Open(name string) (File, error) { return nil, ErrNotExist }

func TestReadDir(t *testing.T) {
	check := func(desc string, dirs []DirEntry, err error) {
		t.Helper()
		if err != nil || len(dirs) != 2 || dirs[0].Name() != "hello.txt" || dirs[1].Name() != "sub" {
			var names []string
			for _, d := range dirs {
				names = append(names, d.Name())
			}
			t.Errorf("ReadDir(%s) = %v, %v, want %v, nil", desc, names, err, []string{"hello.txt", "sub"})
		}
	}

	// Test that ReadDir uses the method when present.
	dirs, err := ReadDir(readDirOnly{testFsys}, ".")
	check("readDirOnly", dirs, err)

	// Test that ReadDir uses Open when the method is not present.
	dirs, err = ReadDir(openOnly{testFsys}, ".")
	check("openOnly", dirs, err)

	// Test that ReadDir on Sub of . works (sub_test checks non-trivial subs).
	sub, err := Sub(testFsys, ".")
	if err != nil {
		t.Fatal(err)
	}
	dirs, err = ReadDir(sub, ".")
	check("sub(.)", dirs, err)
}

func TestFileInfoToDirEntry(t *testing.T) {
	testFs := fstest.MapFS{
		"notadir.txt": {
			Data:    []byte("hello, world"),
			Mode:    0,
			ModTime: time.Now(),
			Sys:     &sysValue,
		},
		"adir": {
			Data:    nil,
			Mode:    os.ModeDir,
			ModTime: time.Now(),
			Sys:     &sysValue,
		},
	}

	tests := []struct {
		path     string
		wantMode FileMode
		wantDir  bool
	}{
		{path: "notadir.txt", wantMode: 0, wantDir: false},
		{path: "adir", wantMode: os.ModeDir, wantDir: true},
	}

	for _, test := range tests {
		test := test
		t.Run(test.path, func(t *testing.T) {
			fi, err := Stat(testFs, test.path)
			if err != nil {
				t.Fatal(err)
			}

			dirEntry := FileInfoToDirEntry(fi)
			if g, w := dirEntry.Type(), test.wantMode; g != w {
				t.Errorf("FileMode mismatch: got=%v, want=%v", g, w)
			}
			if g, w := dirEntry.Name(), test.path; g != w {
				t.Errorf("Name mismatch: got=%v, want=%v", g, w)
			}
			if g, w := dirEntry.IsDir(), test.wantDir; g != w {
				t.Errorf("IsDir mismatch: got=%v, want=%v", g, w)
			}
		})
	}
}

func errorPath(err error) string {
	var perr *PathError
	if !errors.As(err, &perr) {
		return ""
	}
	return perr.Path
}

func TestReadDirPath(t *testing.T) {
	fsys := os.DirFS(t.TempDir())
	_, err1 := ReadDir(fsys, "non-existent")
	_, err2 := ReadDir(struct{ FS }{fsys}, "non-existent")
	if s1, s2 := errorPath(err1), errorPath(err2); s1 != s2 {
		t.Fatalf("s1: %s != s2: %s", s1, s2)
	}
}
