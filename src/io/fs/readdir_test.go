// Copyright 2020 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

package fs_test

import (
	. "io/fs"
	"testing"
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
