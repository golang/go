// Copyright 2020 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

package fs_test

import (
	. "io/fs"
	"testing"
)

type subOnly struct{ SubFS }

func (subOnly) Open(name string) (File, error) { return nil, ErrNotExist }

func TestSub(t *testing.T) {
	check := func(desc string, sub FS, err error) {
		t.Helper()
		if err != nil {
			t.Errorf("Sub(sub): %v", err)
			return
		}
		data, err := ReadFile(sub, "goodbye.txt")
		if string(data) != "goodbye, world" || err != nil {
			t.Errorf(`ReadFile(%s, "goodbye.txt" = %q, %v, want %q, nil`, desc, string(data), err, "goodbye, world")
		}

		dirs, err := ReadDir(sub, ".")
		if err != nil || len(dirs) != 1 || dirs[0].Name() != "goodbye.txt" {
			var names []string
			for _, d := range dirs {
				names = append(names, d.Name())
			}
			t.Errorf(`ReadDir(%s, ".") = %v, %v, want %v, nil`, desc, names, err, []string{"goodbye.txt"})
		}
	}

	// Test that Sub uses the method when present.
	sub, err := Sub(subOnly{testFsys}, "sub")
	check("subOnly", sub, err)

	// Test that Sub uses Open when the method is not present.
	sub, err = Sub(openOnly{testFsys}, "sub")
	check("openOnly", sub, err)

	_, err = sub.Open("nonexist")
	if err == nil {
		t.Fatal("Open(nonexist): succeeded")
	}
	pe, ok := err.(*PathError)
	if !ok {
		t.Fatalf("Open(nonexist): error is %T, want *PathError", err)
	}
	if pe.Path != "nonexist" {
		t.Fatalf("Open(nonexist): err.Path = %q, want %q", pe.Path, "nonexist")
	}
}
