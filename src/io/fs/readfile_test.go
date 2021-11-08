// Copyright 2020 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

package fs_test

import (
	. "io/fs"
	"testing"
	"testing/fstest"
	"time"
)

var testFsys = fstest.MapFS{
	"hello.txt": {
		Data:    []byte("hello, world"),
		Mode:    0456,
		ModTime: time.Now(),
		Sys:     &sysValue,
	},
	"sub/goodbye.txt": {
		Data:    []byte("goodbye, world"),
		Mode:    0456,
		ModTime: time.Now(),
		Sys:     &sysValue,
	},
}

var sysValue int

type readFileOnly struct{ ReadFileFS }

func (readFileOnly) Open(name string) (File, error) { return nil, ErrNotExist }

type openOnly struct{ FS }

func TestReadFile(t *testing.T) {
	// Test that ReadFile uses the method when present.
	data, err := ReadFile(readFileOnly{testFsys}, "hello.txt")
	if string(data) != "hello, world" || err != nil {
		t.Fatalf(`ReadFile(readFileOnly, "hello.txt") = %q, %v, want %q, nil`, data, err, "hello, world")
	}

	// Test that ReadFile uses Open when the method is not present.
	data, err = ReadFile(openOnly{testFsys}, "hello.txt")
	if string(data) != "hello, world" || err != nil {
		t.Fatalf(`ReadFile(openOnly, "hello.txt") = %q, %v, want %q, nil`, data, err, "hello, world")
	}

	// Test that ReadFile on Sub of . works (sub_test checks non-trivial subs).
	sub, err := Sub(testFsys, ".")
	if err != nil {
		t.Fatal(err)
	}
	data, err = ReadFile(sub, "hello.txt")
	if string(data) != "hello, world" || err != nil {
		t.Fatalf(`ReadFile(sub(.), "hello.txt") = %q, %v, want %q, nil`, data, err, "hello, world")
	}
}
