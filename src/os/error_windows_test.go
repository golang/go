// Copyright 2012 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

package os_test

import (
	"io/ioutil"
	"os"
	"path/filepath"
	"testing"
)

func TestErrIsExistAfterRename(t *testing.T) {
	dir, err := ioutil.TempDir("", "go-build")
	if err != nil {
		t.Fatalf("Create temp directory: %v", err)
	}
	defer os.RemoveAll(dir)

	src := filepath.Join(dir, "src")
	dest := filepath.Join(dir, "dest")

	f, err := os.Create(src)
	if err != nil {
		t.Fatalf("Create file %v: %v", src, err)
	}
	f.Close()
	err = os.Rename(src, dest)
	if err != nil {
		t.Fatalf("Rename %v to %v: %v", src, dest, err)
	}

	f, err = os.Create(src)
	if err != nil {
		t.Fatalf("Create file %v: %v", src, err)
	}
	f.Close()
	err = os.Rename(src, dest)
	if err == nil {
		t.Fatal("Rename should have failed")
	}
	if s := checkErrorPredicate("os.IsExist", os.IsExist, err); s != "" {
		t.Fatal(s)
		return
	}
}
