// Copyright 2012 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

package os_test

import (
	"fmt"
	"io/ioutil"
	"os"
	"path/filepath"
	"testing"
)

func TestErrIsExist(t *testing.T) {
	f, err := ioutil.TempFile("", "_Go_ErrIsExist")
	if err != nil {
		t.Fatalf("open ErrIsExist tempfile: %s", err)
		return
	}
	defer os.Remove(f.Name())
	defer f.Close()
	f2, err := os.OpenFile(f.Name(), os.O_RDWR|os.O_CREATE|os.O_EXCL, 0600)
	if err == nil {
		f2.Close()
		t.Fatal("Open should have failed")
		return
	}
	if s := checkErrorPredicate("os.IsExist", os.IsExist, err); s != "" {
		t.Fatal(s)
		return
	}
}

func testErrNotExist(name string) string {
	f, err := os.Open(name)
	if err == nil {
		f.Close()
		return "Open should have failed"
	}
	if s := checkErrorPredicate("os.IsNotExist", os.IsNotExist, err); s != "" {
		return s
	}

	err = os.Chdir(name)
	if err == nil {
		return "Chdir should have failed"
	}
	if s := checkErrorPredicate("os.IsNotExist", os.IsNotExist, err); s != "" {
		return s
	}
	return ""
}

func TestErrIsNotExist(t *testing.T) {
	tmpDir, err := ioutil.TempDir("", "_Go_ErrIsNotExist")
	if err != nil {
		t.Fatalf("create ErrIsNotExist tempdir: %s", err)
		return
	}
	defer os.RemoveAll(tmpDir)

	name := filepath.Join(tmpDir, "NotExists")
	if s := testErrNotExist(name); s != "" {
		t.Fatal(s)
		return
	}

	name = filepath.Join(name, "NotExists2")
	if s := testErrNotExist(name); s != "" {
		t.Fatal(s)
		return
	}
}

func checkErrorPredicate(predName string, pred func(error) bool, err error) string {
	if !pred(err) {
		return fmt.Sprintf("%s does not work as expected for %#v", predName, err)
	}
	return ""
}
