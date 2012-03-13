// Copyright 2012 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

package os_test

import (
	"io/ioutil"
	"os"
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
	if !os.IsExist(err) {
		t.Fatalf("os.IsExist does not work as expected for %#v", err)
		return
	}
}
