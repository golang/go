// Copyright 2010 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

package ioutil_test

import (
	. "io/ioutil"
	"os"
	"path/filepath"
	"regexp"
	"testing"
)

func TestTempFile(t *testing.T) {
	f, err := TempFile("/_not_exists_", "foo")
	if f != nil || err == nil {
		t.Errorf("TempFile(`/_not_exists_`, `foo`) = %v, %v", f, err)
	}

	dir := os.TempDir()
	f, err = TempFile(dir, "ioutil_test")
	if f == nil || err != nil {
		t.Errorf("TempFile(dir, `ioutil_test`) = %v, %v", f, err)
	}
	if f != nil {
		f.Close()
		os.Remove(f.Name())
		re := regexp.MustCompile("^" + regexp.QuoteMeta(filepath.Join(dir, "ioutil_test")) + "[0-9]+$")
		if !re.MatchString(f.Name()) {
			t.Errorf("TempFile(`"+dir+"`, `ioutil_test`) created bad name %s", f.Name())
		}
	}
}

func TestTempDir(t *testing.T) {
	name, err := TempDir("/_not_exists_", "foo")
	if name != "" || err == nil {
		t.Errorf("TempDir(`/_not_exists_`, `foo`) = %v, %v", name, err)
	}

	dir := os.TempDir()
	name, err = TempDir(dir, "ioutil_test")
	if name == "" || err != nil {
		t.Errorf("TempDir(dir, `ioutil_test`) = %v, %v", name, err)
	}
	if name != "" {
		os.Remove(name)
		re := regexp.MustCompile("^" + regexp.QuoteMeta(filepath.Join(dir, "ioutil_test")) + "[0-9]+$")
		if !re.MatchString(name) {
			t.Errorf("TempDir(`"+dir+"`, `ioutil_test`) created bad name %s", name)
		}
	}
}
