// Copyright 2010 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

package ioutil

import (
	"os"
	"path/filepath"
	"regexp"
	"strings"
	"testing"
)

func TestTempFile(t *testing.T) {
	dir, err := TempDir("", "TestTempFile_BadDir")
	if err != nil {
		t.Fatal(err)
	}
	defer os.RemoveAll(dir)

	nonexistentDir := filepath.Join(dir, "_not_exists_")
	f, err := TempFile(nonexistentDir, "foo")
	if f != nil || err == nil {
		t.Errorf("TempFile(%q, `foo`) = %v, %v", nonexistentDir, f, err)
	}
}

func TestTempFile_pattern(t *testing.T) {
	tests := []struct{ pattern, prefix, suffix string }{
		{"ioutil_test", "ioutil_test", ""},
		{"ioutil_test*", "ioutil_test", ""},
		{"ioutil_test*xyz", "ioutil_test", "xyz"},
	}
	for _, test := range tests {
		f, err := TempFile("", test.pattern)
		if err != nil {
			t.Errorf("TempFile(..., %q) error: %v", test.pattern, err)
			continue
		}
		defer os.Remove(f.Name())
		base := filepath.Base(f.Name())
		f.Close()
		if !(strings.HasPrefix(base, test.prefix) && strings.HasSuffix(base, test.suffix)) {
			t.Errorf("TempFile pattern %q created bad name %q; want prefix %q & suffix %q",
				test.pattern, base, test.prefix, test.suffix)
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

// test that we return a nice error message if the dir argument to TempDir doesn't
// exist (or that it's empty and os.TempDir doesn't exist)
func TestTempDir_BadDir(t *testing.T) {
	dir, err := TempDir("", "TestTempDir_BadDir")
	if err != nil {
		t.Fatal(err)
	}
	defer os.RemoveAll(dir)

	badDir := filepath.Join(dir, "not-exist")
	_, err = TempDir(badDir, "foo")
	if pe, ok := err.(*os.PathError); !ok || !os.IsNotExist(err) || pe.Path != badDir {
		t.Errorf("TempDir error = %#v; want PathError for path %q satisifying os.IsNotExist", err, badDir)
	}
}
