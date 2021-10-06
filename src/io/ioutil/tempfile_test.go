// Copyright 2010 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

package ioutil_test

import (
	"io/fs"
	. "io/ioutil"
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

// This string is from os.errPatternHasSeparator.
const patternHasSeparator = "pattern contains path separator"

func TestTempFile_BadPattern(t *testing.T) {
	tmpDir, err := TempDir("", t.Name())
	if err != nil {
		t.Fatal(err)
	}
	defer os.RemoveAll(tmpDir)

	const sep = string(os.PathSeparator)
	tests := []struct {
		pattern string
		wantErr bool
	}{
		{"ioutil*test", false},
		{"ioutil_test*foo", false},
		{"ioutil_test" + sep + "foo", true},
		{"ioutil_test*" + sep + "foo", true},
		{"ioutil_test" + sep + "*foo", true},
		{sep + "ioutil_test" + sep + "*foo", true},
		{"ioutil_test*foo" + sep, true},
	}
	for _, tt := range tests {
		t.Run(tt.pattern, func(t *testing.T) {
			tmpfile, err := TempFile(tmpDir, tt.pattern)
			defer func() {
				if tmpfile != nil {
					tmpfile.Close()
				}
			}()
			if tt.wantErr {
				if err == nil {
					t.Errorf("Expected an error for pattern %q", tt.pattern)
				} else if !strings.Contains(err.Error(), patternHasSeparator) {
					t.Errorf("Error mismatch: got %#v, want %q for pattern %q", err, patternHasSeparator, tt.pattern)
				}
			} else if err != nil {
				t.Errorf("Unexpected error %v for pattern %q", err, tt.pattern)
			}
		})
	}
}

func TestTempDir(t *testing.T) {
	name, err := TempDir("/_not_exists_", "foo")
	if name != "" || err == nil {
		t.Errorf("TempDir(`/_not_exists_`, `foo`) = %v, %v", name, err)
	}

	tests := []struct {
		pattern                string
		wantPrefix, wantSuffix string
	}{
		{"ioutil_test", "ioutil_test", ""},
		{"ioutil_test*", "ioutil_test", ""},
		{"ioutil_test*xyz", "ioutil_test", "xyz"},
	}

	dir := os.TempDir()

	runTestTempDir := func(t *testing.T, pattern, wantRePat string) {
		name, err := TempDir(dir, pattern)
		if name == "" || err != nil {
			t.Fatalf("TempDir(dir, `ioutil_test`) = %v, %v", name, err)
		}
		defer os.Remove(name)

		re := regexp.MustCompile(wantRePat)
		if !re.MatchString(name) {
			t.Errorf("TempDir(%q, %q) created bad name\n\t%q\ndid not match pattern\n\t%q", dir, pattern, name, wantRePat)
		}
	}

	for _, tt := range tests {
		t.Run(tt.pattern, func(t *testing.T) {
			wantRePat := "^" + regexp.QuoteMeta(filepath.Join(dir, tt.wantPrefix)) + "[0-9]+" + regexp.QuoteMeta(tt.wantSuffix) + "$"
			runTestTempDir(t, tt.pattern, wantRePat)
		})
	}

	// Separately testing "*xyz" (which has no prefix). That is when constructing the
	// pattern to assert on, as in the previous loop, using filepath.Join for an empty
	// prefix filepath.Join(dir, ""), produces the pattern:
	//     ^<DIR>[0-9]+xyz$
	// yet we just want to match
	//     "^<DIR>/[0-9]+xyz"
	t.Run("*xyz", func(t *testing.T) {
		wantRePat := "^" + regexp.QuoteMeta(filepath.Join(dir)) + regexp.QuoteMeta(string(filepath.Separator)) + "[0-9]+xyz$"
		runTestTempDir(t, "*xyz", wantRePat)
	})
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
	if pe, ok := err.(*fs.PathError); !ok || !os.IsNotExist(err) || pe.Path != badDir {
		t.Errorf("TempDir error = %#v; want PathError for path %q satisifying os.IsNotExist", err, badDir)
	}
}

func TestTempDir_BadPattern(t *testing.T) {
	tmpDir, err := TempDir("", t.Name())
	if err != nil {
		t.Fatal(err)
	}
	defer os.RemoveAll(tmpDir)

	const sep = string(os.PathSeparator)
	tests := []struct {
		pattern string
		wantErr bool
	}{
		{"ioutil*test", false},
		{"ioutil_test*foo", false},
		{"ioutil_test" + sep + "foo", true},
		{"ioutil_test*" + sep + "foo", true},
		{"ioutil_test" + sep + "*foo", true},
		{sep + "ioutil_test" + sep + "*foo", true},
		{"ioutil_test*foo" + sep, true},
	}
	for _, tt := range tests {
		t.Run(tt.pattern, func(t *testing.T) {
			_, err := TempDir(tmpDir, tt.pattern)
			if tt.wantErr {
				if err == nil {
					t.Errorf("Expected an error for pattern %q", tt.pattern)
				} else if !strings.Contains(err.Error(), patternHasSeparator) {
					t.Errorf("Error mismatch: got %#v, want %q for pattern %q", err, patternHasSeparator, tt.pattern)
				}
			} else if err != nil {
				t.Errorf("Unexpected error %v for pattern %q", err, tt.pattern)
			}
		})
	}
}
