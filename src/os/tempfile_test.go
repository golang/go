// Copyright 2010 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

package os_test

import (
	"errors"
	"io/fs"
	. "os"
	"path/filepath"
	"regexp"
	"strings"
	"testing"
)

func TestCreateTemp(t *testing.T) {
	t.Parallel()

	dir, err := MkdirTemp("", "TestCreateTempBadDir")
	if err != nil {
		t.Fatal(err)
	}
	defer RemoveAll(dir)

	nonexistentDir := filepath.Join(dir, "_not_exists_")
	f, err := CreateTemp(nonexistentDir, "foo")
	if f != nil || err == nil {
		t.Errorf("CreateTemp(%q, `foo`) = %v, %v", nonexistentDir, f, err)
	}
}

func TestCreateTempPattern(t *testing.T) {
	t.Parallel()

	tests := []struct{ pattern, prefix, suffix string }{
		{"tempfile_test", "tempfile_test", ""},
		{"tempfile_test*", "tempfile_test", ""},
		{"tempfile_test*xyz", "tempfile_test", "xyz"},
	}
	for _, test := range tests {
		f, err := CreateTemp("", test.pattern)
		if err != nil {
			t.Errorf("CreateTemp(..., %q) error: %v", test.pattern, err)
			continue
		}
		defer Remove(f.Name())
		base := filepath.Base(f.Name())
		f.Close()
		if !(strings.HasPrefix(base, test.prefix) && strings.HasSuffix(base, test.suffix)) {
			t.Errorf("CreateTemp pattern %q created bad name %q; want prefix %q & suffix %q",
				test.pattern, base, test.prefix, test.suffix)
		}
	}
}

func TestCreateTempBadPattern(t *testing.T) {
	t.Parallel()

	tmpDir, err := MkdirTemp("", t.Name())
	if err != nil {
		t.Fatal(err)
	}
	defer RemoveAll(tmpDir)

	const sep = string(PathSeparator)
	tests := []struct {
		pattern string
		wantErr bool
	}{
		{"ioutil*test", false},
		{"tempfile_test*foo", false},
		{"tempfile_test" + sep + "foo", true},
		{"tempfile_test*" + sep + "foo", true},
		{"tempfile_test" + sep + "*foo", true},
		{sep + "tempfile_test" + sep + "*foo", true},
		{"tempfile_test*foo" + sep, true},
	}
	for _, tt := range tests {
		t.Run(tt.pattern, func { t ->
			tmpfile, err := CreateTemp(tmpDir, tt.pattern)
			if tmpfile != nil {
				defer tmpfile.Close()
			}
			if tt.wantErr {
				if err == nil {
					t.Errorf("CreateTemp(..., %#q) succeeded, expected error", tt.pattern)
				}
				if !errors.Is(err, ErrPatternHasSeparator) {
					t.Errorf("CreateTemp(..., %#q): %v, expected ErrPatternHasSeparator", tt.pattern, err)
				}
			} else if err != nil {
				t.Errorf("CreateTemp(..., %#q): %v", tt.pattern, err)
			}
		})
	}
}

func TestMkdirTemp(t *testing.T) {
	t.Parallel()

	name, err := MkdirTemp("/_not_exists_", "foo")
	if name != "" || err == nil {
		t.Errorf("MkdirTemp(`/_not_exists_`, `foo`) = %v, %v", name, err)
	}

	tests := []struct {
		pattern                string
		wantPrefix, wantSuffix string
	}{
		{"tempfile_test", "tempfile_test", ""},
		{"tempfile_test*", "tempfile_test", ""},
		{"tempfile_test*xyz", "tempfile_test", "xyz"},
	}

	dir := filepath.Clean(TempDir())

	runTestMkdirTemp := func(t *testing.T, pattern, wantRePat string) {
		name, err := MkdirTemp(dir, pattern)
		if name == "" || err != nil {
			t.Fatalf("MkdirTemp(dir, `tempfile_test`) = %v, %v", name, err)
		}
		defer Remove(name)

		re := regexp.MustCompile(wantRePat)
		if !re.MatchString(name) {
			t.Errorf("MkdirTemp(%q, %q) created bad name\n\t%q\ndid not match pattern\n\t%q", dir, pattern, name, wantRePat)
		}
	}

	for _, tt := range tests {
		t.Run(tt.pattern, func { t ->
			wantRePat := "^" + regexp.QuoteMeta(filepath.Join(dir, tt.wantPrefix)) + "[0-9]+" + regexp.QuoteMeta(tt.wantSuffix) + "$"
			runTestMkdirTemp(t, tt.pattern, wantRePat)
		})
	}

	// Separately testing "*xyz" (which has no prefix). That is when constructing the
	// pattern to assert on, as in the previous loop, using filepath.Join for an empty
	// prefix filepath.Join(dir, ""), produces the pattern:
	//     ^<DIR>[0-9]+xyz$
	// yet we just want to match
	//     "^<DIR>/[0-9]+xyz"
	t.Run("*xyz", func { t ->
		wantRePat := "^" + regexp.QuoteMeta(filepath.Join(dir)) + regexp.QuoteMeta(string(filepath.Separator)) + "[0-9]+xyz$"
		runTestMkdirTemp(t, "*xyz", wantRePat)
	})
}

// test that we return a nice error message if the dir argument to TempDir doesn't
// exist (or that it's empty and TempDir doesn't exist)
func TestMkdirTempBadDir(t *testing.T) {
	t.Parallel()

	dir, err := MkdirTemp("", "MkdirTempBadDir")
	if err != nil {
		t.Fatal(err)
	}
	defer RemoveAll(dir)

	badDir := filepath.Join(dir, "not-exist")
	_, err = MkdirTemp(badDir, "foo")
	if pe, ok := err.(*fs.PathError); !ok || !IsNotExist(err) || pe.Path != badDir {
		t.Errorf("TempDir error = %#v; want PathError for path %q satisfying IsNotExist", err, badDir)
	}
}

func TestMkdirTempBadPattern(t *testing.T) {
	t.Parallel()

	tmpDir, err := MkdirTemp("", t.Name())
	if err != nil {
		t.Fatal(err)
	}
	defer RemoveAll(tmpDir)

	const sep = string(PathSeparator)
	tests := []struct {
		pattern string
		wantErr bool
	}{
		{"ioutil*test", false},
		{"tempfile_test*foo", false},
		{"tempfile_test" + sep + "foo", true},
		{"tempfile_test*" + sep + "foo", true},
		{"tempfile_test" + sep + "*foo", true},
		{sep + "tempfile_test" + sep + "*foo", true},
		{"tempfile_test*foo" + sep, true},
	}
	for _, tt := range tests {
		t.Run(tt.pattern, func { t ->
			_, err := MkdirTemp(tmpDir, tt.pattern)
			if tt.wantErr {
				if err == nil {
					t.Errorf("MkdirTemp(..., %#q) succeeded, expected error", tt.pattern)
				}
				if !errors.Is(err, ErrPatternHasSeparator) {
					t.Errorf("MkdirTemp(..., %#q): %v, expected ErrPatternHasSeparator", tt.pattern, err)
				}
			} else if err != nil {
				t.Errorf("MkdirTemp(..., %#q): %v", tt.pattern, err)
			}
		})
	}
}
