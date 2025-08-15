// Copyright 2009 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

package filepath_test

import (
	"fmt"
	"internal/testenv"
	"os"
	. "path/filepath"
	"runtime"
	"slices"
	"strings"
	"testing"
)

type MatchTest struct {
	pattern, s string
	match      bool
	err        error
}

var matchTests = []MatchTest{
	{"abc", "abc", true, nil},
	{"*", "abc", true, nil},
	{"*c", "abc", true, nil},
	{"a*", "a", true, nil},
	{"a*", "abc", true, nil},
	{"a*", "ab/c", false, nil},
	{"a*/b", "abc/b", true, nil},
	{"a*/b", "a/c/b", false, nil},
	{"a*b*c*d*e*/f", "axbxcxdxe/f", true, nil},
	{"a*b*c*d*e*/f", "axbxcxdxexxx/f", true, nil},
	{"a*b*c*d*e*/f", "axbxcxdxe/xxx/f", false, nil},
	{"a*b*c*d*e*/f", "axbxcxdxexxx/fff", false, nil},
	{"a*b?c*x", "abxbbxdbxebxczzx", true, nil},
	{"a*b?c*x", "abxbbxdbxebxczzy", false, nil},
	{"ab[c]", "abc", true, nil},
	{"ab[b-d]", "abc", true, nil},
	{"ab[e-g]", "abc", false, nil},
	{"ab[^c]", "abc", false, nil},
	{"ab[^b-d]", "abc", false, nil},
	{"ab[^e-g]", "abc", true, nil},
	{"a\\*b", "a*b", true, nil},
	{"a\\*b", "ab", false, nil},
	{"a?b", "a☺b", true, nil},
	{"a[^a]b", "a☺b", true, nil},
	{"a???b", "a☺b", false, nil},
	{"a[^a][^a][^a]b", "a☺b", false, nil},
	{"[a-ζ]*", "α", true, nil},
	{"*[a-ζ]", "A", false, nil},
	{"a?b", "a/b", false, nil},
	{"a*b", "a/b", false, nil},
	{"[\\]a]", "]", true, nil},
	{"[\\-]", "-", true, nil},
	{"[x\\-]", "x", true, nil},
	{"[x\\-]", "-", true, nil},
	{"[x\\-]", "z", false, nil},
	{"[\\-x]", "x", true, nil},
	{"[\\-x]", "-", true, nil},
	{"[\\-x]", "a", false, nil},
	{"[]a]", "]", false, ErrBadPattern},
	{"[-]", "-", false, ErrBadPattern},
	{"[x-]", "x", false, ErrBadPattern},
	{"[x-]", "-", false, ErrBadPattern},
	{"[x-]", "z", false, ErrBadPattern},
	{"[-x]", "x", false, ErrBadPattern},
	{"[-x]", "-", false, ErrBadPattern},
	{"[-x]", "a", false, ErrBadPattern},
	{"\\", "a", false, ErrBadPattern},
	{"[a-b-c]", "a", false, ErrBadPattern},
	{"[", "a", false, ErrBadPattern},
	{"[^", "a", false, ErrBadPattern},
	{"[^bc", "a", false, ErrBadPattern},
	{"a[", "a", false, ErrBadPattern},
	{"a[", "ab", false, ErrBadPattern},
	{"a[", "x", false, ErrBadPattern},
	{"a/b[", "x", false, ErrBadPattern},
	{"*x", "xxx", true, nil},
}

func errp(e error) string {
	if e == nil {
		return "<nil>"
	}
	return e.Error()
}

func TestMatch(t *testing.T) {
	for _, tt := range matchTests {
		pattern := tt.pattern
		s := tt.s
		if runtime.GOOS == "windows" {
			if strings.Contains(pattern, "\\") {
				// no escape allowed on windows.
				continue
			}
			pattern = Clean(pattern)
			s = Clean(s)
		}
		ok, err := Match(pattern, s)
		if ok != tt.match || err != tt.err {
			t.Errorf("Match(%#q, %#q) = %v, %q want %v, %q", pattern, s, ok, errp(err), tt.match, errp(tt.err))
		}
	}
}

var globTests = []struct {
	pattern, result string
}{
	{"match.go", "match.go"},
	{"mat?h.go", "match.go"},
	{"*", "match.go"},
	{"../*/match.go", "../filepath/match.go"},
}

func TestGlob(t *testing.T) {
	for _, tt := range globTests {
		pattern := tt.pattern
		result := tt.result
		if runtime.GOOS == "windows" {
			pattern = Clean(pattern)
			result = Clean(result)
		}
		matches, err := Glob(pattern)
		if err != nil {
			t.Errorf("Glob error for %q: %s", pattern, err)
			continue
		}
		if !slices.Contains(matches, result) {
			t.Errorf("Glob(%#q) = %#v want %v", pattern, matches, result)
		}
	}
	for _, pattern := range []string{"no_match", "../*/no_match"} {
		matches, err := Glob(pattern)
		if err != nil {
			t.Errorf("Glob error for %q: %s", pattern, err)
			continue
		}
		if len(matches) != 0 {
			t.Errorf("Glob(%#q) = %#v want []", pattern, matches)
		}
	}
}

func TestCVE202230632(t *testing.T) {
	// Prior to CVE-2022-30632, this would cause a stack exhaustion given a
	// large number of separators (more than 4,000,000). There is now a limit
	// of 10,000.
	_, err := Glob("/*" + strings.Repeat("/", 10001))
	if err != ErrBadPattern {
		t.Fatalf("Glob returned err=%v, want ErrBadPattern", err)
	}
}

func TestGlobError(t *testing.T) {
	bad := []string{`[]`, `nonexist/[]`}
	for _, pattern := range bad {
		if _, err := Glob(pattern); err != ErrBadPattern {
			t.Errorf("Glob(%#q) returned err=%v, want ErrBadPattern", pattern, err)
		}
	}
}

func TestGlobUNC(t *testing.T) {
	// Just make sure this runs without crashing for now.
	// See issue 15879.
	Glob(`\\?\C:\*`)
}

var globSymlinkTests = []struct {
	path, dest string
	brokenLink bool
}{
	{"test1", "link1", false},
	{"test2", "link2", true},
}

func TestGlobSymlink(t *testing.T) {
	testenv.MustHaveSymlink(t)

	tmpDir := t.TempDir()
	for _, tt := range globSymlinkTests {
		path := Join(tmpDir, tt.path)
		dest := Join(tmpDir, tt.dest)
		f, err := os.Create(path)
		if err != nil {
			t.Fatal(err)
		}
		if err := f.Close(); err != nil {
			t.Fatal(err)
		}
		err = os.Symlink(path, dest)
		if err != nil {
			t.Fatal(err)
		}
		if tt.brokenLink {
			// Break the symlink.
			os.Remove(path)
		}
		matches, err := Glob(dest)
		if err != nil {
			t.Errorf("GlobSymlink error for %q: %s", dest, err)
		}
		if !slices.Contains(matches, dest) {
			t.Errorf("Glob(%#q) = %#v want %v", dest, matches, dest)
		}
	}
}

type globTest struct {
	pattern string
	matches []string
}

func (test *globTest) buildWant(root string) []string {
	want := make([]string, 0)
	for _, m := range test.matches {
		want = append(want, root+FromSlash(m))
	}
	slices.Sort(want)
	return want
}

func (test *globTest) globAbs(root, rootPattern string) error {
	p := FromSlash(rootPattern + `\` + test.pattern)
	have, err := Glob(p)
	if err != nil {
		return err
	}
	slices.Sort(have)
	want := test.buildWant(root + `\`)
	if slices.Equal(want, have) {
		return nil
	}
	return fmt.Errorf("Glob(%q) returns %q, but %q expected", p, have, want)
}

func (test *globTest) globRel(root string) error {
	p := root + FromSlash(test.pattern)
	have, err := Glob(p)
	if err != nil {
		return err
	}
	slices.Sort(have)
	want := test.buildWant(root)
	if slices.Equal(want, have) {
		return nil
	}
	// try also matching version without root prefix
	wantWithNoRoot := test.buildWant("")
	if slices.Equal(wantWithNoRoot, have) {
		return nil
	}
	return fmt.Errorf("Glob(%q) returns %q, but %q expected", p, have, want)
}

func TestWindowsGlob(t *testing.T) {
	if runtime.GOOS != "windows" {
		t.Skipf("skipping windows specific test")
	}

	tmpDir := tempDirCanonical(t)
	if len(tmpDir) < 3 {
		t.Fatalf("tmpDir path %q is too short", tmpDir)
	}
	if tmpDir[1] != ':' {
		t.Fatalf("tmpDir path %q must have drive letter in it", tmpDir)
	}

	dirs := []string{
		"a",
		"b",
		"dir/d/bin",
	}
	files := []string{
		"dir/d/bin/git.exe",
	}
	for _, dir := range dirs {
		err := os.MkdirAll(Join(tmpDir, dir), 0777)
		if err != nil {
			t.Fatal(err)
		}
	}
	for _, file := range files {
		err := os.WriteFile(Join(tmpDir, file), nil, 0666)
		if err != nil {
			t.Fatal(err)
		}
	}

	tests := []globTest{
		{"a", []string{"a"}},
		{"b", []string{"b"}},
		{"c", []string{}},
		{"*", []string{"a", "b", "dir"}},
		{"d*", []string{"dir"}},
		{"*i*", []string{"dir"}},
		{"*r", []string{"dir"}},
		{"?ir", []string{"dir"}},
		{"?r", []string{}},
		{"d*/*/bin/git.exe", []string{"dir/d/bin/git.exe"}},
	}

	// test absolute paths
	for _, test := range tests {
		var p string
		if err := test.globAbs(tmpDir, tmpDir); err != nil {
			t.Error(err)
		}
		// test C:\*Documents and Settings\...
		p = tmpDir
		p = strings.Replace(p, `:\`, `:\*`, 1)
		if err := test.globAbs(tmpDir, p); err != nil {
			t.Error(err)
		}
		// test C:\Documents and Settings*\...
		p = tmpDir
		p = strings.Replace(p, `:\`, `:`, 1)
		p = strings.Replace(p, `\`, `*\`, 1)
		p = strings.Replace(p, `:`, `:\`, 1)
		if err := test.globAbs(tmpDir, p); err != nil {
			t.Error(err)
		}
	}

	// test relative paths
	t.Chdir(tmpDir)
	for _, test := range tests {
		err := test.globRel("")
		if err != nil {
			t.Error(err)
		}
		err = test.globRel(`.\`)
		if err != nil {
			t.Error(err)
		}
		err = test.globRel(tmpDir[:2]) // C:
		if err != nil {
			t.Error(err)
		}
	}
}

func TestNonWindowsGlobEscape(t *testing.T) {
	if runtime.GOOS == "windows" {
		t.Skipf("skipping non-windows specific test")
	}
	pattern := `\match.go`
	want := []string{"match.go"}
	matches, err := Glob(pattern)
	if err != nil {
		t.Fatalf("Glob error for %q: %s", pattern, err)
	}
	if !slices.Equal(matches, want) {
		t.Fatalf("Glob(%#q) = %v want %v", pattern, matches, want)
	}
}
