// Copyright 2009 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

package filepath_test

import (
	"fmt"
	"internal/testenv"
	"io/ioutil"
	"os"
	. "path/filepath"
	"reflect"
	"runtime"
	"sort"
	"strings"
	"testing"
)

type MatchTest struct {
	pattern, s string
	match      bool
	valid      bool
	err        error
}

var matchTests = []MatchTest{
	{"abc", "abc", true, true, nil},
	{"*", "abc", true, true, nil},
	{"*c", "abc", true, true, nil},
	{"a*", "a", true, true, nil},
	{"a*", "abc", true, true, nil},
	{"a*", "ab/c", false, true, nil},
	{"a*/b", "abc/b", true, true, nil},
	{"a*/b", "a/c/b", false, true, nil},
	{"a*b*c*d*e*/f", "axbxcxdxe/f", true, true, nil},
	{"a*b*c*d*e*/f", "axbxcxdxexxx/f", true, true, nil},
	{"a*b*c*d*e*/f", "axbxcxdxe/xxx/f", false, true, nil},
	{"a*b*c*d*e*/f", "axbxcxdxexxx/fff", false, true, nil},
	{"a*b?c*x", "abxbbxdbxebxczzx", true, true, nil},
	{"a*b?c*x", "abxbbxdbxebxczzy", false, true, nil},
	{"ab[c]", "abc", true, true, nil},
	{"ab[b-d]", "abc", true, true, nil},
	{"ab[e-g]", "abc", false, true, nil},
	{"ab[^c]", "abc", false, true, nil},
	{"ab[^b-d]", "abc", false, true, nil},
	{"ab[^e-g]", "abc", true, true, nil},
	{"a\\*b", "a*b", true, true, nil},
	{"a\\*b", "ab", false, true, nil},
	{"a?b", "a☺b", true, true, nil},
	{"a[^a]b", "a☺b", true, true, nil},
	{"a???b", "a☺b", false, true, nil},
	{"a[^a][^a][^a]b", "a☺b", false, true, nil},
	{"[a-ζ]*", "α", true, true, nil},
	{"*[a-ζ]", "A", false, true, nil},
	{"a?b", "a/b", false, true, nil},
	{"a*b", "a/b", false, true, nil},
	{"[\\]a]", "]", true, true, nil},
	{"[\\-]", "-", true, true, nil},
	{"[x\\-]", "x", true, true, nil},
	{"[x\\-]", "-", true, true, nil},
	{"[x\\-]", "z", false, true, nil},
	{"[\\-x]", "x", true, true, nil},
	{"[\\-x]", "-", true, true, nil},
	{"[\\-x]", "a", false, true, nil},
	{"[]a]", "]", false, false, ErrBadPattern},
	{"[-]", "-", false, false, ErrBadPattern},
	{"[x-]", "x", false, false, ErrBadPattern},
	{"[x-]", "-", false, false, ErrBadPattern},
	{"[x-]", "z", false, false, ErrBadPattern},
	{"[-x]", "x", false, false, ErrBadPattern},
	{"[-x]", "-", false, false, ErrBadPattern},
	{"[-x]", "a", false, false, ErrBadPattern},
	{"\\", "a", false, false, ErrBadPattern},
	{"[a-b-c]", "a", false, false, ErrBadPattern},
	{"[", "a", false, false, ErrBadPattern},
	{"[^", "a", false, false, ErrBadPattern},
	{"[^bc", "a", false, false, ErrBadPattern},
	{"a[", "a", false, false, nil},
	{"a[", "ab", false, false, ErrBadPattern},
	{"*x", "xxx", true, true, nil},
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

// contains returns true if vector contains the string s.
func contains(vector []string, s string) bool {
	for _, elem := range vector {
		if elem == s {
			return true
		}
	}
	return false
}

func TestIsPatternValid(t *testing.T) {
	for _, tt := range matchTests {
		valid := IsPatternValid(tt.pattern)
		if valid && !tt.valid || !valid && tt.valid {
			t.Errorf("IsPatternValid(%#q) returned %t", tt.pattern, tt.valid)
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
		if !contains(matches, result) {
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

func TestGlobError(t *testing.T) {
	_, err := Glob("[]")
	if err == nil {
		t.Error("expected error for bad pattern; got none")
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

	tmpDir, err := ioutil.TempDir("", "globsymlink")
	if err != nil {
		t.Fatal("creating temp dir:", err)
	}
	defer os.RemoveAll(tmpDir)

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
		if !contains(matches, dest) {
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
	sort.Strings(want)
	return want
}

func (test *globTest) globAbs(root, rootPattern string) error {
	p := FromSlash(rootPattern + `\` + test.pattern)
	have, err := Glob(p)
	if err != nil {
		return err
	}
	sort.Strings(have)
	want := test.buildWant(root + `\`)
	if strings.Join(want, "_") == strings.Join(have, "_") {
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
	sort.Strings(have)
	want := test.buildWant(root)
	if strings.Join(want, "_") == strings.Join(have, "_") {
		return nil
	}
	// try also matching version without root prefix
	wantWithNoRoot := test.buildWant("")
	if strings.Join(wantWithNoRoot, "_") == strings.Join(have, "_") {
		return nil
	}
	return fmt.Errorf("Glob(%q) returns %q, but %q expected", p, have, want)
}

func TestWindowsGlob(t *testing.T) {
	if runtime.GOOS != "windows" {
		t.Skipf("skipping windows specific test")
	}

	tmpDir, err := ioutil.TempDir("", "TestWindowsGlob")
	if err != nil {
		t.Fatal(err)
	}
	defer os.RemoveAll(tmpDir)

	// /tmp may itself be a symlink
	tmpDir, err = EvalSymlinks(tmpDir)
	if err != nil {
		t.Fatal("eval symlink for tmp dir:", err)
	}

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
		err := ioutil.WriteFile(Join(tmpDir, file), nil, 0666)
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
		err = test.globAbs(tmpDir, tmpDir)
		if err != nil {
			t.Error(err)
		}
		// test C:\*Documents and Settings\...
		p = tmpDir
		p = strings.Replace(p, `:\`, `:\*`, 1)
		err = test.globAbs(tmpDir, p)
		if err != nil {
			t.Error(err)
		}
		// test C:\Documents and Settings*\...
		p = tmpDir
		p = strings.Replace(p, `:\`, `:`, 1)
		p = strings.Replace(p, `\`, `*\`, 1)
		p = strings.Replace(p, `:`, `:\`, 1)
		err = test.globAbs(tmpDir, p)
		if err != nil {
			t.Error(err)
		}
	}

	// test relative paths
	wd, err := os.Getwd()
	if err != nil {
		t.Fatal(err)
	}
	err = os.Chdir(tmpDir)
	if err != nil {
		t.Fatal(err)
	}
	defer func() {
		err := os.Chdir(wd)
		if err != nil {
			t.Fatal(err)
		}
	}()
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
	if !reflect.DeepEqual(matches, want) {
		t.Fatalf("Glob(%#q) = %v want %v", pattern, matches, want)
	}
}
