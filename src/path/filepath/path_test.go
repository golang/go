// Copyright 2009 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

package filepath_test

import (
	"errors"
	"fmt"
	"internal/testenv"
	"io/fs"
	"os"
	"path/filepath"
	"reflect"
	"runtime"
	"sort"
	"strings"
	"syscall"
	"testing"
)

type PathTest struct {
	path, result string
}

var cleantests = []PathTest{
	// Already clean
	{"abc", "abc"},
	{"abc/def", "abc/def"},
	{"a/b/c", "a/b/c"},
	{".", "."},
	{"..", ".."},
	{"../..", "../.."},
	{"../../abc", "../../abc"},
	{"/abc", "/abc"},
	{"/", "/"},

	// Empty is current dir
	{"", "."},

	// Remove trailing slash
	{"abc/", "abc"},
	{"abc/def/", "abc/def"},
	{"a/b/c/", "a/b/c"},
	{"./", "."},
	{"../", ".."},
	{"../../", "../.."},
	{"/abc/", "/abc"},

	// Remove doubled slash
	{"abc//def//ghi", "abc/def/ghi"},
	{"//abc", "/abc"},
	{"///abc", "/abc"},
	{"//abc//", "/abc"},
	{"abc//", "abc"},

	// Remove . elements
	{"abc/./def", "abc/def"},
	{"/./abc/def", "/abc/def"},
	{"abc/.", "abc"},

	// Remove .. elements
	{"abc/def/ghi/../jkl", "abc/def/jkl"},
	{"abc/def/../ghi/../jkl", "abc/jkl"},
	{"abc/def/..", "abc"},
	{"abc/def/../..", "."},
	{"/abc/def/../..", "/"},
	{"abc/def/../../..", ".."},
	{"/abc/def/../../..", "/"},
	{"abc/def/../../../ghi/jkl/../../../mno", "../../mno"},
	{"/../abc", "/abc"},

	// Combinations
	{"abc/./../def", "def"},
	{"abc//./../def", "def"},
	{"abc/../../././../def", "../../def"},
}

var wincleantests = []PathTest{
	{`c:`, `c:.`},
	{`c:\`, `c:\`},
	{`c:\abc`, `c:\abc`},
	{`c:abc\..\..\.\.\..\def`, `c:..\..\def`},
	{`c:\abc\def\..\..`, `c:\`},
	{`c:\..\abc`, `c:\abc`},
	{`c:..\abc`, `c:..\abc`},
	{`\`, `\`},
	{`/`, `\`},
	{`\\i\..\c$`, `\c$`},
	{`\\i\..\i\c$`, `\i\c$`},
	{`\\i\..\I\c$`, `\I\c$`},
	{`\\host\share\foo\..\bar`, `\\host\share\bar`},
	{`//host/share/foo/../baz`, `\\host\share\baz`},
	{`\\a\b\..\c`, `\\a\b\c`},
	{`\\a\b`, `\\a\b`},
}

func TestClean(t *testing.T) {
	tests := cleantests
	if runtime.GOOS == "windows" {
		for i := range tests {
			tests[i].result = filepath.FromSlash(tests[i].result)
		}
		tests = append(tests, wincleantests...)
	}
	for _, test := range tests {
		if s := filepath.Clean(test.path); s != test.result {
			t.Errorf("Clean(%q) = %q, want %q", test.path, s, test.result)
		}
		if s := filepath.Clean(test.result); s != test.result {
			t.Errorf("Clean(%q) = %q, want %q", test.result, s, test.result)
		}
	}

	if testing.Short() {
		t.Skip("skipping malloc count in short mode")
	}
	if runtime.GOMAXPROCS(0) > 1 {
		t.Log("skipping AllocsPerRun checks; GOMAXPROCS>1")
		return
	}

	for _, test := range tests {
		allocs := testing.AllocsPerRun(100, func() { filepath.Clean(test.result) })
		if allocs > 0 {
			t.Errorf("Clean(%q): %v allocs, want zero", test.result, allocs)
		}
	}
}

const sep = filepath.Separator

var slashtests = []PathTest{
	{"", ""},
	{"/", string(sep)},
	{"/a/b", string([]byte{sep, 'a', sep, 'b'})},
	{"a//b", string([]byte{'a', sep, sep, 'b'})},
}

func TestFromAndToSlash(t *testing.T) {
	for _, test := range slashtests {
		if s := filepath.FromSlash(test.path); s != test.result {
			t.Errorf("FromSlash(%q) = %q, want %q", test.path, s, test.result)
		}
		if s := filepath.ToSlash(test.result); s != test.path {
			t.Errorf("ToSlash(%q) = %q, want %q", test.result, s, test.path)
		}
	}
}

type SplitListTest struct {
	list   string
	result []string
}

const lsep = filepath.ListSeparator

var splitlisttests = []SplitListTest{
	{"", []string{}},
	{string([]byte{'a', lsep, 'b'}), []string{"a", "b"}},
	{string([]byte{lsep, 'a', lsep, 'b'}), []string{"", "a", "b"}},
}

var winsplitlisttests = []SplitListTest{
	// quoted
	{`"a"`, []string{`a`}},

	// semicolon
	{`";"`, []string{`;`}},
	{`"a;b"`, []string{`a;b`}},
	{`";";`, []string{`;`, ``}},
	{`;";"`, []string{``, `;`}},

	// partially quoted
	{`a";"b`, []string{`a;b`}},
	{`a; ""b`, []string{`a`, ` b`}},
	{`"a;b`, []string{`a;b`}},
	{`""a;b`, []string{`a`, `b`}},
	{`"""a;b`, []string{`a;b`}},
	{`""""a;b`, []string{`a`, `b`}},
	{`a";b`, []string{`a;b`}},
	{`a;b";c`, []string{`a`, `b;c`}},
	{`"a";b";c`, []string{`a`, `b;c`}},
}

func TestSplitList(t *testing.T) {
	tests := splitlisttests
	if runtime.GOOS == "windows" {
		tests = append(tests, winsplitlisttests...)
	}
	for _, test := range tests {
		if l := filepath.SplitList(test.list); !reflect.DeepEqual(l, test.result) {
			t.Errorf("SplitList(%#q) = %#q, want %#q", test.list, l, test.result)
		}
	}
}

type SplitTest struct {
	path, dir, file string
}

var unixsplittests = []SplitTest{
	{"a/b", "a/", "b"},
	{"a/b/", "a/b/", ""},
	{"a/", "a/", ""},
	{"a", "", "a"},
	{"/", "/", ""},
}

var winsplittests = []SplitTest{
	{`c:`, `c:`, ``},
	{`c:/`, `c:/`, ``},
	{`c:/foo`, `c:/`, `foo`},
	{`c:/foo/bar`, `c:/foo/`, `bar`},
	{`//host/share`, `//host/share`, ``},
	{`//host/share/`, `//host/share/`, ``},
	{`//host/share/foo`, `//host/share/`, `foo`},
	{`\\host\share`, `\\host\share`, ``},
	{`\\host\share\`, `\\host\share\`, ``},
	{`\\host\share\foo`, `\\host\share\`, `foo`},
}

func TestSplit(t *testing.T) {
	var splittests []SplitTest
	splittests = unixsplittests
	if runtime.GOOS == "windows" {
		splittests = append(splittests, winsplittests...)
	}
	for _, test := range splittests {
		if d, f := filepath.Split(test.path); d != test.dir || f != test.file {
			t.Errorf("Split(%q) = %q, %q, want %q, %q", test.path, d, f, test.dir, test.file)
		}
	}
}

type JoinTest struct {
	elem []string
	path string
}

var jointests = []JoinTest{
	// zero parameters
	{[]string{}, ""},

	// one parameter
	{[]string{""}, ""},
	{[]string{"/"}, "/"},
	{[]string{"a"}, "a"},

	// two parameters
	{[]string{"a", "b"}, "a/b"},
	{[]string{"a", ""}, "a"},
	{[]string{"", "b"}, "b"},
	{[]string{"/", "a"}, "/a"},
	{[]string{"/", "a/b"}, "/a/b"},
	{[]string{"/", ""}, "/"},
	{[]string{"//", "a"}, "/a"},
	{[]string{"/a", "b"}, "/a/b"},
	{[]string{"a/", "b"}, "a/b"},
	{[]string{"a/", ""}, "a"},
	{[]string{"", ""}, ""},

	// three parameters
	{[]string{"/", "a", "b"}, "/a/b"},
}

var winjointests = []JoinTest{
	{[]string{`directory`, `file`}, `directory\file`},
	{[]string{`C:\Windows\`, `System32`}, `C:\Windows\System32`},
	{[]string{`C:\Windows\`, ``}, `C:\Windows`},
	{[]string{`C:\`, `Windows`}, `C:\Windows`},
	{[]string{`C:`, `a`}, `C:a`},
	{[]string{`C:`, `a\b`}, `C:a\b`},
	{[]string{`C:`, `a`, `b`}, `C:a\b`},
	{[]string{`C:`, ``, `b`}, `C:b`},
	{[]string{`C:`, ``, ``, `b`}, `C:b`},
	{[]string{`C:`, ``}, `C:.`},
	{[]string{`C:`, ``, ``}, `C:.`},
	{[]string{`C:.`, `a`}, `C:a`},
	{[]string{`C:a`, `b`}, `C:a\b`},
	{[]string{`C:a`, `b`, `d`}, `C:a\b\d`},
	{[]string{`\\host\share`, `foo`}, `\\host\share\foo`},
	{[]string{`\\host\share\foo`}, `\\host\share\foo`},
	{[]string{`//host/share`, `foo/bar`}, `\\host\share\foo\bar`},
	{[]string{`\`}, `\`},
	{[]string{`\`, ``}, `\`},
	{[]string{`\`, `a`}, `\a`},
	{[]string{`\\`, `a`}, `\a`},
	{[]string{`\`, `a`, `b`}, `\a\b`},
	{[]string{`\\`, `a`, `b`}, `\a\b`},
	{[]string{`\`, `\\a\b`, `c`}, `\a\b\c`},
	{[]string{`\\a`, `b`, `c`}, `\a\b\c`},
	{[]string{`\\a\`, `b`, `c`}, `\a\b\c`},
}

func TestJoin(t *testing.T) {
	if runtime.GOOS == "windows" {
		jointests = append(jointests, winjointests...)
	}
	for _, test := range jointests {
		expected := filepath.FromSlash(test.path)
		if p := filepath.Join(test.elem...); p != expected {
			t.Errorf("join(%q) = %q, want %q", test.elem, p, expected)
		}
	}
}

type ExtTest struct {
	path, ext string
}

var exttests = []ExtTest{
	{"path.go", ".go"},
	{"path.pb.go", ".go"},
	{"a.dir/b", ""},
	{"a.dir/b.go", ".go"},
	{"a.dir/", ""},
}

func TestExt(t *testing.T) {
	for _, test := range exttests {
		if x := filepath.Ext(test.path); x != test.ext {
			t.Errorf("Ext(%q) = %q, want %q", test.path, x, test.ext)
		}
	}
}

type Node struct {
	name    string
	entries []*Node // nil if the entry is a file
	mark    int
}

var tree = &Node{
	"testdata",
	[]*Node{
		{"a", nil, 0},
		{"b", []*Node{}, 0},
		{"c", nil, 0},
		{
			"d",
			[]*Node{
				{"x", nil, 0},
				{"y", []*Node{}, 0},
				{
					"z",
					[]*Node{
						{"u", nil, 0},
						{"v", nil, 0},
					},
					0,
				},
			},
			0,
		},
	},
	0,
}

func walkTree(n *Node, path string, f func(path string, n *Node)) {
	f(path, n)
	for _, e := range n.entries {
		walkTree(e, filepath.Join(path, e.name), f)
	}
}

func makeTree(t *testing.T) {
	walkTree(tree, tree.name, func(path string, n *Node) {
		if n.entries == nil {
			fd, err := os.Create(path)
			if err != nil {
				t.Errorf("makeTree: %v", err)
				return
			}
			fd.Close()
		} else {
			os.Mkdir(path, 0770)
		}
	})
}

func markTree(n *Node) { walkTree(n, "", func(path string, n *Node) { n.mark++ }) }

func checkMarks(t *testing.T, report bool) {
	walkTree(tree, tree.name, func(path string, n *Node) {
		if n.mark != 1 && report {
			t.Errorf("node %s mark = %d; expected 1", path, n.mark)
		}
		n.mark = 0
	})
}

// Assumes that each node name is unique. Good enough for a test.
// If clear is true, any incoming error is cleared before return. The errors
// are always accumulated, though.
func mark(d fs.DirEntry, err error, errors *[]error, clear bool) error {
	name := d.Name()
	walkTree(tree, tree.name, func(path string, n *Node) {
		if n.name == name {
			n.mark++
		}
	})
	if err != nil {
		*errors = append(*errors, err)
		if clear {
			return nil
		}
		return err
	}
	return nil
}

// chdir changes the current working directory to the named directory,
// and then restore the original working directory at the end of the test.
func chdir(t *testing.T, dir string) {
	olddir, err := os.Getwd()
	if err != nil {
		t.Fatalf("getwd %s: %v", dir, err)
	}
	if err := os.Chdir(dir); err != nil {
		t.Fatalf("chdir %s: %v", dir, err)
	}

	t.Cleanup(func() {
		if err := os.Chdir(olddir); err != nil {
			t.Errorf("restore original working directory %s: %v", olddir, err)
			os.Exit(1)
		}
	})
}

func chtmpdir(t *testing.T) (restore func()) {
	oldwd, err := os.Getwd()
	if err != nil {
		t.Fatalf("chtmpdir: %v", err)
	}
	d, err := os.MkdirTemp("", "test")
	if err != nil {
		t.Fatalf("chtmpdir: %v", err)
	}
	if err := os.Chdir(d); err != nil {
		t.Fatalf("chtmpdir: %v", err)
	}
	return func() {
		if err := os.Chdir(oldwd); err != nil {
			t.Fatalf("chtmpdir: %v", err)
		}
		os.RemoveAll(d)
	}
}

// tempDirCanonical returns a temporary directory for the test to use, ensuring
// that the returned path does not contain symlinks.
func tempDirCanonical(t *testing.T) string {
	dir := t.TempDir()

	cdir, err := filepath.EvalSymlinks(dir)
	if err != nil {
		t.Errorf("tempDirCanonical: %v", err)
	}

	return cdir
}

func TestWalk(t *testing.T) {
	walk := func(root string, fn fs.WalkDirFunc) error {
		return filepath.Walk(root, func(path string, info fs.FileInfo, err error) error {
			return fn(path, &statDirEntry{info}, err)
		})
	}
	testWalk(t, walk, 1)
}

type statDirEntry struct {
	info fs.FileInfo
}

func (d *statDirEntry) Name() string               { return d.info.Name() }
func (d *statDirEntry) IsDir() bool                { return d.info.IsDir() }
func (d *statDirEntry) Type() fs.FileMode          { return d.info.Mode().Type() }
func (d *statDirEntry) Info() (fs.FileInfo, error) { return d.info, nil }

func TestWalkDir(t *testing.T) {
	testWalk(t, filepath.WalkDir, 2)
}

func testWalk(t *testing.T, walk func(string, fs.WalkDirFunc) error, errVisit int) {
	if runtime.GOOS == "ios" {
		restore := chtmpdir(t)
		defer restore()
	}

	tmpDir := t.TempDir()

	origDir, err := os.Getwd()
	if err != nil {
		t.Fatal("finding working dir:", err)
	}
	if err = os.Chdir(tmpDir); err != nil {
		t.Fatal("entering temp dir:", err)
	}
	defer os.Chdir(origDir)

	makeTree(t)
	errors := make([]error, 0, 10)
	clear := true
	markFn := func(path string, d fs.DirEntry, err error) error {
		return mark(d, err, &errors, clear)
	}
	// Expect no errors.
	err = walk(tree.name, markFn)
	if err != nil {
		t.Fatalf("no error expected, found: %s", err)
	}
	if len(errors) != 0 {
		t.Fatalf("unexpected errors: %s", errors)
	}
	checkMarks(t, true)
	errors = errors[0:0]

	t.Run("PermErr", func(t *testing.T) {
		// Test permission errors. Only possible if we're not root
		// and only on some file systems (AFS, FAT).  To avoid errors during
		// all.bash on those file systems, skip during go test -short.
		if runtime.GOOS == "windows" {
			t.Skip("skipping on Windows")
		}
		if os.Getuid() == 0 {
			t.Skip("skipping as root")
		}
		if testing.Short() {
			t.Skip("skipping in short mode")
		}

		// introduce 2 errors: chmod top-level directories to 0
		os.Chmod(filepath.Join(tree.name, tree.entries[1].name), 0)
		os.Chmod(filepath.Join(tree.name, tree.entries[3].name), 0)

		// 3) capture errors, expect two.
		// mark respective subtrees manually
		markTree(tree.entries[1])
		markTree(tree.entries[3])
		// correct double-marking of directory itself
		tree.entries[1].mark -= errVisit
		tree.entries[3].mark -= errVisit
		err := walk(tree.name, markFn)
		if err != nil {
			t.Fatalf("expected no error return from Walk, got %s", err)
		}
		if len(errors) != 2 {
			t.Errorf("expected 2 errors, got %d: %s", len(errors), errors)
		}
		// the inaccessible subtrees were marked manually
		checkMarks(t, true)
		errors = errors[0:0]

		// 4) capture errors, stop after first error.
		// mark respective subtrees manually
		markTree(tree.entries[1])
		markTree(tree.entries[3])
		// correct double-marking of directory itself
		tree.entries[1].mark -= errVisit
		tree.entries[3].mark -= errVisit
		clear = false // error will stop processing
		err = walk(tree.name, markFn)
		if err == nil {
			t.Fatalf("expected error return from Walk")
		}
		if len(errors) != 1 {
			t.Errorf("expected 1 error, got %d: %s", len(errors), errors)
		}
		// the inaccessible subtrees were marked manually
		checkMarks(t, false)
		errors = errors[0:0]

		// restore permissions
		os.Chmod(filepath.Join(tree.name, tree.entries[1].name), 0770)
		os.Chmod(filepath.Join(tree.name, tree.entries[3].name), 0770)
	})
}

func touch(t *testing.T, name string) {
	f, err := os.Create(name)
	if err != nil {
		t.Fatal(err)
	}
	if err := f.Close(); err != nil {
		t.Fatal(err)
	}
}

func TestWalkSkipDirOnFile(t *testing.T) {
	td := t.TempDir()

	if err := os.MkdirAll(filepath.Join(td, "dir"), 0755); err != nil {
		t.Fatal(err)
	}
	touch(t, filepath.Join(td, "dir/foo1"))
	touch(t, filepath.Join(td, "dir/foo2"))

	sawFoo2 := false
	walker := func(path string) error {
		if strings.HasSuffix(path, "foo2") {
			sawFoo2 = true
		}
		if strings.HasSuffix(path, "foo1") {
			return filepath.SkipDir
		}
		return nil
	}
	walkFn := func(path string, _ fs.FileInfo, _ error) error { return walker(path) }
	walkDirFn := func(path string, _ fs.DirEntry, _ error) error { return walker(path) }

	check := func(t *testing.T, walk func(root string) error, root string) {
		t.Helper()
		sawFoo2 = false
		err := walk(root)
		if err != nil {
			t.Fatal(err)
		}
		if sawFoo2 {
			t.Errorf("SkipDir on file foo1 did not block processing of foo2")
		}
	}

	t.Run("Walk", func(t *testing.T) {
		Walk := func(root string) error { return filepath.Walk(td, walkFn) }
		check(t, Walk, td)
		check(t, Walk, filepath.Join(td, "dir"))
	})
	t.Run("WalkDir", func(t *testing.T) {
		WalkDir := func(root string) error { return filepath.WalkDir(td, walkDirFn) }
		check(t, WalkDir, td)
		check(t, WalkDir, filepath.Join(td, "dir"))
	})
}

func TestWalkFileError(t *testing.T) {
	td := t.TempDir()

	touch(t, filepath.Join(td, "foo"))
	touch(t, filepath.Join(td, "bar"))
	dir := filepath.Join(td, "dir")
	if err := os.MkdirAll(filepath.Join(td, "dir"), 0755); err != nil {
		t.Fatal(err)
	}
	touch(t, filepath.Join(dir, "baz"))
	touch(t, filepath.Join(dir, "stat-error"))
	defer func() {
		*filepath.LstatP = os.Lstat
	}()
	statErr := errors.New("some stat error")
	*filepath.LstatP = func(path string) (fs.FileInfo, error) {
		if strings.HasSuffix(path, "stat-error") {
			return nil, statErr
		}
		return os.Lstat(path)
	}
	got := map[string]error{}
	err := filepath.Walk(td, func(path string, fi fs.FileInfo, err error) error {
		rel, _ := filepath.Rel(td, path)
		got[filepath.ToSlash(rel)] = err
		return nil
	})
	if err != nil {
		t.Errorf("Walk error: %v", err)
	}
	want := map[string]error{
		".":              nil,
		"foo":            nil,
		"bar":            nil,
		"dir":            nil,
		"dir/baz":        nil,
		"dir/stat-error": statErr,
	}
	if !reflect.DeepEqual(got, want) {
		t.Errorf("Walked %#v; want %#v", got, want)
	}
}

var basetests = []PathTest{
	{"", "."},
	{".", "."},
	{"/.", "."},
	{"/", "/"},
	{"////", "/"},
	{"x/", "x"},
	{"abc", "abc"},
	{"abc/def", "def"},
	{"a/b/.x", ".x"},
	{"a/b/c.", "c."},
	{"a/b/c.x", "c.x"},
}

var winbasetests = []PathTest{
	{`c:\`, `\`},
	{`c:.`, `.`},
	{`c:\a\b`, `b`},
	{`c:a\b`, `b`},
	{`c:a\b\c`, `c`},
	{`\\host\share\`, `\`},
	{`\\host\share\a`, `a`},
	{`\\host\share\a\b`, `b`},
}

func TestBase(t *testing.T) {
	tests := basetests
	if runtime.GOOS == "windows" {
		// make unix tests work on windows
		for i := range tests {
			tests[i].result = filepath.Clean(tests[i].result)
		}
		// add windows specific tests
		tests = append(tests, winbasetests...)
	}
	for _, test := range tests {
		if s := filepath.Base(test.path); s != test.result {
			t.Errorf("Base(%q) = %q, want %q", test.path, s, test.result)
		}
	}
}

var dirtests = []PathTest{
	{"", "."},
	{".", "."},
	{"/.", "/"},
	{"/", "/"},
	{"////", "/"},
	{"/foo", "/"},
	{"x/", "x"},
	{"abc", "."},
	{"abc/def", "abc"},
	{"a/b/.x", "a/b"},
	{"a/b/c.", "a/b"},
	{"a/b/c.x", "a/b"},
}

var windirtests = []PathTest{
	{`c:\`, `c:\`},
	{`c:.`, `c:.`},
	{`c:\a\b`, `c:\a`},
	{`c:a\b`, `c:a`},
	{`c:a\b\c`, `c:a\b`},
	{`\\host\share`, `\\host\share`},
	{`\\host\share\`, `\\host\share\`},
	{`\\host\share\a`, `\\host\share\`},
	{`\\host\share\a\b`, `\\host\share\a`},
}

func TestDir(t *testing.T) {
	tests := dirtests
	if runtime.GOOS == "windows" {
		// make unix tests work on windows
		for i := range tests {
			tests[i].result = filepath.Clean(tests[i].result)
		}
		// add windows specific tests
		tests = append(tests, windirtests...)
	}
	for _, test := range tests {
		if s := filepath.Dir(test.path); s != test.result {
			t.Errorf("Dir(%q) = %q, want %q", test.path, s, test.result)
		}
	}
}

type IsAbsTest struct {
	path  string
	isAbs bool
}

var isabstests = []IsAbsTest{
	{"", false},
	{"/", true},
	{"/usr/bin/gcc", true},
	{"..", false},
	{"/a/../bb", true},
	{".", false},
	{"./", false},
	{"lala", false},
}

var winisabstests = []IsAbsTest{
	{`C:\`, true},
	{`c\`, false},
	{`c::`, false},
	{`c:`, false},
	{`/`, false},
	{`\`, false},
	{`\Windows`, false},
	{`c:a\b`, false},
	{`c:\a\b`, true},
	{`c:/a/b`, true},
	{`\\host\share`, true},
	{`\\host\share\`, true},
	{`\\host\share\foo`, true},
	{`//host/share/foo/bar`, true},
}

func TestIsAbs(t *testing.T) {
	var tests []IsAbsTest
	if runtime.GOOS == "windows" {
		tests = append(tests, winisabstests...)
		// All non-windows tests should fail, because they have no volume letter.
		for _, test := range isabstests {
			tests = append(tests, IsAbsTest{test.path, false})
		}
		// All non-windows test should work as intended if prefixed with volume letter.
		for _, test := range isabstests {
			tests = append(tests, IsAbsTest{"c:" + test.path, test.isAbs})
		}
		// Test reserved names.
		tests = append(tests, IsAbsTest{os.DevNull, true})
		tests = append(tests, IsAbsTest{"NUL", true})
		tests = append(tests, IsAbsTest{"nul", true})
		tests = append(tests, IsAbsTest{"CON", true})
	} else {
		tests = isabstests
	}

	for _, test := range tests {
		if r := filepath.IsAbs(test.path); r != test.isAbs {
			t.Errorf("IsAbs(%q) = %v, want %v", test.path, r, test.isAbs)
		}
	}
}

type EvalSymlinksTest struct {
	// If dest is empty, the path is created; otherwise the dest is symlinked to the path.
	path, dest string
}

var EvalSymlinksTestDirs = []EvalSymlinksTest{
	{"test", ""},
	{"test/dir", ""},
	{"test/dir/link3", "../../"},
	{"test/link1", "../test"},
	{"test/link2", "dir"},
	{"test/linkabs", "/"},
	{"test/link4", "../test2"},
	{"test2", "test/dir"},
	// Issue 23444.
	{"src", ""},
	{"src/pool", ""},
	{"src/pool/test", ""},
	{"src/versions", ""},
	{"src/versions/current", "../../version"},
	{"src/versions/v1", ""},
	{"src/versions/v1/modules", ""},
	{"src/versions/v1/modules/test", "../../../pool/test"},
	{"version", "src/versions/v1"},
}

var EvalSymlinksTests = []EvalSymlinksTest{
	{"test", "test"},
	{"test/dir", "test/dir"},
	{"test/dir/../..", "."},
	{"test/link1", "test"},
	{"test/link2", "test/dir"},
	{"test/link1/dir", "test/dir"},
	{"test/link2/..", "test"},
	{"test/dir/link3", "."},
	{"test/link2/link3/test", "test"},
	{"test/linkabs", "/"},
	{"test/link4/..", "test"},
	{"src/versions/current/modules/test", "src/pool/test"},
}

// simpleJoin builds a file name from the directory and path.
// It does not use Join because we don't want ".." to be evaluated.
func simpleJoin(dir, path string) string {
	return dir + string(filepath.Separator) + path
}

func testEvalSymlinks(t *testing.T, path, want string) {
	have, err := filepath.EvalSymlinks(path)
	if err != nil {
		t.Errorf("EvalSymlinks(%q) error: %v", path, err)
		return
	}
	if filepath.Clean(have) != filepath.Clean(want) {
		t.Errorf("EvalSymlinks(%q) returns %q, want %q", path, have, want)
	}
}

func testEvalSymlinksAfterChdir(t *testing.T, wd, path, want string) {
	cwd, err := os.Getwd()
	if err != nil {
		t.Fatal(err)
	}
	defer func() {
		err := os.Chdir(cwd)
		if err != nil {
			t.Fatal(err)
		}
	}()

	err = os.Chdir(wd)
	if err != nil {
		t.Fatal(err)
	}

	have, err := filepath.EvalSymlinks(path)
	if err != nil {
		t.Errorf("EvalSymlinks(%q) in %q directory error: %v", path, wd, err)
		return
	}
	if filepath.Clean(have) != filepath.Clean(want) {
		t.Errorf("EvalSymlinks(%q) in %q directory returns %q, want %q", path, wd, have, want)
	}
}

func TestEvalSymlinks(t *testing.T) {
	testenv.MustHaveSymlink(t)

	tmpDir := t.TempDir()

	// /tmp may itself be a symlink! Avoid the confusion, although
	// it means trusting the thing we're testing.
	var err error
	tmpDir, err = filepath.EvalSymlinks(tmpDir)
	if err != nil {
		t.Fatal("eval symlink for tmp dir:", err)
	}

	// Create the symlink farm using relative paths.
	for _, d := range EvalSymlinksTestDirs {
		var err error
		path := simpleJoin(tmpDir, d.path)
		if d.dest == "" {
			err = os.Mkdir(path, 0755)
		} else {
			err = os.Symlink(d.dest, path)
		}
		if err != nil {
			t.Fatal(err)
		}
	}

	// Evaluate the symlink farm.
	for _, test := range EvalSymlinksTests {
		path := simpleJoin(tmpDir, test.path)

		dest := simpleJoin(tmpDir, test.dest)
		if filepath.IsAbs(test.dest) || os.IsPathSeparator(test.dest[0]) {
			dest = test.dest
		}
		testEvalSymlinks(t, path, dest)

		// test EvalSymlinks(".")
		testEvalSymlinksAfterChdir(t, path, ".", ".")

		// test EvalSymlinks("C:.") on Windows
		if runtime.GOOS == "windows" {
			volDot := filepath.VolumeName(tmpDir) + "."
			testEvalSymlinksAfterChdir(t, path, volDot, volDot)
		}

		// test EvalSymlinks(".."+path)
		dotdotPath := simpleJoin("..", test.dest)
		if filepath.IsAbs(test.dest) || os.IsPathSeparator(test.dest[0]) {
			dotdotPath = test.dest
		}
		testEvalSymlinksAfterChdir(t,
			simpleJoin(tmpDir, "test"),
			simpleJoin("..", test.path),
			dotdotPath)

		// test EvalSymlinks(p) where p is relative path
		testEvalSymlinksAfterChdir(t, tmpDir, test.path, test.dest)
	}
}

func TestEvalSymlinksIsNotExist(t *testing.T) {
	testenv.MustHaveSymlink(t)

	defer chtmpdir(t)()

	_, err := filepath.EvalSymlinks("notexist")
	if !os.IsNotExist(err) {
		t.Errorf("expected the file is not found, got %v\n", err)
	}

	err = os.Symlink("notexist", "link")
	if err != nil {
		t.Fatal(err)
	}
	defer os.Remove("link")

	_, err = filepath.EvalSymlinks("link")
	if !os.IsNotExist(err) {
		t.Errorf("expected the file is not found, got %v\n", err)
	}
}

func TestIssue13582(t *testing.T) {
	testenv.MustHaveSymlink(t)

	tmpDir := t.TempDir()

	dir := filepath.Join(tmpDir, "dir")
	err := os.Mkdir(dir, 0755)
	if err != nil {
		t.Fatal(err)
	}
	linkToDir := filepath.Join(tmpDir, "link_to_dir")
	err = os.Symlink(dir, linkToDir)
	if err != nil {
		t.Fatal(err)
	}
	file := filepath.Join(linkToDir, "file")
	err = os.WriteFile(file, nil, 0644)
	if err != nil {
		t.Fatal(err)
	}
	link1 := filepath.Join(linkToDir, "link1")
	err = os.Symlink(file, link1)
	if err != nil {
		t.Fatal(err)
	}
	link2 := filepath.Join(linkToDir, "link2")
	err = os.Symlink(link1, link2)
	if err != nil {
		t.Fatal(err)
	}

	// /tmp may itself be a symlink!
	realTmpDir, err := filepath.EvalSymlinks(tmpDir)
	if err != nil {
		t.Fatal(err)
	}
	realDir := filepath.Join(realTmpDir, "dir")
	realFile := filepath.Join(realDir, "file")

	tests := []struct {
		path, want string
	}{
		{dir, realDir},
		{linkToDir, realDir},
		{file, realFile},
		{link1, realFile},
		{link2, realFile},
	}
	for i, test := range tests {
		have, err := filepath.EvalSymlinks(test.path)
		if err != nil {
			t.Fatal(err)
		}
		if have != test.want {
			t.Errorf("test#%d: EvalSymlinks(%q) returns %q, want %q", i, test.path, have, test.want)
		}
	}
}

// Test directories relative to temporary directory.
// The tests are run in absTestDirs[0].
var absTestDirs = []string{
	"a",
	"a/b",
	"a/b/c",
}

// Test paths relative to temporary directory. $ expands to the directory.
// The tests are run in absTestDirs[0].
// We create absTestDirs first.
var absTests = []string{
	".",
	"b",
	"b/",
	"../a",
	"../a/b",
	"../a/b/./c/../../.././a",
	"../a/b/./c/../../.././a/",
	"$",
	"$/.",
	"$/a/../a/b",
	"$/a/b/c/../../.././a",
	"$/a/b/c/../../.././a/",
}

func TestAbs(t *testing.T) {
	root := t.TempDir()
	wd, err := os.Getwd()
	if err != nil {
		t.Fatal("getwd failed: ", err)
	}
	err = os.Chdir(root)
	if err != nil {
		t.Fatal("chdir failed: ", err)
	}
	defer os.Chdir(wd)

	for _, dir := range absTestDirs {
		err = os.Mkdir(dir, 0777)
		if err != nil {
			t.Fatal("Mkdir failed: ", err)
		}
	}

	if runtime.GOOS == "windows" {
		vol := filepath.VolumeName(root)
		var extra []string
		for _, path := range absTests {
			if strings.Contains(path, "$") {
				continue
			}
			path = vol + path
			extra = append(extra, path)
		}
		absTests = append(absTests, extra...)
	}

	err = os.Chdir(absTestDirs[0])
	if err != nil {
		t.Fatal("chdir failed: ", err)
	}

	for _, path := range absTests {
		path = strings.ReplaceAll(path, "$", root)
		info, err := os.Stat(path)
		if err != nil {
			t.Errorf("%s: %s", path, err)
			continue
		}

		abspath, err := filepath.Abs(path)
		if err != nil {
			t.Errorf("Abs(%q) error: %v", path, err)
			continue
		}
		absinfo, err := os.Stat(abspath)
		if err != nil || !os.SameFile(absinfo, info) {
			t.Errorf("Abs(%q)=%q, not the same file", path, abspath)
		}
		if !filepath.IsAbs(abspath) {
			t.Errorf("Abs(%q)=%q, not an absolute path", path, abspath)
		}
		if filepath.IsAbs(abspath) && abspath != filepath.Clean(abspath) {
			t.Errorf("Abs(%q)=%q, isn't clean", path, abspath)
		}
	}
}

// Empty path needs to be special-cased on Windows. See golang.org/issue/24441.
// We test it separately from all other absTests because the empty string is not
// a valid path, so it can't be used with os.Stat.
func TestAbsEmptyString(t *testing.T) {
	root := t.TempDir()

	wd, err := os.Getwd()
	if err != nil {
		t.Fatal("getwd failed: ", err)
	}
	err = os.Chdir(root)
	if err != nil {
		t.Fatal("chdir failed: ", err)
	}
	defer os.Chdir(wd)

	info, err := os.Stat(root)
	if err != nil {
		t.Fatalf("%s: %s", root, err)
	}

	abspath, err := filepath.Abs("")
	if err != nil {
		t.Fatalf(`Abs("") error: %v`, err)
	}
	absinfo, err := os.Stat(abspath)
	if err != nil || !os.SameFile(absinfo, info) {
		t.Errorf(`Abs("")=%q, not the same file`, abspath)
	}
	if !filepath.IsAbs(abspath) {
		t.Errorf(`Abs("")=%q, not an absolute path`, abspath)
	}
	if filepath.IsAbs(abspath) && abspath != filepath.Clean(abspath) {
		t.Errorf(`Abs("")=%q, isn't clean`, abspath)
	}
}

type RelTests struct {
	root, path, want string
}

var reltests = []RelTests{
	{"a/b", "a/b", "."},
	{"a/b/.", "a/b", "."},
	{"a/b", "a/b/.", "."},
	{"./a/b", "a/b", "."},
	{"a/b", "./a/b", "."},
	{"ab/cd", "ab/cde", "../cde"},
	{"ab/cd", "ab/c", "../c"},
	{"a/b", "a/b/c/d", "c/d"},
	{"a/b", "a/b/../c", "../c"},
	{"a/b/../c", "a/b", "../b"},
	{"a/b/c", "a/c/d", "../../c/d"},
	{"a/b", "c/d", "../../c/d"},
	{"a/b/c/d", "a/b", "../.."},
	{"a/b/c/d", "a/b/", "../.."},
	{"a/b/c/d/", "a/b", "../.."},
	{"a/b/c/d/", "a/b/", "../.."},
	{"../../a/b", "../../a/b/c/d", "c/d"},
	{"/a/b", "/a/b", "."},
	{"/a/b/.", "/a/b", "."},
	{"/a/b", "/a/b/.", "."},
	{"/ab/cd", "/ab/cde", "../cde"},
	{"/ab/cd", "/ab/c", "../c"},
	{"/a/b", "/a/b/c/d", "c/d"},
	{"/a/b", "/a/b/../c", "../c"},
	{"/a/b/../c", "/a/b", "../b"},
	{"/a/b/c", "/a/c/d", "../../c/d"},
	{"/a/b", "/c/d", "../../c/d"},
	{"/a/b/c/d", "/a/b", "../.."},
	{"/a/b/c/d", "/a/b/", "../.."},
	{"/a/b/c/d/", "/a/b", "../.."},
	{"/a/b/c/d/", "/a/b/", "../.."},
	{"/../../a/b", "/../../a/b/c/d", "c/d"},
	{".", "a/b", "a/b"},
	{".", "..", ".."},

	// can't do purely lexically
	{"..", ".", "err"},
	{"..", "a", "err"},
	{"../..", "..", "err"},
	{"a", "/a", "err"},
	{"/a", "a", "err"},
}

var winreltests = []RelTests{
	{`C:a\b\c`, `C:a/b/d`, `..\d`},
	{`C:\`, `D:\`, `err`},
	{`C:`, `D:`, `err`},
	{`C:\Projects`, `c:\projects\src`, `src`},
	{`C:\Projects`, `c:\projects`, `.`},
	{`C:\Projects\a\..`, `c:\projects`, `.`},
	{`\\host\share`, `\\host\share\file.txt`, `file.txt`},
}

func TestRel(t *testing.T) {
	tests := append([]RelTests{}, reltests...)
	if runtime.GOOS == "windows" {
		for i := range tests {
			tests[i].want = filepath.FromSlash(tests[i].want)
		}
		tests = append(tests, winreltests...)
	}
	for _, test := range tests {
		got, err := filepath.Rel(test.root, test.path)
		if test.want == "err" {
			if err == nil {
				t.Errorf("Rel(%q, %q)=%q, want error", test.root, test.path, got)
			}
			continue
		}
		if err != nil {
			t.Errorf("Rel(%q, %q): want %q, got error: %s", test.root, test.path, test.want, err)
		}
		if got != test.want {
			t.Errorf("Rel(%q, %q)=%q, want %q", test.root, test.path, got, test.want)
		}
	}
}

type VolumeNameTest struct {
	path string
	vol  string
}

var volumenametests = []VolumeNameTest{
	{`c:/foo/bar`, `c:`},
	{`c:`, `c:`},
	{`2:`, ``},
	{``, ``},
	{`\\\host`, ``},
	{`\\\host\`, ``},
	{`\\\host\share`, ``},
	{`\\\host\\share`, ``},
	{`\\host`, ``},
	{`//host`, ``},
	{`\\host\`, ``},
	{`//host/`, ``},
	{`\\host\share`, `\\host\share`},
	{`//host/share`, `//host/share`},
	{`\\host\share\`, `\\host\share`},
	{`//host/share/`, `//host/share`},
	{`\\host\share\foo`, `\\host\share`},
	{`//host/share/foo`, `//host/share`},
	{`\\host\share\\foo\\\bar\\\\baz`, `\\host\share`},
	{`//host/share//foo///bar////baz`, `//host/share`},
	{`\\host\share\foo\..\bar`, `\\host\share`},
	{`//host/share/foo/../bar`, `//host/share`},
}

func TestVolumeName(t *testing.T) {
	if runtime.GOOS != "windows" {
		return
	}
	for _, v := range volumenametests {
		if vol := filepath.VolumeName(v.path); vol != v.vol {
			t.Errorf("VolumeName(%q)=%q, want %q", v.path, vol, v.vol)
		}
	}
}

func TestDriveLetterInEvalSymlinks(t *testing.T) {
	if runtime.GOOS != "windows" {
		return
	}
	wd, _ := os.Getwd()
	if len(wd) < 3 {
		t.Errorf("Current directory path %q is too short", wd)
	}
	lp := strings.ToLower(wd)
	up := strings.ToUpper(wd)
	flp, err := filepath.EvalSymlinks(lp)
	if err != nil {
		t.Fatalf("EvalSymlinks(%q) failed: %q", lp, err)
	}
	fup, err := filepath.EvalSymlinks(up)
	if err != nil {
		t.Fatalf("EvalSymlinks(%q) failed: %q", up, err)
	}
	if flp != fup {
		t.Errorf("Results of EvalSymlinks do not match: %q and %q", flp, fup)
	}
}

func TestBug3486(t *testing.T) { // https://golang.org/issue/3486
	if runtime.GOOS == "ios" {
		t.Skipf("skipping on %s/%s", runtime.GOOS, runtime.GOARCH)
	}
	root, err := filepath.EvalSymlinks(runtime.GOROOT() + "/test")
	if err != nil {
		t.Fatal(err)
	}
	bugs := filepath.Join(root, "fixedbugs")
	ken := filepath.Join(root, "ken")
	seenBugs := false
	seenKen := false
	err = filepath.Walk(root, func(pth string, info fs.FileInfo, err error) error {
		if err != nil {
			t.Fatal(err)
		}

		switch pth {
		case bugs:
			seenBugs = true
			return filepath.SkipDir
		case ken:
			if !seenBugs {
				t.Fatal("filepath.Walk out of order - ken before fixedbugs")
			}
			seenKen = true
		}
		return nil
	})
	if err != nil {
		t.Fatal(err)
	}
	if !seenKen {
		t.Fatalf("%q not seen", ken)
	}
}

func testWalkSymlink(t *testing.T, mklink func(target, link string) error) {
	tmpdir := t.TempDir()

	wd, err := os.Getwd()
	if err != nil {
		t.Fatal(err)
	}
	defer os.Chdir(wd)

	err = os.Chdir(tmpdir)
	if err != nil {
		t.Fatal(err)
	}

	err = mklink(tmpdir, "link")
	if err != nil {
		t.Fatal(err)
	}

	var visited []string
	err = filepath.Walk(tmpdir, func(path string, info fs.FileInfo, err error) error {
		if err != nil {
			t.Fatal(err)
		}
		rel, err := filepath.Rel(tmpdir, path)
		if err != nil {
			t.Fatal(err)
		}
		visited = append(visited, rel)
		return nil
	})
	if err != nil {
		t.Fatal(err)
	}
	sort.Strings(visited)
	want := []string{".", "link"}
	if fmt.Sprintf("%q", visited) != fmt.Sprintf("%q", want) {
		t.Errorf("unexpected paths visited %q, want %q", visited, want)
	}
}

func TestWalkSymlink(t *testing.T) {
	testenv.MustHaveSymlink(t)
	testWalkSymlink(t, os.Symlink)
}

func TestIssue29372(t *testing.T) {
	tmpDir := t.TempDir()

	path := filepath.Join(tmpDir, "file.txt")
	err := os.WriteFile(path, nil, 0644)
	if err != nil {
		t.Fatal(err)
	}

	pathSeparator := string(filepath.Separator)
	tests := []string{
		path + strings.Repeat(pathSeparator, 1),
		path + strings.Repeat(pathSeparator, 2),
		path + strings.Repeat(pathSeparator, 1) + ".",
		path + strings.Repeat(pathSeparator, 2) + ".",
		path + strings.Repeat(pathSeparator, 1) + "..",
		path + strings.Repeat(pathSeparator, 2) + "..",
	}

	for i, test := range tests {
		_, err = filepath.EvalSymlinks(test)
		if err != syscall.ENOTDIR {
			t.Fatalf("test#%d: want %q, got %q", i, syscall.ENOTDIR, err)
		}
	}
}

// Issue 30520 part 1.
func TestEvalSymlinksAboveRoot(t *testing.T) {
	testenv.MustHaveSymlink(t)

	t.Parallel()

	tmpDir := t.TempDir()

	evalTmpDir, err := filepath.EvalSymlinks(tmpDir)
	if err != nil {
		t.Fatal(err)
	}

	if err := os.Mkdir(filepath.Join(evalTmpDir, "a"), 0777); err != nil {
		t.Fatal(err)
	}
	if err := os.Symlink(filepath.Join(evalTmpDir, "a"), filepath.Join(evalTmpDir, "b")); err != nil {
		t.Fatal(err)
	}
	if err := os.WriteFile(filepath.Join(evalTmpDir, "a", "file"), nil, 0666); err != nil {
		t.Fatal(err)
	}

	// Count the number of ".." elements to get to the root directory.
	vol := filepath.VolumeName(evalTmpDir)
	c := strings.Count(evalTmpDir[len(vol):], string(os.PathSeparator))
	var dd []string
	for i := 0; i < c+2; i++ {
		dd = append(dd, "..")
	}

	wantSuffix := strings.Join([]string{"a", "file"}, string(os.PathSeparator))

	// Try different numbers of "..".
	for _, i := range []int{c, c + 1, c + 2} {
		check := strings.Join([]string{evalTmpDir, strings.Join(dd[:i], string(os.PathSeparator)), evalTmpDir[len(vol)+1:], "b", "file"}, string(os.PathSeparator))
		resolved, err := filepath.EvalSymlinks(check)
		switch {
		case runtime.GOOS == "darwin" && errors.Is(err, fs.ErrNotExist):
			// On darwin, the temp dir is sometimes cleaned up mid-test (issue 37910).
			testenv.SkipFlaky(t, 37910)
		case err != nil:
			t.Errorf("EvalSymlinks(%q) failed: %v", check, err)
		case !strings.HasSuffix(resolved, wantSuffix):
			t.Errorf("EvalSymlinks(%q) = %q does not end with %q", check, resolved, wantSuffix)
		default:
			t.Logf("EvalSymlinks(%q) = %q", check, resolved)
		}
	}
}

// Issue 30520 part 2.
func TestEvalSymlinksAboveRootChdir(t *testing.T) {
	testenv.MustHaveSymlink(t)

	tmpDir, err := os.MkdirTemp("", "TestEvalSymlinksAboveRootChdir")
	if err != nil {
		t.Fatal(err)
	}
	defer os.RemoveAll(tmpDir)
	chdir(t, tmpDir)

	subdir := filepath.Join("a", "b")
	if err := os.MkdirAll(subdir, 0777); err != nil {
		t.Fatal(err)
	}
	if err := os.Symlink(subdir, "c"); err != nil {
		t.Fatal(err)
	}
	if err := os.WriteFile(filepath.Join(subdir, "file"), nil, 0666); err != nil {
		t.Fatal(err)
	}

	subdir = filepath.Join("d", "e", "f")
	if err := os.MkdirAll(subdir, 0777); err != nil {
		t.Fatal(err)
	}
	if err := os.Chdir(subdir); err != nil {
		t.Fatal(err)
	}

	check := filepath.Join("..", "..", "..", "c", "file")
	wantSuffix := filepath.Join("a", "b", "file")
	if resolved, err := filepath.EvalSymlinks(check); err != nil {
		t.Errorf("EvalSymlinks(%q) failed: %v", check, err)
	} else if !strings.HasSuffix(resolved, wantSuffix) {
		t.Errorf("EvalSymlinks(%q) = %q does not end with %q", check, resolved, wantSuffix)
	} else {
		t.Logf("EvalSymlinks(%q) = %q", check, resolved)
	}
}
