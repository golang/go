// Copyright 2009 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

package filepath_test

import (
	"io/ioutil"
	"os"
	"path/filepath"
	"reflect"
	"runtime"
	"strings"
	"testing"
)

type PathTest struct {
	path, result string
}

var cleantests = []PathTest{
	// Already clean
	{"", "."},
	{"abc", "abc"},
	{"abc/def", "abc/def"},
	{"a/b/c", "a/b/c"},
	{".", "."},
	{"..", ".."},
	{"../..", "../.."},
	{"../../abc", "../../abc"},
	{"/abc", "/abc"},
	{"/", "/"},

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

func TestSplitList(t *testing.T) {
	for _, test := range splitlisttests {
		if l := filepath.SplitList(test.list); !reflect.DeepEqual(l, test.result) {
			t.Errorf("SplitList(%q) = %s, want %s", test.list, l, test.result)
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
	{[]string{"a"}, "a"},

	// two parameters
	{[]string{"a", "b"}, "a/b"},
	{[]string{"a", ""}, "a"},
	{[]string{"", "b"}, "b"},
	{[]string{"/", "a"}, "/a"},
	{[]string{"/", ""}, "/"},
	{[]string{"a/", "b"}, "a/b"},
	{[]string{"a/", ""}, "a"},
	{[]string{"", ""}, ""},
}

var winjointests = []JoinTest{
	{[]string{`directory`, `file`}, `directory\file`},
	{[]string{`C:\Windows\`, `System32`}, `C:\Windows\System32`},
	{[]string{`C:\Windows\`, ``}, `C:\Windows`},
	{[]string{`C:\`, `Windows`}, `C:\Windows`},
	{[]string{`C:`, `Windows`}, `C:\Windows`},
	{[]string{`\\host\share`, `foo`}, `\\host\share\foo`},
	{[]string{`//host/share`, `foo/bar`}, `\\host\share\foo\bar`},
}

// join takes a []string and passes it to Join.
func join(elem []string, args ...string) string {
	args = elem
	return filepath.Join(args...)
}

func TestJoin(t *testing.T) {
	if runtime.GOOS == "windows" {
		jointests = append(jointests, winjointests...)
	}
	for _, test := range jointests {
		if p := join(test.elem); p != filepath.FromSlash(test.path) {
			t.Errorf("join(%q) = %q, want %q", test.elem, p, test.path)
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
func mark(path string, info os.FileInfo, err error, errors *[]error, clear bool) error {
	if err != nil {
		*errors = append(*errors, err)
		if clear {
			return nil
		}
		return err
	}
	name := info.Name()
	walkTree(tree, tree.name, func(path string, n *Node) {
		if n.name == name {
			n.mark++
		}
	})
	return nil
}

func TestWalk(t *testing.T) {
	makeTree(t)
	errors := make([]error, 0, 10)
	clear := true
	markFn := func(path string, info os.FileInfo, err error) error {
		return mark(path, info, err, &errors, clear)
	}
	// Expect no errors.
	err := filepath.Walk(tree.name, markFn)
	if err != nil {
		t.Fatalf("no error expected, found: %s", err)
	}
	if len(errors) != 0 {
		t.Fatalf("unexpected errors: %s", errors)
	}
	checkMarks(t, true)
	errors = errors[0:0]

	// Test permission errors.  Only possible if we're not root
	// and only on some file systems (AFS, FAT).  To avoid errors during
	// all.bash on those file systems, skip during go test -short.
	if os.Getuid() > 0 && !testing.Short() {
		// introduce 2 errors: chmod top-level directories to 0
		os.Chmod(filepath.Join(tree.name, tree.entries[1].name), 0)
		os.Chmod(filepath.Join(tree.name, tree.entries[3].name), 0)

		// 3) capture errors, expect two.
		// mark respective subtrees manually
		markTree(tree.entries[1])
		markTree(tree.entries[3])
		// correct double-marking of directory itself
		tree.entries[1].mark--
		tree.entries[3].mark--
		err := filepath.Walk(tree.name, markFn)
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
		tree.entries[1].mark--
		tree.entries[3].mark--
		clear = false // error will stop processing
		err = filepath.Walk(tree.name, markFn)
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
	}

	// cleanup
	if err := os.RemoveAll(tree.name); err != nil {
		t.Errorf("removeTree: %v", err)
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
		for i, _ := range tests {
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
	{`\\host\share\`, `\\host\share\`},
	{`\\host\share\a`, `\\host\share\`},
	{`\\host\share\a\b`, `\\host\share\a`},
}

func TestDir(t *testing.T) {
	tests := dirtests
	if runtime.GOOS == "windows" {
		// make unix tests work on windows
		for i, _ := range tests {
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
}

var EvalSymlinksAbsWindowsTests = []EvalSymlinksTest{
	{`c:\`, `c:\`},
}

// simpleJoin builds a file name from the directory and path.
// It does not use Join because we don't want ".." to be evaluated.
func simpleJoin(dir, path string) string {
	return dir + string(filepath.Separator) + path
}

func TestEvalSymlinks(t *testing.T) {
	tmpDir, err := ioutil.TempDir("", "evalsymlink")
	if err != nil {
		t.Fatal("creating temp dir:", err)
	}
	defer os.RemoveAll(tmpDir)

	// /tmp may itself be a symlink! Avoid the confusion, although
	// it means trusting the thing we're testing.
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
			if runtime.GOOS != "windows" {
				err = os.Symlink(d.dest, path)
			}
		}
		if err != nil {
			t.Fatal(err)
		}
	}

	var tests []EvalSymlinksTest
	if runtime.GOOS == "windows" {
		for _, d := range EvalSymlinksTests {
			if d.path == d.dest {
				// will test only real files and directories
				tests = append(tests, d)
			}
		}
	} else {
		tests = EvalSymlinksTests
	}

	// Evaluate the symlink farm.
	for _, d := range tests {
		path := simpleJoin(tmpDir, d.path)
		dest := simpleJoin(tmpDir, d.dest)
		if p, err := filepath.EvalSymlinks(path); err != nil {
			t.Errorf("EvalSymlinks(%q) error: %v", d.path, err)
		} else if filepath.Clean(p) != filepath.Clean(dest) {
			t.Errorf("Clean(%q)=%q, want %q", path, p, dest)
		}
	}
}

// Test paths relative to $GOROOT/src
var abstests = []string{
	"../AUTHORS",
	"pkg/../../AUTHORS",
	"Make.inc",
	"pkg/math",
	".",
	"$GOROOT/src/Make.inc",
	"$GOROOT/src/../src/Make.inc",
	"$GOROOT/misc/cgo",
	"$GOROOT",
}

func TestAbs(t *testing.T) {
	t.Logf("test needs to be rewritten; disabled")
	return

	oldwd, err := os.Getwd()
	if err != nil {
		t.Fatal("Getwd failed: " + err.Error())
	}
	defer os.Chdir(oldwd)
	goroot := os.Getenv("GOROOT")
	cwd := filepath.Join(goroot, "src")
	os.Chdir(cwd)
	for _, path := range abstests {
		path = strings.Replace(path, "$GOROOT", goroot, -1)
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
		if filepath.IsAbs(path) && abspath != filepath.Clean(path) {
			t.Errorf("Abs(%q)=%q, isn't clean", path, abspath)
		}
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
