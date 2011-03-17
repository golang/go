// Copyright 2009 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

package filepath_test

import (
	"os"
	"path/filepath"
	"reflect"
	"runtime"
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

func TestClean(t *testing.T) {
	for _, test := range cleantests {
		if s := filepath.ToSlash(filepath.Clean(test.path)); s != test.result {
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

func TestSplit(t *testing.T) {
	var splittests []SplitTest
	splittests = unixsplittests
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
		&Node{"a", nil, 0},
		&Node{"b", []*Node{}, 0},
		&Node{"c", nil, 0},
		&Node{
			"d",
			[]*Node{
				&Node{"x", nil, 0},
				&Node{"y", []*Node{}, 0},
				&Node{
					"z",
					[]*Node{
						&Node{"u", nil, 0},
						&Node{"v", nil, 0},
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
			fd, err := os.Open(path, os.O_CREAT, 0660)
			if err != nil {
				t.Errorf("makeTree: %v", err)
			}
			fd.Close()
		} else {
			os.Mkdir(path, 0770)
		}
	})
}

func markTree(n *Node) { walkTree(n, "", func(path string, n *Node) { n.mark++ }) }

func checkMarks(t *testing.T) {
	walkTree(tree, tree.name, func(path string, n *Node) {
		if n.mark != 1 {
			t.Errorf("node %s mark = %d; expected 1", path, n.mark)
		}
		n.mark = 0
	})
}

// Assumes that each node name is unique. Good enough for a test.
func mark(name string) {
	name = filepath.ToSlash(name)
	walkTree(tree, tree.name, func(path string, n *Node) {
		if n.name == name {
			n.mark++
		}
	})
}

type TestVisitor struct{}

func (v *TestVisitor) VisitDir(path string, f *os.FileInfo) bool {
	mark(f.Name)
	return true
}

func (v *TestVisitor) VisitFile(path string, f *os.FileInfo) {
	mark(f.Name)
}

func TestWalk(t *testing.T) {
	// TODO(brainman): enable test once Windows version is implemented.
	if runtime.GOOS == "windows" {
		return
	}
	makeTree(t)

	// 1) ignore error handling, expect none
	v := &TestVisitor{}
	filepath.Walk(tree.name, v, nil)
	checkMarks(t)

	// 2) handle errors, expect none
	errors := make(chan os.Error, 64)
	filepath.Walk(tree.name, v, errors)
	select {
	case err := <-errors:
		t.Errorf("no error expected, found: %s", err)
	default:
		// ok
	}
	checkMarks(t)

	if os.Getuid() > 0 {
		// introduce 2 errors: chmod top-level directories to 0
		os.Chmod(filepath.Join(tree.name, tree.entries[1].name), 0)
		os.Chmod(filepath.Join(tree.name, tree.entries[3].name), 0)
		// mark respective subtrees manually
		markTree(tree.entries[1])
		markTree(tree.entries[3])
		// correct double-marking of directory itself
		tree.entries[1].mark--
		tree.entries[3].mark--

		// 3) handle errors, expect two
		errors = make(chan os.Error, 64)
		os.Chmod(filepath.Join(tree.name, tree.entries[1].name), 0)
		filepath.Walk(tree.name, v, errors)
	Loop:
		for i := 1; i <= 2; i++ {
			select {
			case <-errors:
				// ok
			default:
				t.Errorf("%d. error expected, none found", i)
				break Loop
			}
		}
		select {
		case err := <-errors:
			t.Errorf("only two errors expected, found 3rd: %v", err)
		default:
			// ok
		}
		// the inaccessible subtrees were marked manually
		checkMarks(t)
	}

	// cleanup
	os.Chmod(filepath.Join(tree.name, tree.entries[1].name), 0770)
	os.Chmod(filepath.Join(tree.name, tree.entries[3].name), 0770)
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

func TestBase(t *testing.T) {
	for _, test := range basetests {
		if s := filepath.ToSlash(filepath.Base(test.path)); s != test.result {
			t.Errorf("Base(%q) = %q, want %q", test.path, s, test.result)
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
	{`/`, true},
	{`\`, true},
	{`\Windows`, true},
}

func TestIsAbs(t *testing.T) {
	if runtime.GOOS == "windows" {
		isabstests = append(isabstests, winisabstests...)
	}
	for _, test := range isabstests {
		if r := filepath.IsAbs(test.path); r != test.isAbs {
			t.Errorf("IsAbs(%q) = %v, want %v", test.path, r, test.isAbs)
		}
	}
}

type EvalSymlinksTest struct {
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

func TestEvalSymlinks(t *testing.T) {
	defer os.RemoveAll("test")
	for _, d := range EvalSymlinksTestDirs {
		var err os.Error
		if d.dest == "" {
			err = os.Mkdir(d.path, 0755)
		} else {
			err = os.Symlink(d.dest, d.path)
		}
		if err != nil {
			t.Fatal(err)
		}
	}
	// relative
	for _, d := range EvalSymlinksTests {
		if p, err := filepath.EvalSymlinks(d.path); err != nil {
			t.Errorf("EvalSymlinks(%v) error: %v", d.path, err)
		} else if p != d.dest {
			t.Errorf("EvalSymlinks(%v)=%v, want %v", d.path, p, d.dest)
		}
	}
	// absolute
	testroot := filepath.Join(os.Getenv("GOROOT"), "src", "pkg", "path", "filepath")
	for _, d := range EvalSymlinksTests {
		a := EvalSymlinksTest{
			filepath.Join(testroot, d.path),
			filepath.Join(testroot, d.dest),
		}
		if p, err := filepath.EvalSymlinks(a.path); err != nil {
			t.Errorf("EvalSymlinks(%v) error: %v", a.path, err)
		} else if p != a.dest {
			t.Errorf("EvalSymlinks(%v)=%v, want %v", a.path, p, a.dest)
		}
	}
}
