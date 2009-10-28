// Copyright 2009 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

package path

import (
	"os";
	"testing";
)

type CleanTest struct {
	path, clean string;
}

var cleantests = []CleanTest{
	// Already clean
	CleanTest{"", "."},
	CleanTest{"abc", "abc"},
	CleanTest{"abc/def", "abc/def"},
	CleanTest{"a/b/c", "a/b/c"},
	CleanTest{".", "."},
	CleanTest{"..", ".."},
	CleanTest{"../..", "../.."},
	CleanTest{"../../abc", "../../abc"},
	CleanTest{"/abc", "/abc"},
	CleanTest{"/", "/"},

	// Remove trailing slash
	CleanTest{"abc/", "abc"},
	CleanTest{"abc/def/", "abc/def"},
	CleanTest{"a/b/c/", "a/b/c"},
	CleanTest{"./", "."},
	CleanTest{"../", ".."},
	CleanTest{"../../", "../.."},
	CleanTest{"/abc/", "/abc"},

	// Remove doubled slash
	CleanTest{"abc//def//ghi", "abc/def/ghi"},
	CleanTest{"//abc", "/abc"},
	CleanTest{"///abc", "/abc"},
	CleanTest{"//abc//", "/abc"},
	CleanTest{"abc//", "abc"},

	// Remove . elements
	CleanTest{"abc/./def", "abc/def"},
	CleanTest{"/./abc/def", "/abc/def"},
	CleanTest{"abc/.", "abc"},

	// Remove .. elements
	CleanTest{"abc/def/ghi/../jkl", "abc/def/jkl"},
	CleanTest{"abc/def/../ghi/../jkl", "abc/jkl"},
	CleanTest{"abc/def/..", "abc"},
	CleanTest{"abc/def/../..", "."},
	CleanTest{"/abc/def/../..", "/"},
	CleanTest{"abc/def/../../..", ".."},
	CleanTest{"/abc/def/../../..", "/"},
	CleanTest{"abc/def/../../../ghi/jkl/../../../mno", "../../mno"},

	// Combinations
	CleanTest{"abc/./../def", "def"},
	CleanTest{"abc//./../def", "def"},
	CleanTest{"abc/../../././../def", "../../def"},
}

func TestClean(t *testing.T) {
	for _, test := range cleantests {
		if s := Clean(test.path); s != test.clean {
			t.Errorf("Clean(%q) = %q, want %q", test.path, s, test.clean);
		}
	}
}

type SplitTest struct {
	path, dir, file string;
}

var splittests = []SplitTest{
	SplitTest{"a/b", "a/", "b"},
	SplitTest{"a/b/", "a/b/", ""},
	SplitTest{"a/", "a/", ""},
	SplitTest{"a", "", "a"},
	SplitTest{"/", "/", ""},
}

func TestSplit(t *testing.T) {
	for _, test := range splittests {
		if d, f := Split(test.path); d != test.dir || f != test.file {
			t.Errorf("Split(%q) = %q, %q, want %q, %q", test.path, d, f, test.dir, test.file);
		}
	}
}

type JoinTest struct {
	dir, file, path string;
}

var jointests = []JoinTest{
	JoinTest{"a", "b", "a/b"},
	JoinTest{"a", "", "a"},
	JoinTest{"", "b", "b"},
	JoinTest{"/", "a", "/a"},
	JoinTest{"/", "", "/"},
	JoinTest{"a/", "b", "a/b"},
	JoinTest{"a/", "", "a"},
}

func TestJoin(t *testing.T) {
	for _, test := range jointests {
		if p := Join(test.dir, test.file); p != test.path {
			t.Errorf("Join(%q, %q) = %q, want %q", test.dir, test.file, p, test.path);
		}
	}
}

type ExtTest struct {
	path, ext string;
}

var exttests = []ExtTest{
	ExtTest{"path.go", ".go"},
	ExtTest{"path.pb.go", ".go"},
	ExtTest{"a.dir/b", ""},
	ExtTest{"a.dir/b.go", ".go"},
	ExtTest{"a.dir/", ""},
}

func TestExt(t *testing.T) {
	for _, test := range exttests {
		if x := Ext(test.path); x != test.ext {
			t.Errorf("Ext(%q) = %q, want %q", test.path, x, test.ext);
		}
	}
}

type Node struct {
	name	string;
	entries	[]*Node;	// nil if the entry is a file
	mark	int;
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
	f(path, n);
	for _, e := range n.entries {
		walkTree(e, Join(path, e.name), f);
	}
}

func makeTree(t *testing.T) {
	walkTree(tree, tree.name, func(path string, n *Node) {
		if n.entries == nil {
			fd, err := os.Open(path, os.O_CREAT, 0660);
			if err != nil {
				t.Errorf("makeTree: %v", err);
			}
			fd.Close();
		} else {
			os.Mkdir(path, 0770);
		}
	});
}

func markTree(n *Node) {
	walkTree(n, "", func(path string, n *Node) { n.mark++ });
}

func checkMarks(t *testing.T) {
	walkTree(tree, tree.name, func(path string, n *Node) {
		if n.mark != 1 {
			t.Errorf("node %s mark = %d; expected 1", path, n.mark);
		}
		n.mark = 0;
	});
}

// Assumes that each node name is unique. Good enough for a test.
func mark(name string) {
	walkTree(tree, tree.name, func(path string, n *Node) {
		if n.name == name {
			n.mark++;
		}
	});
}

type TestVisitor struct{}

func (v *TestVisitor) VisitDir(path string, d *os.Dir) bool {
	mark(d.Name);
	return true;
}

func (v *TestVisitor) VisitFile(path string, d *os.Dir) {
	mark(d.Name);
}

func TestWalk(t *testing.T) {
	makeTree(t);

	// 1) ignore error handling, expect none
	v := &TestVisitor{};
	Walk(tree.name, v, nil);
	checkMarks(t);

	// 2) handle errors, expect none
	errors := make(chan os.Error, 64);
	Walk(tree.name, v, errors);
	if err, ok := <-errors; ok {
		t.Errorf("no error expected, found: s", err);
	}
	checkMarks(t);

	// introduce 2 errors: chmod top-level directories to 0
	os.Chmod(Join(tree.name, tree.entries[1].name), 0);
	os.Chmod(Join(tree.name, tree.entries[3].name), 0);
	// mark respective subtrees manually
	markTree(tree.entries[1]);
	markTree(tree.entries[3]);
	// correct double-marking of directory itself
	tree.entries[1].mark--;
	tree.entries[3].mark--;

	// 3) handle errors, expect two
	errors = make(chan os.Error, 64);
	os.Chmod(Join(tree.name, tree.entries[1].name), 0);
	Walk(tree.name, v, errors);
	for i := 1; i <= 2; i++ {
		if _, ok := <-errors; !ok {
			t.Errorf("%d. error expected, none found", i);
			break;
		}
	}
	if err, ok := <-errors; ok {
		t.Errorf("only two errors expected, found 3rd: %v", err);
	}
	// the inaccessible subtrees were marked manually
	checkMarks(t);

	// cleanup
	os.Chmod(Join(tree.name, tree.entries[1].name), 0770);
	os.Chmod(Join(tree.name, tree.entries[3].name), 0770);
	if err := os.RemoveAll(tree.name); err != nil {
		t.Errorf("removeTree: %v", err);
	}
}
