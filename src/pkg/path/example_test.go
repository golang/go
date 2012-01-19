// Copyright 2012 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

package path_test

import (
	"fmt"
	"path"
)

// b
func ExampleBase() {
	fmt.Println(path.Base("/a/b"))
}

// Clean("a/c") = "a/c"
// Clean("a//c") = "a/c"
// Clean("a/c/.") = "a/c"
// Clean("a/c/b/..") = "a/c"
// Clean("/../a/c") = "/a/c"
// Clean("/../a/b/../././/c") = "/a/c"
func ExampleClean() {
	paths := []string{
		"a/c",
		"a//c",
		"a/c/.",
		"a/c/b/..",
		"/../a/c",
		"/../a/b/../././/c",
	}

	for _, p := range paths {
		fmt.Printf("Clean(%q) = %q\n", p, path.Clean(p))
	}
}

// /a/b
func ExampleDir() {
	fmt.Println(path.Dir("/a/b/c"))
}

// .css
func ExampleExt() {
	fmt.Println(path.Ext("/a/b/c/bar.css"))
}

// true
func ExampleIsAbs() {
	fmt.Println(path.IsAbs("/dev/null"))
}

// a/b/c
func ExampleJoin() {
	fmt.Println(path.Join("a", "b", "c"))
}

// static/ myfile.css
func ExampleSplit() {
	fmt.Println(path.Split("static/myfile.css"))
}
