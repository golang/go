// Copyright 2012 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

package path_test

import (
	"fmt"
	"path"
)

func ExampleBase() {
	fmt.Println(path.Base("/a/b"))
	fmt.Println(path.Base("/"))
	fmt.Println(path.Base(""))
	// Output:
	// b
	// /
	// .
}

func ExampleClean() {
	paths := []string{
		"a/c",
		"a//c",
		"a/c/.",
		"a/c/b/..",
		"/../a/c",
		"/../a/b/../././/c",
		"",
	}

	for _, p := range paths {
		fmt.Printf("Clean(%q) = %q\n", p, path.Clean(p))
	}

	// Output:
	// Clean("a/c") = "a/c"
	// Clean("a//c") = "a/c"
	// Clean("a/c/.") = "a/c"
	// Clean("a/c/b/..") = "a/c"
	// Clean("/../a/c") = "/a/c"
	// Clean("/../a/b/../././/c") = "/a/c"
	// Clean("") = "."
}

func ExampleDir() {
	fmt.Println(path.Dir("/a/b/c"))
	fmt.Println(path.Dir("a/b/c"))
	fmt.Println(path.Dir("/"))
	fmt.Println(path.Dir(""))
	// Output:
	// /a/b
	// a/b
	// /
	// .
}

func ExampleExt() {
	fmt.Println(path.Ext("/a/b/c/bar.css"))
	fmt.Println(path.Ext("/"))
	fmt.Println(path.Ext(""))
	// Output:
	// .css
	//
	//
}

func ExampleIsAbs() {
	fmt.Println(path.IsAbs("/dev/null"))
	// Output: true
}

func ExampleJoin() {
	fmt.Println(path.Join("a", "b", "c"))
	fmt.Println(path.Join("a", "b/c"))
	fmt.Println(path.Join("a/b", "c"))
	fmt.Println(path.Join("", ""))
	fmt.Println(path.Join("a", ""))
	fmt.Println(path.Join("", "a"))
	// Output:
	// a/b/c
	// a/b/c
	// a/b/c
	//
	// a
	// a
}

func ExampleSplit() {
	fmt.Println(path.Split("static/myfile.css"))
	fmt.Println(path.Split("myfile.css"))
	fmt.Println(path.Split(""))
	// Output:
	// static/ myfile.css
	//  myfile.css
	//
}
