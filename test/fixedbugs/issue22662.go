// run

// Copyright 2018 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

// Verify effect of various line directives.
// TODO: check columns

package main

import (
	"fmt"
	"runtime"
	"strings"
)

// Since go.dev/issue/70478, the compiler resolves a relative filename in a
// line directive against the directory of the source file, so the expected
// path may be a suffix of the actual filename rather than equal to it.
// Accept either an exact match or the file as a path-component suffix.
// Compiler-emitted paths are always slash-normalized (cmd/internal/objabi.AbsFile).
func check(file string, line int) {
	_, f, l, ok := runtime.Caller(1)
	if !ok {
		panic("runtime.Caller(1) failed")
	}
	// Prepend exactly one "/" even if file already starts with one, so that
	// e.g. file="/foo/bar.go" looks for "/foo/bar.go" as the suffix, not "//".
	want := "/" + strings.TrimPrefix(file, "/")
	if (f != file && !strings.HasSuffix(f, want)) || l != line {
		panic(fmt.Sprintf("got %s:%d; want %s:%d (or suffix %s)", f, l, file, line, want))
	}
}

func main() {
//-style line directives
//line :1
	check("??", 1) // no file specified
//line foo.go:1
	check("foo.go", 1)
//line bar.go:10:20
	check("bar.go", 10)
//line :11:22
	check("bar.go", 11) // no file, but column specified => keep old filename

/*-style line directives */
/*line :1*/ check("??", 1) // no file specified
/*line foo.go:1*/ check("foo.go", 1)
/*line bar.go:10:20*/ check("bar.go", 10)
/*line :11:22*/ check("bar.go", 11) // no file, but column specified => keep old filename

	/*line :10*/ check("??", 10); /*line foo.go:20*/ check("foo.go", 20); /*line :30:1*/ check("foo.go", 30)
	check("foo.go", 31)
}
