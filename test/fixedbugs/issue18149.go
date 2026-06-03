// run

// Copyright 2016 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

// Verify that //line directives with filenames
// containing ':' (Windows) are correctly parsed.
// (For a related issue, see test/fixedbugs/bug305.go)

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
//line /foo/bar.go:123
	check("/foo/bar.go", 123)
//line c:/foo/bar.go:987
	check("c:/foo/bar.go", 987)
}
