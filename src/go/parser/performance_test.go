// Copyright 2012 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

package parser

import (
	"go/token"
	"os"
	"testing"
)

// TODO(rFindley): use a testdata file or file from another package here, to
//                 avoid a moving target.
var src = readFile("parser.go")

func readFile(filename string) []byte {
	data, err := os.ReadFile(filename)
	if err != nil {
		panic(err)
	}
	return data
}

func BenchmarkParse(b *testing.B) {
	b.SetBytes(int64(len(src)))
	for i := 0; i < b.N; i++ {
		if _, err := ParseFile(token.NewFileSet(), "", src, ParseComments); err != nil {
			b.Fatalf("benchmark failed due to parse error: %s", err)
		}
	}
}

func BenchmarkParseOnly(b *testing.B) {
	b.SetBytes(int64(len(src)))
	for i := 0; i < b.N; i++ {
		if _, err := ParseFile(token.NewFileSet(), "", src, ParseComments|SkipObjectResolution); err != nil {
			b.Fatalf("benchmark failed due to parse error: %s", err)
		}
	}
}

func BenchmarkResolve(b *testing.B) {
	b.SetBytes(int64(len(src)))
	for i := 0; i < b.N; i++ {
		b.StopTimer()
		fset := token.NewFileSet()
		file, err := ParseFile(fset, "", src, SkipObjectResolution)
		if err != nil {
			b.Fatalf("benchmark failed due to parse error: %s", err)
		}
		b.StartTimer()
		handle := fset.File(file.Package)
		resolveFile(file, handle, nil)
	}
}
