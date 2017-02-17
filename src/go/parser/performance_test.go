// Copyright 2012 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

package parser

import (
	"go/token"
	"io/ioutil"
	"testing"
)

func BenchmarkParse(b *testing.B) {
	src, err := ioutil.ReadFile("parser.go")
	if err != nil {
		b.Fatal(err)
	}
	b.ResetTimer()
	b.SetBytes(int64(len(src)))
	for i := 0; i < b.N; i++ {
		if _, err := ParseFile(token.NewFileSet(), "", src, ParseComments); err != nil {
			b.Fatalf("benchmark failed due to parse error: %s", err)
		}
	}
}
