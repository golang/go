// Copyright 2012 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

package go1

// go parser benchmark based on go/parser/performance_test.go

import (
	"compress/bzip2"
	"encoding/base64"
	"go/parser"
	"go/token"
	"io"
	"strings"
	"testing"
)

var (
	parserbytes = makeParserBytes()
)

func makeParserBytes() []byte {
	var r io.Reader
	r = strings.NewReader(parserbz2_base64)
	r = base64.NewDecoder(base64.StdEncoding, r)
	r = bzip2.NewReader(r)
	b, err := io.ReadAll(r)
	if err != nil {
		panic(err)
	}
	return b
}

func BenchmarkGoParse(b *testing.B) {
	b.SetBytes(int64(len(parserbytes)))
	for i := 0; i < b.N; i++ {
		if _, err := parser.ParseFile(token.NewFileSet(), "", parserbytes, parser.ParseComments); err != nil {
			b.Fatalf("benchmark failed due to parse error: %s", err)
		}
	}
}
