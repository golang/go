// Copyright 2012 The Go Authors.  All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

package main

func init() {
	addTestCases(newWriterTests, newWriter)
}

var newWriterTests = []testCase{
	{
		Name: "newWriter.0",
		In: `package main

import (
	"bufio"
	"compress/gzip"
	"compress/zlib"
	"io"

	"foo"
)

func f() *gzip.Compressor {
	var (
		_ gzip.Compressor
		_ *gzip.Decompressor
		_ struct {
			W *gzip.Compressor
			R gzip.Decompressor
		}
	)

	var w io.Writer
	br := bufio.NewReader(nil)
	br, _ = bufio.NewReaderSize(nil, 256)
	bw, err := bufio.NewWriterSize(w, 256) // Unfixable, as it declares an err variable.
	bw, _ = bufio.NewWriterSize(w, 256)
	fw, _ := foo.NewWriter(w)
	gw, _ := gzip.NewWriter(w)
	gw, _ = gzip.NewWriter(w)
	zw, _ := zlib.NewWriter(w)
	_ = zlib.NewWriterDict(zw, 0, nil)
	return gw
}
`,
		Out: `package main

import (
	"bufio"
	"compress/gzip"
	"compress/zlib"
	"io"

	"foo"
)

func f() *gzip.Writer {
	var (
		_ gzip.Writer
		_ *gzip.Reader
		_ struct {
			W *gzip.Writer
			R gzip.Reader
		}
	)

	var w io.Writer
	br := bufio.NewReader(nil)
	br = bufio.NewReaderSize(nil, 256)
	bw, err := bufio.NewWriterSize(w, 256) // Unfixable, as it declares an err variable.
	bw = bufio.NewWriterSize(w, 256)
	fw, _ := foo.NewWriter(w)
	gw := gzip.NewWriter(w)
	gw = gzip.NewWriter(w)
	zw := zlib.NewWriter(w)
	_ = zlib.NewWriterLevelDict(zw, 0, nil)
	return gw
}
`,
	},
}
