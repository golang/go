// Copyright 2011 The Go Authors.  All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

// This benchmark tests gzip and gunzip performance.

package go1

import (
	"bytes"
	gz "compress/gzip"
	"io"
	"io/ioutil"
	"testing"
)

var (
	jsongunz = bytes.Repeat(jsonbytes, 10)
	jsongz   []byte
)

func init() {
	var buf bytes.Buffer
	c := gz.NewWriter(&buf)
	c.Write(jsongunz)
	c.Close()
	jsongz = buf.Bytes()
}

func gzip() {
	c := gz.NewWriter(ioutil.Discard)
	if _, err := c.Write(jsongunz); err != nil {
		panic(err)
	}
	if err := c.Close(); err != nil {
		panic(err)
	}
}

func gunzip() {
	r, err := gz.NewReader(bytes.NewBuffer(jsongz))
	if err != nil {
		panic(err)
	}
	if _, err := io.Copy(ioutil.Discard, r); err != nil {
		panic(err)
	}
	r.Close()
}

func BenchmarkGzip(b *testing.B) {
	b.SetBytes(int64(len(jsongunz)))
	for i := 0; i < b.N; i++ {
		gzip()
	}
}

func BenchmarkGunzip(b *testing.B) {
	b.SetBytes(int64(len(jsongunz)))
	for i := 0; i < b.N; i++ {
		gunzip()
	}
}
