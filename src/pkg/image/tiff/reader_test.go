// Copyright 2011 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

package tiff

import (
	"io/ioutil"
	"os"
	"testing"
)

// Read makes *buffer implements io.Reader, so that we can pass one to Decode.
func (*buffer) Read([]byte) (int, os.Error) {
	panic("unimplemented")
}

// TestNoRPS tries to decode an image that has no RowsPerStrip tag.
// The tag is mandatory according to the spec but some software omits
// it in the case of a single strip.
func TestNoRPS(t *testing.T) {
	f, err := os.Open("testdata/no_rps.tiff")
	if err != nil {
		t.Fatal(err)
	}
	defer f.Close()
	_, err = Decode(f)
	if err != nil {
		t.Fatal(err)
	}
}

const filename = "testdata/video-001-uncompressed.tiff"

// BenchmarkDecode benchmarks the decoding of an image.
func BenchmarkDecode(b *testing.B) {
	b.StopTimer()
	contents, err := ioutil.ReadFile(filename)
	if err != nil {
		panic(err)
	}
	r := &buffer{buf: contents}
	b.StartTimer()
	for i := 0; i < b.N; i++ {
		_, err := Decode(r)
		if err != nil {
			panic(err)
		}
	}
}
