// Copyright 2010 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

package gzip

import (
	"io"
	"io/ioutil"
	"testing"
)

// pipe creates two ends of a pipe that gzip and gunzip, and runs dfunc at the
// writer end and cfunc at the reader end.
func pipe(t *testing.T, dfunc func(*Compressor), cfunc func(*Decompressor)) {
	piper, pipew := io.Pipe()
	defer piper.Close()
	go func() {
		defer pipew.Close()
		compressor, err := NewWriter(pipew)
		if err != nil {
			t.Fatalf("%v", err)
		}
		defer compressor.Close()
		dfunc(compressor)
	}()
	decompressor, err := NewReader(piper)
	if err != nil {
		t.Fatalf("%v", err)
	}
	defer decompressor.Close()
	cfunc(decompressor)
}

// Tests that an empty payload still forms a valid GZIP stream.
func TestEmpty(t *testing.T) {
	pipe(t,
		func(compressor *Compressor) {},
		func(decompressor *Decompressor) {
			b, err := ioutil.ReadAll(decompressor)
			if err != nil {
				t.Fatalf("%v", err)
			}
			if len(b) != 0 {
				t.Fatalf("did not read an empty slice")
			}
		})
}

// Tests that gzipping and then gunzipping is the identity function.
func TestWriter(t *testing.T) {
	pipe(t,
		func(compressor *Compressor) {
			compressor.Comment = "comment"
			compressor.Extra = []byte("extra")
			compressor.Mtime = 1e8
			compressor.Name = "name"
			_, err := compressor.Write([]byte("payload"))
			if err != nil {
				t.Fatalf("%v", err)
			}
		},
		func(decompressor *Decompressor) {
			b, err := ioutil.ReadAll(decompressor)
			if err != nil {
				t.Fatalf("%v", err)
			}
			if string(b) != "payload" {
				t.Fatalf("payload is %q, want %q", string(b), "payload")
			}
			if decompressor.Comment != "comment" {
				t.Fatalf("comment is %q, want %q", decompressor.Comment, "comment")
			}
			if string(decompressor.Extra) != "extra" {
				t.Fatalf("extra is %q, want %q", decompressor.Extra, "extra")
			}
			if decompressor.Mtime != 1e8 {
				t.Fatalf("mtime is %d, want %d", decompressor.Mtime, uint32(1e8))
			}
			if decompressor.Name != "name" {
				t.Fatalf("name is %q, want %q", decompressor.Name, "name")
			}
		})
}
