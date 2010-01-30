// Copyright 2010 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

package gzip

import (
	"io"
	"io/ioutil"
	"strings"
	"testing"
)

// pipe creates two ends of a pipe that gzip and gunzip, and runs dfunc at the
// writer end and ifunc at the reader end.
func pipe(t *testing.T, dfunc func(*Deflater), ifunc func(*Inflater)) {
	piper, pipew := io.Pipe()
	defer piper.Close()
	go func() {
		defer pipew.Close()
		deflater, err := NewDeflater(pipew)
		if err != nil {
			t.Fatalf("%v", err)
		}
		defer deflater.Close()
		dfunc(deflater)
	}()
	inflater, err := NewInflater(piper)
	if err != nil {
		t.Fatalf("%v", err)
	}
	defer inflater.Close()
	ifunc(inflater)
}

// Tests that an empty payload still forms a valid GZIP stream.
func TestEmpty(t *testing.T) {
	pipe(t,
		func(deflater *Deflater) {},
		func(inflater *Inflater) {
			b, err := ioutil.ReadAll(inflater)
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
		func(deflater *Deflater) {
			deflater.Comment = "comment"
			deflater.Extra = strings.Bytes("extra")
			deflater.Mtime = 1e8
			deflater.Name = "name"
			_, err := deflater.Write(strings.Bytes("payload"))
			if err != nil {
				t.Fatalf("%v", err)
			}
		},
		func(inflater *Inflater) {
			b, err := ioutil.ReadAll(inflater)
			if err != nil {
				t.Fatalf("%v", err)
			}
			if string(b) != "payload" {
				t.Fatalf("payload is %q, want %q", string(b), "payload")
			}
			if inflater.Comment != "comment" {
				t.Fatalf("comment is %q, want %q", inflater.Comment, "comment")
			}
			if string(inflater.Extra) != "extra" {
				t.Fatalf("extra is %q, want %q", inflater.Extra, "extra")
			}
			if inflater.Mtime != 1e8 {
				t.Fatalf("mtime is %d, want %d", inflater.Mtime, uint32(1e8))
			}
			if inflater.Name != "name" {
				t.Fatalf("name is %q, want %q", inflater.Name, "name")
			}
		})
}
