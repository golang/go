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

// Tests that gzipping and then gunzipping is the identity function.
func TestWriter(t *testing.T) {
	// Set up the Pipe to do the gzip and gunzip.
	piper, pipew := io.Pipe()
	defer piper.Close()
	go func() {
		defer pipew.Close()
		deflater, err := NewDeflater(pipew)
		if err != nil {
			t.Errorf("%v", err)
			return
		}
		defer deflater.Close()
		deflater.Comment = "comment"
		deflater.Extra = strings.Bytes("extra")
		deflater.Mtime = 1e8
		deflater.Name = "name"
		_, err = deflater.Write(strings.Bytes("payload"))
		if err != nil {
			t.Errorf("%v", err)
			return
		}
	}()
	inflater, err := NewInflater(piper)
	if err != nil {
		t.Errorf("%v", err)
		return
	}
	defer inflater.Close()

	// Read and compare to the original input.
	b, err := ioutil.ReadAll(inflater)
	if err != nil {
		t.Errorf(": %v", err)
		return
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
}
