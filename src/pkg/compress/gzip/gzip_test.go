// Copyright 2010 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

package gzip

import (
	"bufio"
	"bytes"
	"io"
	"io/ioutil"
	"testing"
	"time"
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
			compressor.Comment = "Äußerung"
			//compressor.Comment = "comment"
			compressor.Extra = []byte("extra")
			compressor.ModTime = time.Unix(1e8, 0)
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
			if decompressor.Comment != "Äußerung" {
				t.Fatalf("comment is %q, want %q", decompressor.Comment, "Äußerung")
			}
			if string(decompressor.Extra) != "extra" {
				t.Fatalf("extra is %q, want %q", decompressor.Extra, "extra")
			}
			if decompressor.ModTime.Unix() != 1e8 {
				t.Fatalf("mtime is %d, want %d", decompressor.ModTime.Unix(), uint32(1e8))
			}
			if decompressor.Name != "name" {
				t.Fatalf("name is %q, want %q", decompressor.Name, "name")
			}
		})
}

func TestLatin1(t *testing.T) {
	latin1 := []byte{0xc4, 'u', 0xdf, 'e', 'r', 'u', 'n', 'g', 0}
	utf8 := "Äußerung"
	z := Decompressor{r: bufio.NewReader(bytes.NewBuffer(latin1))}
	s, err := z.readString()
	if err != nil {
		t.Fatalf("%v", err)
	}
	if s != utf8 {
		t.Fatalf("string is %q, want %q", s, utf8)
	}

	buf := bytes.NewBuffer(make([]byte, 0, len(latin1)))
	c := Compressor{w: buf}
	if err = c.writeString(utf8); err != nil {
		t.Fatalf("%v", err)
	}
	s = buf.String()
	if s != string(latin1) {
		t.Fatalf("string is %v, want %v", s, latin1)
	}
	//if s, err = buf.ReadString(0); err != nil {
	//t.Fatalf("%v", err)
	//}
}
