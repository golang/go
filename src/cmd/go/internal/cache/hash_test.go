// Copyright 2017 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

package cache

import (
	"fmt"
	"io/ioutil"
	"os"
	"testing"
)

func TestHash(t *testing.T) {
	oldSalt := hashSalt
	hashSalt = nil
	defer func() {
		hashSalt = oldSalt
	}()

	h := NewHash("alice")
	h.Write([]byte("hello world"))
	sum := fmt.Sprintf("%x", h.Sum())
	want := "b94d27b9934d3e08a52e52d7da7dabfac484efe37a5380ee9088f7ace2efcde9"
	if sum != want {
		t.Errorf("hash(hello world) = %v, want %v", sum, want)
	}
}

func TestHashFile(t *testing.T) {
	f, err := ioutil.TempFile("", "cmd-go-test-")
	if err != nil {
		t.Fatal(err)
	}
	name := f.Name()
	fmt.Fprintf(f, "hello world")
	defer os.Remove(name)
	if err := f.Close(); err != nil {
		t.Fatal(err)
	}

	var h ActionID // make sure hash result is assignable to ActionID
	h, err = FileHash(name)
	if err != nil {
		t.Fatal(err)
	}
	sum := fmt.Sprintf("%x", h)
	want := "b94d27b9934d3e08a52e52d7da7dabfac484efe37a5380ee9088f7ace2efcde9"
	if sum != want {
		t.Errorf("hash(hello world) = %v, want %v", sum, want)
	}
}
