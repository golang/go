// Copyright 2011 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

package s2k

import (
	"bytes"
	"crypto/rand"
	"crypto/sha1"
	"encoding/hex"
	"testing"
)

var saltedTests = []struct {
	in, out string
}{
	{"hello", "10295ac1"},
	{"world", "ac587a5e"},
	{"foo", "4dda8077"},
	{"bar", "bd8aac6b9ea9cae04eae6a91c6133b58b5d9a61c14f355516ed9370456"},
	{"x", "f1d3f289"},
	{"xxxxxxxxxxxxxxxxxxxxxxx", "e00d7b45"},
}

func TestSalted(t *testing.T) {
	h := sha1.New()
	salt := [4]byte{1, 2, 3, 4}

	for i, test := range saltedTests {
		expected, _ := hex.DecodeString(test.out)
		out := make([]byte, len(expected))
		Salted(out, h, []byte(test.in), salt[:])
		if !bytes.Equal(expected, out) {
			t.Errorf("#%d, got: %x want: %x", i, out, expected)
		}
	}
}

var iteratedTests = []struct {
	in, out string
}{
	{"hello", "83126105"},
	{"world", "6fa317f9"},
	{"foo", "8fbc35b9"},
	{"bar", "2af5a99b54f093789fd657f19bd245af7604d0f6ae06f66602a46a08ae"},
	{"x", "5a684dfe"},
	{"xxxxxxxxxxxxxxxxxxxxxxx", "18955174"},
}

func TestIterated(t *testing.T) {
	h := sha1.New()
	salt := [4]byte{4, 3, 2, 1}

	for i, test := range iteratedTests {
		expected, _ := hex.DecodeString(test.out)
		out := make([]byte, len(expected))
		Iterated(out, h, []byte(test.in), salt[:], 31)
		if !bytes.Equal(expected, out) {
			t.Errorf("#%d, got: %x want: %x", i, out, expected)
		}
	}
}

var parseTests = []struct {
	spec, in, out string
}{
	/* Simple with SHA1 */
	{"0102", "hello", "aaf4c61d"},
	/* Salted with SHA1 */
	{"02020102030405060708", "hello", "f4f7d67e"},
	/* Iterated with SHA1 */
	{"03020102030405060708f1", "hello", "f2a57b7c"},
}

func TestParse(t *testing.T) {
	for i, test := range parseTests {
		spec, _ := hex.DecodeString(test.spec)
		buf := bytes.NewBuffer(spec)
		f, err := Parse(buf)
		if err != nil {
			t.Errorf("%d: Parse returned error: %s", i, err)
			continue
		}

		expected, _ := hex.DecodeString(test.out)
		out := make([]byte, len(expected))
		f(out, []byte(test.in))
		if !bytes.Equal(out, expected) {
			t.Errorf("%d: output got: %x want: %x", i, out, expected)
		}
		if testing.Short() {
			break
		}
	}
}

func TestSerialize(t *testing.T) {
	buf := bytes.NewBuffer(nil)
	key := make([]byte, 16)
	passphrase := []byte("testing")
	err := Serialize(buf, key, rand.Reader, passphrase)
	if err != nil {
		t.Errorf("failed to serialize: %s", err)
		return
	}

	f, err := Parse(buf)
	if err != nil {
		t.Errorf("failed to reparse: %s", err)
		return
	}
	key2 := make([]byte, len(key))
	f(key2, passphrase)
	if !bytes.Equal(key2, key) {
		t.Errorf("keys don't match: %x (serialied) vs %x (parsed)", key, key2)
	}
}
