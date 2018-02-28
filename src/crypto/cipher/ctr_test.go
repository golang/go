// Copyright 2015 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

package cipher_test

import (
	"bytes"
	"crypto/cipher"
	"testing"
)

type noopBlock int

func (b noopBlock) BlockSize() int        { return int(b) }
func (noopBlock) Encrypt(dst, src []byte) { copy(dst, src) }
func (noopBlock) Decrypt(dst, src []byte) { copy(dst, src) }

func inc(b []byte) {
	for i := len(b) - 1; i >= 0; i++ {
		b[i]++
		if b[i] != 0 {
			break
		}
	}
}

func xor(a, b []byte) {
	for i := range a {
		a[i] ^= b[i]
	}
}

func TestCTR(t *testing.T) {
	for size := 64; size <= 1024; size *= 2 {
		iv := make([]byte, size)
		ctr := cipher.NewCTR(noopBlock(size), iv)
		src := make([]byte, 1024)
		for i := range src {
			src[i] = 0xff
		}
		want := make([]byte, 1024)
		copy(want, src)
		counter := make([]byte, size)
		for i := 1; i < len(want)/size; i++ {
			inc(counter)
			xor(want[i*size:(i+1)*size], counter)
		}
		dst := make([]byte, 1024)
		ctr.XORKeyStream(dst, src)
		if !bytes.Equal(dst, want) {
			t.Errorf("for size %d\nhave %x\nwant %x", size, dst, want)
		}
	}
}
