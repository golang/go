// Copyright 2013 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

package cipher_test

import (
	"bytes"
	"crypto/aes"
	"crypto/cipher"
	"crypto/des"
	"testing"
)

func TestCryptBlocks(t *testing.T) {
	buf := make([]byte, 16)
	block, _ := aes.NewCipher(buf)

	mode := cipher.NewCBCDecrypter(block, buf)
	mustPanic(t, "crypto/cipher: input not full blocks", func() { mode.CryptBlocks(buf, buf[:3]) })
	mustPanic(t, "crypto/cipher: output smaller than input", func() { mode.CryptBlocks(buf[:3], buf) })

	mode = cipher.NewCBCEncrypter(block, buf)
	mustPanic(t, "crypto/cipher: input not full blocks", func() { mode.CryptBlocks(buf, buf[:3]) })
	mustPanic(t, "crypto/cipher: output smaller than input", func() { mode.CryptBlocks(buf[:3], buf) })
}

func mustPanic(t *testing.T, msg string, f func()) {
	defer func() {
		err := recover()
		if err == nil {
			t.Errorf("function did not panic, wanted %q", msg)
		} else if err != msg {
			t.Errorf("got panic %v, wanted %q", err, msg)
		}
	}()
	f()
}

func TestEmptyPlaintext(t *testing.T) {
	var key [16]byte
	a, err := aes.NewCipher(key[:16])
	if err != nil {
		t.Fatal(err)
	}
	d, err := des.NewCipher(key[:8])
	if err != nil {
		t.Fatal(err)
	}

	s := 16
	pt := make([]byte, s)
	ct := make([]byte, s)
	for i := 0; i < 16; i++ {
		pt[i], ct[i] = byte(i), byte(i)
	}

	assertEqual := func(name string, got, want []byte) {
		if !bytes.Equal(got, want) {
			t.Fatalf("%s: got %v, want %v", name, got, want)
		}
	}

	for _, b := range []cipher.Block{a, d} {
		iv := make([]byte, b.BlockSize())
		cbce := cipher.NewCBCEncrypter(b, iv)
		cbce.CryptBlocks(ct, pt[:0])
		assertEqual("CBC encrypt", ct, pt)

		cbcd := cipher.NewCBCDecrypter(b, iv)
		cbcd.CryptBlocks(ct, pt[:0])
		assertEqual("CBC decrypt", ct, pt)

		cfbe := cipher.NewCFBEncrypter(b, iv)
		cfbe.XORKeyStream(ct, pt[:0])
		assertEqual("CFB encrypt", ct, pt)

		cfbd := cipher.NewCFBDecrypter(b, iv)
		cfbd.XORKeyStream(ct, pt[:0])
		assertEqual("CFB decrypt", ct, pt)

		ctr := cipher.NewCTR(b, iv)
		ctr.XORKeyStream(ct, pt[:0])
		assertEqual("CTR", ct, pt)

		ofb := cipher.NewOFB(b, iv)
		ofb.XORKeyStream(ct, pt[:0])
		assertEqual("OFB", ct, pt)
	}
}
