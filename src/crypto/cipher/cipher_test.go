// Copyright 2013 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

package cipher_test

import (
	"crypto/aes"
	"crypto/cipher"
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
