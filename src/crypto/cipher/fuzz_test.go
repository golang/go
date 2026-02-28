// Copyright 2021 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

//go:build ppc64le

package cipher_test

import (
	"bytes"
	"crypto/aes"
	"crypto/cipher"
	"crypto/rand"
	"testing"
	"time"
)

var cbcAESFuzzTests = []struct {
	name string
	key  []byte
}{
	{
		"CBC-AES128",
		commonKey128,
	},
	{
		"CBC-AES192",
		commonKey192,
	},
	{
		"CBC-AES256",
		commonKey256,
	},
}

var timeout *time.Timer

const datalen = 1024

func TestFuzz(t *testing.T) {

	for _, ft := range cbcAESFuzzTests {
		c, _ := aes.NewCipher(ft.key)

		cbcAsm := cipher.NewCBCEncrypter(c, commonIV)
		cbcGeneric := cipher.NewCBCGenericEncrypter(c, commonIV)

		if testing.Short() {
			timeout = time.NewTimer(10 * time.Millisecond)
		} else {
			timeout = time.NewTimer(2 * time.Second)
		}

		indata := make([]byte, datalen)
		outgeneric := make([]byte, datalen)
		outdata := make([]byte, datalen)

	fuzzencrypt:
		for {
			select {
			case <-timeout.C:
				break fuzzencrypt
			default:
			}

			rand.Read(indata[:])

			cbcGeneric.CryptBlocks(indata, outgeneric)
			cbcAsm.CryptBlocks(indata, outdata)

			if !bytes.Equal(outdata, outgeneric) {
				t.Fatalf("AES-CBC encryption does not match reference result: %x and %x, please report this error to security@golang.org", outdata, outgeneric)
			}
		}

		cbcAsm = cipher.NewCBCDecrypter(c, commonIV)
		cbcGeneric = cipher.NewCBCGenericDecrypter(c, commonIV)

		if testing.Short() {
			timeout = time.NewTimer(10 * time.Millisecond)
		} else {
			timeout = time.NewTimer(2 * time.Second)
		}

	fuzzdecrypt:
		for {
			select {
			case <-timeout.C:
				break fuzzdecrypt
			default:
			}

			rand.Read(indata[:])

			cbcGeneric.CryptBlocks(indata, outgeneric)
			cbcAsm.CryptBlocks(indata, outdata)

			if !bytes.Equal(outdata, outgeneric) {
				t.Fatalf("AES-CBC decryption does not match reference result: %x and %x, please report this error to security@golang.org", outdata, outgeneric)
			}
		}
	}
}
