// Copyright 2024 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

package hkdf

import (
	"crypto/internal/fips140"
	"crypto/internal/fips140/hmac"
)

func Extract[H fips140.Hash](h func() H, secret, salt []byte) []byte {
	if len(secret) < 112/8 {
		fips140.RecordNonApproved()
	}
	if salt == nil {
		salt = make([]byte, h().Size())
	}
	extractor := hmac.New(h, salt)
	hmac.MarkAsUsedInHKDF(extractor)
	extractor.Write(secret)

	return extractor.Sum(nil)
}

func Expand[H fips140.Hash](h func() H, pseudorandomKey []byte, info string, keyLen int) []byte {
	out := make([]byte, 0, keyLen)
	expander := hmac.New(h, pseudorandomKey)
	hmac.MarkAsUsedInHKDF(expander)
	var counter uint8
	var buf []byte

	for len(out) < keyLen {
		counter++
		if counter == 0 {
			panic("hkdf: counter overflow")
		}
		if counter > 1 {
			expander.Reset()
		}
		expander.Write(buf)
		expander.Write([]byte(info))
		expander.Write([]byte{counter})
		buf = expander.Sum(buf[:0])
		remain := keyLen - len(out)
		remain = min(remain, len(buf))
		out = append(out, buf[:remain]...)
	}

	return out
}

func Key[H fips140.Hash](h func() H, secret, salt []byte, info string, keyLen int) []byte {
	prk := Extract(h, secret, salt)
	return Expand(h, prk, info, keyLen)
}
