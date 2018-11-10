// Copyright 2016 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

package chacha20poly1305

import (
	"bytes"
	cr "crypto/rand"
	"encoding/hex"
	mr "math/rand"
	"testing"
)

func TestVectors(t *testing.T) {
	for i, test := range chacha20Poly1305Tests {
		key, _ := hex.DecodeString(test.key)
		nonce, _ := hex.DecodeString(test.nonce)
		ad, _ := hex.DecodeString(test.aad)
		plaintext, _ := hex.DecodeString(test.plaintext)

		aead, err := New(key)
		if err != nil {
			t.Fatal(err)
		}

		ct := aead.Seal(nil, nonce, plaintext, ad)
		if ctHex := hex.EncodeToString(ct); ctHex != test.out {
			t.Errorf("#%d: got %s, want %s", i, ctHex, test.out)
			continue
		}

		plaintext2, err := aead.Open(nil, nonce, ct, ad)
		if err != nil {
			t.Errorf("#%d: Open failed", i)
			continue
		}

		if !bytes.Equal(plaintext, plaintext2) {
			t.Errorf("#%d: plaintext's don't match: got %x vs %x", i, plaintext2, plaintext)
			continue
		}

		if len(ad) > 0 {
			alterAdIdx := mr.Intn(len(ad))
			ad[alterAdIdx] ^= 0x80
			if _, err := aead.Open(nil, nonce, ct, ad); err == nil {
				t.Errorf("#%d: Open was successful after altering additional data", i)
			}
			ad[alterAdIdx] ^= 0x80
		}

		alterNonceIdx := mr.Intn(aead.NonceSize())
		nonce[alterNonceIdx] ^= 0x80
		if _, err := aead.Open(nil, nonce, ct, ad); err == nil {
			t.Errorf("#%d: Open was successful after altering nonce", i)
		}
		nonce[alterNonceIdx] ^= 0x80

		alterCtIdx := mr.Intn(len(ct))
		ct[alterCtIdx] ^= 0x80
		if _, err := aead.Open(nil, nonce, ct, ad); err == nil {
			t.Errorf("#%d: Open was successful after altering ciphertext", i)
		}
		ct[alterCtIdx] ^= 0x80
	}
}

func TestRandom(t *testing.T) {
	// Some random tests to verify Open(Seal) == Plaintext
	for i := 0; i < 256; i++ {
		var nonce [12]byte
		var key [32]byte

		al := mr.Intn(128)
		pl := mr.Intn(16384)
		ad := make([]byte, al)
		plaintext := make([]byte, pl)
		cr.Read(key[:])
		cr.Read(nonce[:])
		cr.Read(ad)
		cr.Read(plaintext)

		aead, err := New(key[:])
		if err != nil {
			t.Fatal(err)
		}

		ct := aead.Seal(nil, nonce[:], plaintext, ad)

		plaintext2, err := aead.Open(nil, nonce[:], ct, ad)
		if err != nil {
			t.Errorf("Random #%d: Open failed", i)
			continue
		}

		if !bytes.Equal(plaintext, plaintext2) {
			t.Errorf("Random #%d: plaintext's don't match: got %x vs %x", i, plaintext2, plaintext)
			continue
		}

		if len(ad) > 0 {
			alterAdIdx := mr.Intn(len(ad))
			ad[alterAdIdx] ^= 0x80
			if _, err := aead.Open(nil, nonce[:], ct, ad); err == nil {
				t.Errorf("Random #%d: Open was successful after altering additional data", i)
			}
			ad[alterAdIdx] ^= 0x80
		}

		alterNonceIdx := mr.Intn(aead.NonceSize())
		nonce[alterNonceIdx] ^= 0x80
		if _, err := aead.Open(nil, nonce[:], ct, ad); err == nil {
			t.Errorf("Random #%d: Open was successful after altering nonce", i)
		}
		nonce[alterNonceIdx] ^= 0x80

		alterCtIdx := mr.Intn(len(ct))
		ct[alterCtIdx] ^= 0x80
		if _, err := aead.Open(nil, nonce[:], ct, ad); err == nil {
			t.Errorf("Random #%d: Open was successful after altering ciphertext", i)
		}
		ct[alterCtIdx] ^= 0x80
	}
}

func benchamarkChaCha20Poly1305Seal(b *testing.B, buf []byte) {
	b.SetBytes(int64(len(buf)))

	var key [32]byte
	var nonce [12]byte
	var ad [13]byte
	var out []byte

	aead, _ := New(key[:])
	b.ResetTimer()
	for i := 0; i < b.N; i++ {
		out = aead.Seal(out[:0], nonce[:], buf[:], ad[:])
	}
}

func benchamarkChaCha20Poly1305Open(b *testing.B, buf []byte) {
	b.SetBytes(int64(len(buf)))

	var key [32]byte
	var nonce [12]byte
	var ad [13]byte
	var ct []byte
	var out []byte

	aead, _ := New(key[:])
	ct = aead.Seal(ct[:0], nonce[:], buf[:], ad[:])

	b.ResetTimer()
	for i := 0; i < b.N; i++ {
		out, _ = aead.Open(out[:0], nonce[:], ct[:], ad[:])
	}
}

func BenchmarkChacha20Poly1305Open_64(b *testing.B) {
	benchamarkChaCha20Poly1305Open(b, make([]byte, 64))
}

func BenchmarkChacha20Poly1305Seal_64(b *testing.B) {
	benchamarkChaCha20Poly1305Seal(b, make([]byte, 64))
}

func BenchmarkChacha20Poly1305Open_1350(b *testing.B) {
	benchamarkChaCha20Poly1305Open(b, make([]byte, 1350))
}

func BenchmarkChacha20Poly1305Seal_1350(b *testing.B) {
	benchamarkChaCha20Poly1305Seal(b, make([]byte, 1350))
}

func BenchmarkChacha20Poly1305Open_8K(b *testing.B) {
	benchamarkChaCha20Poly1305Open(b, make([]byte, 8*1024))
}

func BenchmarkChacha20Poly1305Seal_8K(b *testing.B) {
	benchamarkChaCha20Poly1305Seal(b, make([]byte, 8*1024))
}
