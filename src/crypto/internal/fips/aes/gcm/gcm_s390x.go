// Copyright 2016 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

//go:build !purego

package gcm

import (
	"crypto/internal/fips/aes"
	"crypto/internal/fips/subtle"
	"crypto/internal/fipsdeps/byteorder"
	"crypto/internal/fipsdeps/cpu"
	"crypto/internal/impl"
)

// This file contains two implementations of AES-GCM. The first implementation
// (useGHASH) uses the KMCTR instruction to encrypt using AES in counter mode
// and the KIMD instruction for GHASH. The second implementation (useGCM) uses
// the newer KMA instruction which performs both operations (but still requires
// KIMD to hash large nonces).

// Keep in sync with crypto/tls.hasAESGCMHardwareSupport.
var useGHASH = cpu.S390XHasAES && cpu.S390XHasAESCTR && cpu.S390XHasGHASH
var useGCM = useGHASH && cpu.S390XHasAESGCM

func init() {
	impl.Register("gcm", "CPACF/KIMD", &useGHASH)
	impl.Register("gcm", "CPACF/KMA", &useGCM)
}

func checkGenericIsExpected() {
	if useGHASH || useGCM {
		panic("gcm: internal error: using generic implementation despite hardware support")
	}
}

// gcmLengths writes len0 || len1 as big-endian values to a 16-byte array.
func gcmLengths(len0, len1 uint64) [16]byte {
	v := [16]byte{}
	byteorder.BEPutUint64(v[0:], len0)
	byteorder.BEPutUint64(v[8:], len1)
	return v
}

// gcmHashKey represents the 16-byte hash key required by the GHASH algorithm.
type gcmHashKey [16]byte

type gcmPlatformData struct {
	hashKey gcmHashKey
}

func initGCM(g *GCM) {
	if !useGCM && !useGHASH {
		return
	}
	// Note that hashKey is also used in the KMA codepath to hash large nonces.
	g.cipher.Encrypt(g.hashKey[:], g.hashKey[:])
}

// ghashAsm uses the GHASH algorithm to hash data with the given key. The initial
// hash value is given by hash which will be updated with the new hash value.
// The length of data must be a multiple of 16-bytes.
//
//go:noescape
func ghashAsm(key *gcmHashKey, hash *[16]byte, data []byte)

// paddedGHASH pads data with zeroes until its length is a multiple of
// 16-bytes. It then calculates a new value for hash using the GHASH algorithm.
func paddedGHASH(hashKey *gcmHashKey, hash *[16]byte, data []byte) {
	siz := len(data) &^ 0xf // align size to 16-bytes
	if siz > 0 {
		ghashAsm(hashKey, hash, data[:siz])
		data = data[siz:]
	}
	if len(data) > 0 {
		var s [16]byte
		copy(s[:], data)
		ghashAsm(hashKey, hash, s[:])
	}
}

// cryptBlocksGCM encrypts src using AES in counter mode using the given
// function code and key. The rightmost 32-bits of the counter are incremented
// between each block as required by the GCM spec. The initial counter value
// is given by cnt, which is updated with the value of the next counter value
// to use.
//
// The lengths of both dst and buf must be greater than or equal to the length
// of src. buf may be partially or completely overwritten during the execution
// of the function.
//
//go:noescape
func cryptBlocksGCM(fn int, key, dst, src, buf []byte, cnt *[gcmBlockSize]byte)

// counterCrypt encrypts src using AES in counter mode and places the result
// into dst. cnt is the initial count value and will be updated with the next
// count value. The length of dst must be greater than or equal to the length
// of src.
func counterCrypt(g *GCM, dst, src []byte, cnt *[gcmBlockSize]byte) {
	// Copying src into a buffer improves performance on some models when
	// src and dst point to the same underlying array. We also need a
	// buffer for counter values.
	var ctrbuf, srcbuf [2048]byte
	for len(src) >= 16 {
		siz := len(src)
		if len(src) > len(ctrbuf) {
			siz = len(ctrbuf)
		}
		siz &^= 0xf // align siz to 16-bytes
		copy(srcbuf[:], src[:siz])
		cryptBlocksGCM(aes.BlockFunction(&g.cipher), aes.BlockKey(&g.cipher), dst[:siz], srcbuf[:siz], ctrbuf[:], cnt)
		src = src[siz:]
		dst = dst[siz:]
	}
	if len(src) > 0 {
		var x [16]byte
		g.cipher.Encrypt(x[:], cnt[:])
		for i := range src {
			dst[i] = src[i] ^ x[i]
		}
		gcmInc32(cnt)
	}
}

// deriveCounter computes the initial GCM counter state from the given nonce.
// See NIST SP 800-38D, section 7.1 and deriveCounterGeneric in gcm_generic.go.
func deriveCounter(H *gcmHashKey, counter *[gcmBlockSize]byte, nonce []byte) {
	if len(nonce) == gcmStandardNonceSize {
		copy(counter[:], nonce)
		counter[gcmBlockSize-1] = 1
	} else {
		var hash [16]byte
		paddedGHASH(H, &hash, nonce)
		lens := gcmLengths(0, uint64(len(nonce))*8)
		paddedGHASH(H, &hash, lens[:])
		copy(counter[:], hash[:])
	}
}

// gcmAuth calculates GHASH(additionalData, ciphertext), masks the result
// with tagMask and writes the result to out.
func gcmAuth(out []byte, H *gcmHashKey, tagMask *[gcmBlockSize]byte, ciphertext, additionalData []byte) {
	var hash [16]byte
	paddedGHASH(H, &hash, additionalData)
	paddedGHASH(H, &hash, ciphertext)
	lens := gcmLengths(uint64(len(additionalData))*8, uint64(len(ciphertext))*8)
	paddedGHASH(H, &hash, lens[:])

	copy(out, hash[:])
	for i := range out {
		out[i] ^= tagMask[i]
	}
}

func seal(out []byte, g *GCM, nonce, plaintext, data []byte) {
	switch {
	case useGCM:
		sealKMA(out, g, nonce, plaintext, data)
	case useGHASH:
		sealAsm(out, g, nonce, plaintext, data)
	default:
		sealGeneric(out, g, nonce, plaintext, data)
	}
}

func sealAsm(out []byte, g *GCM, nonce, plaintext, additionalData []byte) {
	var counter, tagMask [gcmBlockSize]byte
	deriveCounter(&g.hashKey, &counter, nonce)
	counterCrypt(g, tagMask[:], tagMask[:], &counter)

	counterCrypt(g, out, plaintext, &counter)

	var tag [gcmTagSize]byte
	gcmAuth(tag[:], &g.hashKey, &tagMask, out[:len(plaintext)], additionalData)
	copy(out[len(plaintext):], tag[:])
}

func open(out []byte, g *GCM, nonce, ciphertext, data []byte) error {
	switch {
	case useGCM:
		return openKMA(out, g, nonce, ciphertext, data)
	case useGHASH:
		return openAsm(out, g, nonce, ciphertext, data)
	default:
		return openGeneric(out, g, nonce, ciphertext, data)
	}
}

func openAsm(out []byte, g *GCM, nonce, ciphertext, additionalData []byte) error {
	var counter, tagMask [gcmBlockSize]byte
	deriveCounter(&g.hashKey, &counter, nonce)
	counterCrypt(g, tagMask[:], tagMask[:], &counter)

	tag := ciphertext[len(ciphertext)-g.tagSize:]
	ciphertext = ciphertext[:len(ciphertext)-g.tagSize]

	var expectedTag [gcmTagSize]byte
	gcmAuth(expectedTag[:], &g.hashKey, &tagMask, ciphertext, additionalData)
	if subtle.ConstantTimeCompare(expectedTag[:g.tagSize], tag) != 1 {
		return errOpen
	}

	counterCrypt(g, out, ciphertext, &counter)

	return nil
}

// flags for the KMA instruction
const (
	kmaHS      = 1 << 10 // hash subkey supplied
	kmaLAAD    = 1 << 9  // last series of additional authenticated data
	kmaLPC     = 1 << 8  // last series of plaintext or ciphertext blocks
	kmaDecrypt = 1 << 7  // decrypt
)

// kmaGCM executes the encryption or decryption operation given by fn. The tag
// will be calculated and written to tag. cnt should contain the current
// counter state and will be overwritten with the updated counter state.
// TODO(mundaym): could pass in hash subkey
//
//go:noescape
func kmaGCM(fn int, key, dst, src, aad []byte, tag *[16]byte, cnt *[gcmBlockSize]byte)

func sealKMA(out []byte, g *GCM, nonce, plaintext, data []byte) {
	var counter [gcmBlockSize]byte
	deriveCounter(&g.hashKey, &counter, nonce)
	fc := aes.BlockFunction(&g.cipher) | kmaLAAD | kmaLPC

	var tag [gcmTagSize]byte
	kmaGCM(fc, aes.BlockKey(&g.cipher), out[:len(plaintext)], plaintext, data, &tag, &counter)
	copy(out[len(plaintext):], tag[:])
}

func openKMA(out []byte, g *GCM, nonce, ciphertext, data []byte) error {
	tag := ciphertext[len(ciphertext)-g.tagSize:]
	ciphertext = ciphertext[:len(ciphertext)-g.tagSize]

	var counter [gcmBlockSize]byte
	deriveCounter(&g.hashKey, &counter, nonce)
	fc := aes.BlockFunction(&g.cipher) | kmaLAAD | kmaLPC | kmaDecrypt

	var expectedTag [gcmTagSize]byte
	kmaGCM(fc, aes.BlockKey(&g.cipher), out[:len(ciphertext)], ciphertext, data, &expectedTag, &counter)

	if subtle.ConstantTimeCompare(expectedTag[:g.tagSize], tag) != 1 {
		return errOpen
	}

	return nil
}
