// Copyright 2015 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

//go:build (amd64 || arm64) && !purego

package gcm

import (
	"crypto/internal/fips140/aes"
	"crypto/internal/fips140/subtle"
	"crypto/internal/fips140deps/cpu"
	"crypto/internal/impl"
)

// The following functions are defined in gcm_*.s.

//go:noescape
func gcmAesInit(productTable *[256]byte, ks []uint32)

//go:noescape
func gcmAesData(productTable *[256]byte, data []byte, T *[16]byte)

//go:noescape
func gcmAesEnc(productTable *[256]byte, dst, src []byte, ctr, T *[16]byte, ks []uint32)

//go:noescape
func gcmAesDec(productTable *[256]byte, dst, src []byte, ctr, T *[16]byte, ks []uint32)

//go:noescape
func gcmAesFinish(productTable *[256]byte, tagMask, T *[16]byte, pLen, dLen uint64)

// Keep in sync with crypto/tls.hasAESGCMHardwareSupport.
var supportsAESGCM = cpu.X86HasAES && cpu.X86HasPCLMULQDQ && cpu.X86HasSSE41 && cpu.X86HasSSSE3 ||
	cpu.ARM64HasAES && cpu.ARM64HasPMULL

func init() {
	if cpu.AMD64 {
		impl.Register("gcm", "AES-NI", &supportsAESGCM)
	}
	if cpu.ARM64 {
		impl.Register("gcm", "Armv8.0", &supportsAESGCM)
	}
}

// checkGenericIsExpected is called by the variable-time implementation to make
// sure it is not used when hardware support is available. It shouldn't happen,
// but this way it's more evidently correct.
func checkGenericIsExpected() {
	if supportsAESGCM {
		panic("gcm: internal error: using generic implementation despite hardware support")
	}
}

type gcmPlatformData struct {
	productTable [256]byte
}

func initGCM(g *GCM) {
	if !supportsAESGCM {
		return
	}
	gcmAesInit(&g.productTable, aes.EncryptionKeySchedule(&g.cipher))
}

func seal(out []byte, g *GCM, nonce, plaintext, data []byte) {
	if !supportsAESGCM {
		sealGeneric(out, g, nonce, plaintext, data)
		return
	}

	var counter, tagMask [gcmBlockSize]byte

	if len(nonce) == gcmStandardNonceSize {
		// Init counter to nonce||1
		copy(counter[:], nonce)
		counter[gcmBlockSize-1] = 1
	} else {
		// Otherwise counter = GHASH(nonce)
		gcmAesData(&g.productTable, nonce, &counter)
		gcmAesFinish(&g.productTable, &tagMask, &counter, uint64(len(nonce)), uint64(0))
	}

	aes.EncryptBlockInternal(&g.cipher, tagMask[:], counter[:])

	var tagOut [gcmTagSize]byte
	gcmAesData(&g.productTable, data, &tagOut)

	if len(plaintext) > 0 {
		gcmAesEnc(&g.productTable, out, plaintext, &counter, &tagOut, aes.EncryptionKeySchedule(&g.cipher))
	}
	gcmAesFinish(&g.productTable, &tagMask, &tagOut, uint64(len(plaintext)), uint64(len(data)))
	copy(out[len(plaintext):], tagOut[:])
}

func open(out []byte, g *GCM, nonce, ciphertext, data []byte) error {
	if !supportsAESGCM {
		return openGeneric(out, g, nonce, ciphertext, data)
	}

	tag := ciphertext[len(ciphertext)-g.tagSize:]
	ciphertext = ciphertext[:len(ciphertext)-g.tagSize]

	// See GCM spec, section 7.1.
	var counter, tagMask [gcmBlockSize]byte

	if len(nonce) == gcmStandardNonceSize {
		// Init counter to nonce||1
		copy(counter[:], nonce)
		counter[gcmBlockSize-1] = 1
	} else {
		// Otherwise counter = GHASH(nonce)
		gcmAesData(&g.productTable, nonce, &counter)
		gcmAesFinish(&g.productTable, &tagMask, &counter, uint64(len(nonce)), uint64(0))
	}

	aes.EncryptBlockInternal(&g.cipher, tagMask[:], counter[:])

	var expectedTag [gcmTagSize]byte
	gcmAesData(&g.productTable, data, &expectedTag)

	if len(ciphertext) > 0 {
		gcmAesDec(&g.productTable, out, ciphertext, &counter, &expectedTag, aes.EncryptionKeySchedule(&g.cipher))
	}
	gcmAesFinish(&g.productTable, &tagMask, &expectedTag, uint64(len(ciphertext)), uint64(len(data)))

	if subtle.ConstantTimeCompare(expectedTag[:g.tagSize], tag) != 1 {
		return errOpen
	}
	return nil
}
