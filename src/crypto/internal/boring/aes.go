// Copyright 2017 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

//go:build boringcrypto && linux && (amd64 || arm64) && !android && !msan

package boring

/*

#include "goboringcrypto.h"

// These wrappers allocate out_len on the C stack, and check that it matches the expected
// value, to avoid having to pass a pointer from Go, which would escape to the heap.

int EVP_AEAD_CTX_seal_wrapper(const GO_EVP_AEAD_CTX *ctx, uint8_t *out,
							  size_t exp_out_len,
							  const uint8_t *nonce, size_t nonce_len,
							  const uint8_t *in, size_t in_len,
							  const uint8_t *ad, size_t ad_len) {
	size_t out_len;
	int ok = _goboringcrypto_EVP_AEAD_CTX_seal(ctx, out, &out_len, exp_out_len,
		nonce, nonce_len, in, in_len, ad, ad_len);
	if (out_len != exp_out_len) {
		return 0;
	}
	return ok;
};

int EVP_AEAD_CTX_open_wrapper(const GO_EVP_AEAD_CTX *ctx, uint8_t *out,
							  size_t exp_out_len,
							  const uint8_t *nonce, size_t nonce_len,
							  const uint8_t *in, size_t in_len,
							  const uint8_t *ad, size_t ad_len) {
	size_t out_len;
	int ok = _goboringcrypto_EVP_AEAD_CTX_open(ctx, out, &out_len, exp_out_len,
		nonce, nonce_len, in, in_len, ad, ad_len);
	if (out_len != exp_out_len) {
		return 0;
	}
	return ok;
};

*/
import "C"
import (
	"bytes"
	"crypto/cipher"
	"errors"
	"runtime"
	"strconv"
	"unsafe"
)

type aesKeySizeError int

func (k aesKeySizeError) Error() string {
	return "crypto/aes: invalid key size " + strconv.Itoa(int(k))
}

const aesBlockSize = 16

type aesCipher struct {
	key []byte
	enc C.GO_AES_KEY
	dec C.GO_AES_KEY
}

type extraModes interface {
	// Copied out of crypto/aes/modes.go.
	NewCBCEncrypter(iv []byte) cipher.BlockMode
	NewCBCDecrypter(iv []byte) cipher.BlockMode
	NewCTR(iv []byte) cipher.Stream
	NewGCM(nonceSize, tagSize int) (cipher.AEAD, error)
}

var _ extraModes = (*aesCipher)(nil)

func NewAESCipher(key []byte) (cipher.Block, error) {
	c := &aesCipher{key: bytes.Clone(key)}
	// Note: 0 is success, contradicting the usual BoringCrypto convention.
	if C._goboringcrypto_AES_set_decrypt_key((*C.uint8_t)(unsafe.Pointer(&c.key[0])), C.uint(8*len(c.key)), &c.dec) != 0 ||
		C._goboringcrypto_AES_set_encrypt_key((*C.uint8_t)(unsafe.Pointer(&c.key[0])), C.uint(8*len(c.key)), &c.enc) != 0 {
		return nil, aesKeySizeError(len(key))
	}
	return c, nil
}

func (c *aesCipher) BlockSize() int { return aesBlockSize }

func (c *aesCipher) Encrypt(dst, src []byte) {
	if inexactOverlap(dst, src) {
		panic("crypto/cipher: invalid buffer overlap")
	}
	if len(src) < aesBlockSize {
		panic("crypto/aes: input not full block")
	}
	if len(dst) < aesBlockSize {
		panic("crypto/aes: output not full block")
	}
	C._goboringcrypto_AES_encrypt(
		(*C.uint8_t)(unsafe.Pointer(&src[0])),
		(*C.uint8_t)(unsafe.Pointer(&dst[0])),
		&c.enc)
}

func (c *aesCipher) Decrypt(dst, src []byte) {
	if inexactOverlap(dst, src) {
		panic("crypto/cipher: invalid buffer overlap")
	}
	if len(src) < aesBlockSize {
		panic("crypto/aes: input not full block")
	}
	if len(dst) < aesBlockSize {
		panic("crypto/aes: output not full block")
	}
	C._goboringcrypto_AES_decrypt(
		(*C.uint8_t)(unsafe.Pointer(&src[0])),
		(*C.uint8_t)(unsafe.Pointer(&dst[0])),
		&c.dec)
}

type aesCBC struct {
	key  *C.GO_AES_KEY
	mode C.int
	iv   [aesBlockSize]byte
}

func (x *aesCBC) BlockSize() int { return aesBlockSize }

func (x *aesCBC) CryptBlocks(dst, src []byte) {
	if inexactOverlap(dst, src) {
		panic("crypto/cipher: invalid buffer overlap")
	}
	if len(src)%aesBlockSize != 0 {
		panic("crypto/cipher: input not full blocks")
	}
	if len(dst) < len(src) {
		panic("crypto/cipher: output smaller than input")
	}
	if len(src) > 0 {
		C._goboringcrypto_AES_cbc_encrypt(
			(*C.uint8_t)(unsafe.Pointer(&src[0])),
			(*C.uint8_t)(unsafe.Pointer(&dst[0])),
			C.size_t(len(src)), x.key,
			(*C.uint8_t)(unsafe.Pointer(&x.iv[0])), x.mode)
	}
}

func (x *aesCBC) SetIV(iv []byte) {
	if len(iv) != aesBlockSize {
		panic("cipher: incorrect length IV")
	}
	copy(x.iv[:], iv)
}

func (c *aesCipher) NewCBCEncrypter(iv []byte) cipher.BlockMode {
	x := &aesCBC{key: &c.enc, mode: C.GO_AES_ENCRYPT}
	copy(x.iv[:], iv)
	return x
}

func (c *aesCipher) NewCBCDecrypter(iv []byte) cipher.BlockMode {
	x := &aesCBC{key: &c.dec, mode: C.GO_AES_DECRYPT}
	copy(x.iv[:], iv)
	return x
}

type aesCTR struct {
	key        *C.GO_AES_KEY
	iv         [aesBlockSize]byte
	num        C.uint
	ecount_buf [16]C.uint8_t
}

func (x *aesCTR) XORKeyStream(dst, src []byte) {
	if inexactOverlap(dst, src) {
		panic("crypto/cipher: invalid buffer overlap")
	}
	if len(dst) < len(src) {
		panic("crypto/cipher: output smaller than input")
	}
	if len(src) == 0 {
		return
	}
	C._goboringcrypto_AES_ctr128_encrypt(
		(*C.uint8_t)(unsafe.Pointer(&src[0])),
		(*C.uint8_t)(unsafe.Pointer(&dst[0])),
		C.size_t(len(src)), x.key, (*C.uint8_t)(unsafe.Pointer(&x.iv[0])),
		&x.ecount_buf[0], &x.num)
}

func (c *aesCipher) NewCTR(iv []byte) cipher.Stream {
	x := &aesCTR{key: &c.enc}
	copy(x.iv[:], iv)
	return x
}

type aesGCM struct {
	ctx  C.GO_EVP_AEAD_CTX
	aead *C.GO_EVP_AEAD
}

const (
	gcmBlockSize         = 16
	gcmTagSize           = 16
	gcmStandardNonceSize = 12
)

type aesNonceSizeError int

func (n aesNonceSizeError) Error() string {
	return "crypto/aes: invalid GCM nonce size " + strconv.Itoa(int(n))
}

type noGCM struct {
	cipher.Block
}

func (c *aesCipher) NewGCM(nonceSize, tagSize int) (cipher.AEAD, error) {
	if nonceSize != gcmStandardNonceSize && tagSize != gcmTagSize {
		return nil, errors.New("crypto/aes: GCM tag and nonce sizes can't be non-standard at the same time")
	}
	// Fall back to standard library for GCM with non-standard nonce or tag size.
	if nonceSize != gcmStandardNonceSize {
		return cipher.NewGCMWithNonceSize(&noGCM{c}, nonceSize)
	}
	if tagSize != gcmTagSize {
		return cipher.NewGCMWithTagSize(&noGCM{c}, tagSize)
	}
	return c.newGCM(false)
}

func NewGCMTLS(c cipher.Block) (cipher.AEAD, error) {
	return c.(*aesCipher).newGCM(true)
}

func (c *aesCipher) newGCM(tls bool) (cipher.AEAD, error) {
	var aead *C.GO_EVP_AEAD
	switch len(c.key) * 8 {
	case 128:
		if tls {
			aead = C._goboringcrypto_EVP_aead_aes_128_gcm_tls12()
		} else {
			aead = C._goboringcrypto_EVP_aead_aes_128_gcm()
		}
	case 256:
		if tls {
			aead = C._goboringcrypto_EVP_aead_aes_256_gcm_tls12()
		} else {
			aead = C._goboringcrypto_EVP_aead_aes_256_gcm()
		}
	default:
		// Fall back to standard library for GCM with non-standard key size.
		return cipher.NewGCMWithNonceSize(&noGCM{c}, gcmStandardNonceSize)
	}

	g := &aesGCM{aead: aead}
	if C._goboringcrypto_EVP_AEAD_CTX_init(&g.ctx, aead, (*C.uint8_t)(unsafe.Pointer(&c.key[0])), C.size_t(len(c.key)), C.GO_EVP_AEAD_DEFAULT_TAG_LENGTH, nil) == 0 {
		return nil, fail("EVP_AEAD_CTX_init")
	}
	// Note: Because of the finalizer, any time g.ctx is passed to cgo,
	// that call must be followed by a call to runtime.KeepAlive(g),
	// to make sure g is not collected (and finalized) before the cgo
	// call returns.
	runtime.SetFinalizer(g, (*aesGCM).finalize)
	if g.NonceSize() != gcmStandardNonceSize {
		panic("boringcrypto: internal confusion about nonce size")
	}
	if g.Overhead() != gcmTagSize {
		panic("boringcrypto: internal confusion about tag size")
	}

	return g, nil
}

func (g *aesGCM) finalize() {
	C._goboringcrypto_EVP_AEAD_CTX_cleanup(&g.ctx)
}

func (g *aesGCM) NonceSize() int {
	return int(C._goboringcrypto_EVP_AEAD_nonce_length(g.aead))
}

func (g *aesGCM) Overhead() int {
	return int(C._goboringcrypto_EVP_AEAD_max_overhead(g.aead))
}

// base returns the address of the underlying array in b,
// being careful not to panic when b has zero length.
func base(b []byte) *C.uint8_t {
	if len(b) == 0 {
		return nil
	}
	return (*C.uint8_t)(unsafe.Pointer(&b[0]))
}

func (g *aesGCM) Seal(dst, nonce, plaintext, additionalData []byte) []byte {
	if len(nonce) != gcmStandardNonceSize {
		panic("cipher: incorrect nonce length given to GCM")
	}
	if uint64(len(plaintext)) > ((1<<32)-2)*aesBlockSize || len(plaintext)+gcmTagSize < len(plaintext) {
		panic("cipher: message too large for GCM")
	}
	if len(dst)+len(plaintext)+gcmTagSize < len(dst) {
		panic("cipher: message too large for buffer")
	}

	// Make room in dst to append plaintext+overhead.
	n := len(dst)
	for cap(dst) < n+len(plaintext)+gcmTagSize {
		dst = append(dst[:cap(dst)], 0)
	}
	dst = dst[:n+len(plaintext)+gcmTagSize]

	// Check delayed until now to make sure len(dst) is accurate.
	if inexactOverlap(dst[n:], plaintext) {
		panic("cipher: invalid buffer overlap")
	}

	outLen := C.size_t(len(plaintext) + gcmTagSize)
	ok := C.EVP_AEAD_CTX_seal_wrapper(
		&g.ctx,
		(*C.uint8_t)(unsafe.Pointer(&dst[n])), outLen,
		base(nonce), C.size_t(len(nonce)),
		base(plaintext), C.size_t(len(plaintext)),
		base(additionalData), C.size_t(len(additionalData)))
	runtime.KeepAlive(g)
	if ok == 0 {
		panic(fail("EVP_AEAD_CTX_seal"))
	}
	return dst[:n+int(outLen)]
}

var errOpen = errors.New("cipher: message authentication failed")

func (g *aesGCM) Open(dst, nonce, ciphertext, additionalData []byte) ([]byte, error) {
	if len(nonce) != gcmStandardNonceSize {
		panic("cipher: incorrect nonce length given to GCM")
	}
	if len(ciphertext) < gcmTagSize {
		return nil, errOpen
	}
	if uint64(len(ciphertext)) > ((1<<32)-2)*aesBlockSize+gcmTagSize {
		return nil, errOpen
	}

	// Make room in dst to append ciphertext without tag.
	n := len(dst)
	for cap(dst) < n+len(ciphertext)-gcmTagSize {
		dst = append(dst[:cap(dst)], 0)
	}
	dst = dst[:n+len(ciphertext)-gcmTagSize]

	// Check delayed until now to make sure len(dst) is accurate.
	if inexactOverlap(dst[n:], ciphertext) {
		panic("cipher: invalid buffer overlap")
	}

	outLen := C.size_t(len(ciphertext) - gcmTagSize)
	ok := C.EVP_AEAD_CTX_open_wrapper(
		&g.ctx,
		base(dst[n:]), outLen,
		base(nonce), C.size_t(len(nonce)),
		base(ciphertext), C.size_t(len(ciphertext)),
		base(additionalData), C.size_t(len(additionalData)))
	runtime.KeepAlive(g)
	if ok == 0 {
		return nil, errOpen
	}
	return dst[:n+int(outLen)], nil
}

func anyOverlap(x, y []byte) bool {
	return len(x) > 0 && len(y) > 0 &&
		uintptr(unsafe.Pointer(&x[0])) <= uintptr(unsafe.Pointer(&y[len(y)-1])) &&
		uintptr(unsafe.Pointer(&y[0])) <= uintptr(unsafe.Pointer(&x[len(x)-1]))
}

func inexactOverlap(x, y []byte) bool {
	if len(x) == 0 || len(y) == 0 || &x[0] == &y[0] {
		return false
	}
	return anyOverlap(x, y)
}
