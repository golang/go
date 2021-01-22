// Copyright 2017 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

// +build linux
// +build openssl
// +build !android
// +build !no_openssl
// +build !cmd_go_bootstrap
// +build !msan

package openssl

// #include "goopenssl.h"
import "C"
import (
	"crypto/cipher"
	"errors"
	"runtime"
	"strconv"
	"unsafe"
)

type aes struct{}

type aesKeySizeError int

func (k aesKeySizeError) Error() string {
	return "crypto/aes: invalid key size " + strconv.Itoa(int(k))
}

const aesBlockSize = 16

type aesCipher struct {
	key     []byte
	enc_ctx *C.EVP_CIPHER_CTX
	dec_ctx *C.EVP_CIPHER_CTX
	enc     C.GO_AES_KEY
	dec     C.GO_AES_KEY
	cipher  *C.EVP_CIPHER
}

type extraModes interface {
	// Copied out of crypto/aes/modes.go.
	NewCBCEncrypter(iv []byte) cipher.BlockMode
	NewCBCDecrypter(iv []byte) cipher.BlockMode
	NewCTR(iv []byte) cipher.Stream
	NewGCM(nonceSize, tagSize int) (cipher.AEAD, error)

	// Invented for BoringCrypto.
	NewGCMTLS() (cipher.AEAD, error)
}

var _ extraModes = (*aesCipher)(nil)

func (_ aes) NewAESCipher(key []byte) (cipher.Block, error) {
	c := &aesCipher{key: make([]byte, len(key))}
	copy(c.key, key)

	// Note: 0 is success, contradicting the usual BoringCrypto convention.
	if C._goboringcrypto_AES_set_decrypt_key((*C.uint8_t)(unsafe.Pointer(&c.key[0])), C.uint(8*len(c.key)), &c.dec) != 0 ||
		C._goboringcrypto_AES_set_encrypt_key((*C.uint8_t)(unsafe.Pointer(&c.key[0])), C.uint(8*len(c.key)), &c.enc) != 0 {
		return nil, aesKeySizeError(len(key))
	}

	switch len(c.key) * 8 {
	case 128:
		c.cipher = C._goboringcrypto_EVP_aes_128_ecb()
	case 192:
		c.cipher = C._goboringcrypto_EVP_aes_192_ecb()
	case 256:
		c.cipher = C._goboringcrypto_EVP_aes_256_ecb()
	default:
		return nil, errors.New("crypto/cipher: Invalid key size")
	}

	runtime.SetFinalizer(c, (*aesCipher).finalize)

	return c, nil
}

func (c *aesCipher) finalize() {
	if c.enc_ctx != nil {
		C._goboringcrypto_EVP_CIPHER_CTX_free(c.enc_ctx)
	}
	if c.dec_ctx != nil {
		C._goboringcrypto_EVP_CIPHER_CTX_free(c.dec_ctx)
	}
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

	if c.enc_ctx == nil {
		c.enc_ctx = C._goboringcrypto_EVP_CIPHER_CTX_new()
		if c.enc_ctx == nil {
			panic("cipher: unable to create EVP cipher ctx")
		}

		k := (*C.uchar)(unsafe.Pointer(&c.key[0]))

		if C.int(1) != C._goboringcrypto_EVP_CipherInit_ex(c.enc_ctx, c.cipher, nil, k, nil, C.GO_AES_ENCRYPT) {
			panic("cipher: unable to initialize EVP cipher ctx")
		}
	}

	outlen := C.int(0)
	C._goboringcrypto_EVP_CipherUpdate(c.enc_ctx, (*C.uchar)(unsafe.Pointer(&dst[0])), &outlen, (*C.uchar)(unsafe.Pointer(&src[0])), C.int(aesBlockSize))
	runtime.KeepAlive(c)
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
	if c.dec_ctx == nil {
		c.dec_ctx = C._goboringcrypto_EVP_CIPHER_CTX_new()
		if c.dec_ctx == nil {
			panic("cipher: unable to create EVP cipher ctx")
		}

		k := (*C.uchar)(unsafe.Pointer(&c.key[0]))

		if C.int(1) != C._goboringcrypto_EVP_CipherInit_ex(c.dec_ctx, c.cipher, nil, k, nil, C.GO_AES_DECRYPT) {
			panic("cipher: unable to initialize EVP cipher ctx")
		}
	}

	outlen := C.int(0)
	C._goboringcrypto_EVP_CipherUpdate(c.dec_ctx, (*C.uchar)(unsafe.Pointer(&dst[0])), &outlen, (*C.uchar)(unsafe.Pointer(&src[0])), C.int(aesBlockSize))
	runtime.KeepAlive(c)
}

type aesCBC struct {
	key  []byte
	_key *C.GO_AES_KEY
	mode C.int
	iv   [aesBlockSize]byte
	ctx  *C.EVP_CIPHER_CTX
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
			C.size_t(len(src)), x._key,
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
	x := &aesCBC{key: c.key, _key: &c.enc, mode: C.GO_AES_ENCRYPT}
	copy(x.iv[:], iv)

	x.ctx = C._goboringcrypto_EVP_CIPHER_CTX_new()
	if x.ctx == nil {
		panic("cipher: unable to create EVP cipher ctx")
	}

	k := (*C.uchar)(unsafe.Pointer(&x.key[0]))
	vec := (*C.uchar)(unsafe.Pointer(&x.iv[0]))

	switch len(c.key) * 8 {
	case 128:
		c.cipher = C._goboringcrypto_EVP_aes_128_cbc()
	case 192:
		c.cipher = C._goboringcrypto_EVP_aes_192_cbc()
	case 256:
		c.cipher = C._goboringcrypto_EVP_aes_256_cbc()
	default:
		panic("crypto/boring: unsupported key length")
	}
	if C.int(1) != C._goboringcrypto_EVP_CipherInit_ex(x.ctx, c.cipher, nil, k, vec, x.mode) {
		panic("cipher: unable to initialize EVP cipher ctx")
	}

	runtime.SetFinalizer(x, (*aesCBC).finalize)

	return x
}

func (c *aesCBC) finalize() {
	C._goboringcrypto_EVP_CIPHER_CTX_free(c.ctx)
}

func (c *aesCipher) NewCBCDecrypter(iv []byte) cipher.BlockMode {
	x := &aesCBC{key: c.key, _key: &c.dec, mode: C.GO_AES_DECRYPT}
	copy(x.iv[:], iv)

	x.ctx = C._goboringcrypto_EVP_CIPHER_CTX_new()
	if x.ctx == nil {
		panic("cipher: unable to create EVP cipher ctx")
	}

	k := (*C.uchar)(unsafe.Pointer(&x.key[0]))
	vec := (*C.uchar)(unsafe.Pointer(&x.iv[0]))

	var cipher *C.EVP_CIPHER
	switch len(c.key) * 8 {
	case 128:
		cipher = C._goboringcrypto_EVP_aes_128_cbc()
	case 192:
		cipher = C._goboringcrypto_EVP_aes_192_cbc()
	case 256:
		cipher = C._goboringcrypto_EVP_aes_256_cbc()
	}
	if C.int(1) != C._goboringcrypto_EVP_CipherInit_ex(x.ctx, cipher, nil, k, vec, x.mode) {
		panic("cipher: unable to initialize EVP cipher ctx")
	}

	runtime.SetFinalizer(x, (*aesCBC).finalize)
	return x
}

type aesCTR struct {
	key        []byte
	iv         [aesBlockSize]byte
	ctx        *C.EVP_CIPHER_CTX
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
	C._goboringcrypto_EVP_AES_ctr128_enc(
		x.ctx,
		(*C.uint8_t)(unsafe.Pointer(&src[0])),
		(*C.uint8_t)(unsafe.Pointer(&dst[0])),
		C.size_t(len(src)))
	runtime.KeepAlive(x)
}

func (c *aesCipher) NewCTR(iv []byte) cipher.Stream {
	x := &aesCTR{key: c.key}
	copy(x.iv[:], iv)

	x.ctx = C._goboringcrypto_EVP_CIPHER_CTX_new()
	if x.ctx == nil {
		panic("cipher: unable to create EVP cipher ctx")
	}

	k := (*C.uchar)(unsafe.Pointer(&x.key[0]))
	vec := (*C.uchar)(unsafe.Pointer(&x.iv[0]))

	switch len(c.key) * 8 {
	case 128:
		if C.int(1) != C._goboringcrypto_EVP_EncryptInit_ex(x.ctx, C._goboringcrypto_EVP_aes_128_ctr(), nil, k, vec) {
			panic("cipher: unable to initialize EVP cipher ctx")
		}
	case 192:
		if C.int(1) != C._goboringcrypto_EVP_EncryptInit_ex(x.ctx, C._goboringcrypto_EVP_aes_192_ctr(), nil, k, vec) {
			panic("cipher: unable to initialize EVP cipher ctx")
		}
	case 256:
		if C.int(1) != C._goboringcrypto_EVP_EncryptInit_ex(x.ctx, C._goboringcrypto_EVP_aes_256_ctr(), nil, k, vec) {
			panic("cipher: unable to initialize EVP cipher ctx")
		}
	}

	runtime.SetFinalizer(x, (*aesCTR).finalize)

	return x
}

func (c *aesCTR) finalize() {
	C._goboringcrypto_EVP_CIPHER_CTX_free(c.ctx)
}

type aesGCM struct {
	key []byte
	tls bool
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
	if nonceSize != gcmStandardNonceSize {
		panicIfStrictFIPS("crypto/aes: GCM nonce size can't be non-standard")
		return cipher.NewGCMWithNonceSize(&noGCM{c}, nonceSize)
	}
	if tagSize != gcmTagSize {
		panicIfStrictFIPS("crypto/aes: GCM tag size can't be non-standard")
		return cipher.NewGCMWithTagSize(&noGCM{c}, tagSize)
	}
	return c.newGCM(false)
}

func (c *aesCipher) NewGCMTLS() (cipher.AEAD, error) {
	return c.newGCM(true)
}

func (c *aesCipher) newGCM(tls bool) (cipher.AEAD, error) {
	keyLen := len(c.key) * 8

	if keyLen != 128 && keyLen != 256 {
		// Return error for GCM with non-standard key size.
		panicIfStrictFIPS("GCM invoked with non-standard key size")
	}

	g := &aesGCM{key: c.key, tls: tls}
	if g.NonceSize() != gcmStandardNonceSize {
		panic(fail("internal confusion about nonce size"))
	}
	if g.Overhead() != gcmTagSize {
		panic(fail("internal confusion about tag size"))
	}

	return g, nil
}

func (g *aesGCM) NonceSize() int {
	return gcmStandardNonceSize
}

func (g *aesGCM) Overhead() int {
	return gcmTagSize
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

	var ciphertextLen C.size_t
	ok := C._goboringcrypto_EVP_CIPHER_CTX_seal(
		(*C.uint8_t)(unsafe.Pointer(&dst[n])),
		base(nonce), base(additionalData), C.size_t(len(additionalData)),
		base(plaintext), C.size_t(len(plaintext)), &ciphertextLen,
		base(g.key), C.int(len(g.key)*8))

	runtime.KeepAlive(g)
	if ok != 1 {
		panic(fail("EVP_CIPHER_CTX_seal fail"))
	}
	return dst[:n+int(ciphertextLen)]
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

	tag := ciphertext[len(ciphertext)-gcmTagSize:]

	var outLen C.size_t

	ok := C._goboringcrypto_EVP_CIPHER_CTX_open(
		base(ciphertext), C.int(len(ciphertext)-gcmTagSize),
		base(additionalData), C.int(len(additionalData)),
		base(tag), base(g.key), C.int(len(g.key)*8),
		base(nonce), C.int(len(nonce)),
		base(dst[n:]), &outLen)
	runtime.KeepAlive(g)
	if ok == 0 {
		// Zero output buffer on error.
		for i := range dst {
			dst[i] = 0
		}
		return nil, errOpen
	}
	if outLen != C.size_t(len(ciphertext)-gcmTagSize) {
		panic("boringcrypto: [open] internal confusion about GCM tag size")
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
