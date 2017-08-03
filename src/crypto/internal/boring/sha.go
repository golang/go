// Copyright 2017 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

// +build linux,amd64
// +build !cmd_go_bootstrap

package boring

// #include "goboringcrypto.h"
import "C"
import (
	"hash"
	"unsafe"
)

// NewSHA1 returns a new SHA1 hash.
func NewSHA1() hash.Hash {
	h := new(sha1Hash)
	h.Reset()
	return h
}

type sha1Hash struct {
	ctx C.GO_SHA_CTX
	out [20]byte
}

func (h *sha1Hash) Reset()               { C._goboringcrypto_SHA1_Init(&h.ctx) }
func (h *sha1Hash) Size() int            { return 20 }
func (h *sha1Hash) BlockSize() int       { return 64 }
func (h *sha1Hash) Sum(in []byte) []byte { return append(in, h.sum()...) }

func (h *sha1Hash) Write(p []byte) (int, error) {
	if len(p) > 0 && C._goboringcrypto_SHA1_Update(&h.ctx, unsafe.Pointer(&p[0]), C.size_t(len(p))) == 0 {
		panic("boringcrypto: SHA1_Update failed")
	}
	return len(p), nil
}

func (h0 *sha1Hash) sum() []byte {
	h := *h0 // make copy so future Write+Sum is valid
	if C._goboringcrypto_SHA1_Final((*C.uint8_t)(unsafe.Pointer(&h.out[0])), &h.ctx) == 0 {
		panic("boringcrypto: SHA1_Final failed")
	}
	return h.out[:]
}

// NewSHA224 returns a new SHA224 hash.
func NewSHA224() hash.Hash {
	h := new(sha224Hash)
	h.Reset()
	return h
}

type sha224Hash struct {
	ctx C.GO_SHA256_CTX
	out [224 / 8]byte
}

func (h *sha224Hash) Reset()               { C._goboringcrypto_SHA224_Init(&h.ctx) }
func (h *sha224Hash) Size() int            { return 224 / 8 }
func (h *sha224Hash) BlockSize() int       { return 64 }
func (h *sha224Hash) Sum(in []byte) []byte { return append(in, h.sum()...) }

func (h *sha224Hash) Write(p []byte) (int, error) {
	if len(p) > 0 && C._goboringcrypto_SHA224_Update(&h.ctx, unsafe.Pointer(&p[0]), C.size_t(len(p))) == 0 {
		panic("boringcrypto: SHA224_Update failed")
	}
	return len(p), nil
}

func (h0 *sha224Hash) sum() []byte {
	h := *h0 // make copy so future Write+Sum is valid
	if C._goboringcrypto_SHA224_Final((*C.uint8_t)(unsafe.Pointer(&h.out[0])), &h.ctx) == 0 {
		panic("boringcrypto: SHA224_Final failed")
	}
	return h.out[:]
}

// NewSHA256 returns a new SHA256 hash.
func NewSHA256() hash.Hash {
	h := new(sha256Hash)
	h.Reset()
	return h
}

type sha256Hash struct {
	ctx C.GO_SHA256_CTX
	out [256 / 8]byte
}

func (h *sha256Hash) Reset()               { C._goboringcrypto_SHA256_Init(&h.ctx) }
func (h *sha256Hash) Size() int            { return 256 / 8 }
func (h *sha256Hash) BlockSize() int       { return 64 }
func (h *sha256Hash) Sum(in []byte) []byte { return append(in, h.sum()...) }

func (h *sha256Hash) Write(p []byte) (int, error) {
	if len(p) > 0 && C._goboringcrypto_SHA256_Update(&h.ctx, unsafe.Pointer(&p[0]), C.size_t(len(p))) == 0 {
		panic("boringcrypto: SHA256_Update failed")
	}
	return len(p), nil
}

func (h0 *sha256Hash) sum() []byte {
	h := *h0 // make copy so future Write+Sum is valid
	if C._goboringcrypto_SHA256_Final((*C.uint8_t)(unsafe.Pointer(&h.out[0])), &h.ctx) == 0 {
		panic("boringcrypto: SHA256_Final failed")
	}
	return h.out[:]
}

// NewSHA384 returns a new SHA384 hash.
func NewSHA384() hash.Hash {
	h := new(sha384Hash)
	h.Reset()
	return h
}

type sha384Hash struct {
	ctx C.GO_SHA512_CTX
	out [384 / 8]byte
}

func (h *sha384Hash) Reset()               { C._goboringcrypto_SHA384_Init(&h.ctx) }
func (h *sha384Hash) Size() int            { return 384 / 8 }
func (h *sha384Hash) BlockSize() int       { return 128 }
func (h *sha384Hash) Sum(in []byte) []byte { return append(in, h.sum()...) }

func (h *sha384Hash) Write(p []byte) (int, error) {
	if len(p) > 0 && C._goboringcrypto_SHA384_Update(&h.ctx, unsafe.Pointer(&p[0]), C.size_t(len(p))) == 0 {
		panic("boringcrypto: SHA384_Update failed")
	}
	return len(p), nil
}

func (h0 *sha384Hash) sum() []byte {
	h := *h0 // make copy so future Write+Sum is valid
	if C._goboringcrypto_SHA384_Final((*C.uint8_t)(unsafe.Pointer(&h.out[0])), &h.ctx) == 0 {
		panic("boringcrypto: SHA384_Final failed")
	}
	return h.out[:]
}

// NewSHA512 returns a new SHA512 hash.
func NewSHA512() hash.Hash {
	h := new(sha512Hash)
	h.Reset()
	return h
}

type sha512Hash struct {
	ctx C.GO_SHA512_CTX
	out [512 / 8]byte
}

func (h *sha512Hash) Reset()               { C._goboringcrypto_SHA512_Init(&h.ctx) }
func (h *sha512Hash) Size() int            { return 512 / 8 }
func (h *sha512Hash) BlockSize() int       { return 128 }
func (h *sha512Hash) Sum(in []byte) []byte { return append(in, h.sum()...) }

func (h *sha512Hash) Write(p []byte) (int, error) {
	if len(p) > 0 && C._goboringcrypto_SHA512_Update(&h.ctx, unsafe.Pointer(&p[0]), C.size_t(len(p))) == 0 {
		panic("boringcrypto: SHA512_Update failed")
	}
	return len(p), nil
}

func (h0 *sha512Hash) sum() []byte {
	h := *h0 // make copy so future Write+Sum is valid
	if C._goboringcrypto_SHA512_Final((*C.uint8_t)(unsafe.Pointer(&h.out[0])), &h.ctx) == 0 {
		panic("boringcrypto: SHA512_Final failed")
	}
	return h.out[:]
}
