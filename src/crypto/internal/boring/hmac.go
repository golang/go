// Copyright 2017 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

//go:build boringcrypto && linux && (amd64 || arm64) && !android && !cmd_go_bootstrap && !msan

package boring

// #include "goboringcrypto.h"
import "C"
import (
	"bytes"
	"crypto"
	"hash"
	"runtime"
	"unsafe"
)

// hashToMD converts a hash.Hash implementation from this package
// to a BoringCrypto *C.GO_EVP_MD.
func hashToMD(h hash.Hash) *C.GO_EVP_MD {
	switch h.(type) {
	case *sha1Hash:
		return C._goboringcrypto_EVP_sha1()
	case *sha224Hash:
		return C._goboringcrypto_EVP_sha224()
	case *sha256Hash:
		return C._goboringcrypto_EVP_sha256()
	case *sha384Hash:
		return C._goboringcrypto_EVP_sha384()
	case *sha512Hash:
		return C._goboringcrypto_EVP_sha512()
	}
	return nil
}

// cryptoHashToMD converts a crypto.Hash
// to a BoringCrypto *C.GO_EVP_MD.
func cryptoHashToMD(ch crypto.Hash) *C.GO_EVP_MD {
	switch ch {
	case crypto.MD5:
		return C._goboringcrypto_EVP_md5()
	case crypto.MD5SHA1:
		return C._goboringcrypto_EVP_md5_sha1()
	case crypto.SHA1:
		return C._goboringcrypto_EVP_sha1()
	case crypto.SHA224:
		return C._goboringcrypto_EVP_sha224()
	case crypto.SHA256:
		return C._goboringcrypto_EVP_sha256()
	case crypto.SHA384:
		return C._goboringcrypto_EVP_sha384()
	case crypto.SHA512:
		return C._goboringcrypto_EVP_sha512()
	}
	return nil
}

// NewHMAC returns a new HMAC using BoringCrypto.
// The function h must return a hash implemented by
// BoringCrypto (for example, h could be boring.NewSHA256).
// If h is not recognized, NewHMAC returns nil.
func NewHMAC(h func() hash.Hash, key []byte) hash.Hash {
	ch := h()
	md := hashToMD(ch)
	if md == nil {
		return nil
	}

	// Note: Could hash down long keys here using EVP_Digest.
	hkey := bytes.Clone(key)
	hmac := &boringHMAC{
		md:        md,
		size:      ch.Size(),
		blockSize: ch.BlockSize(),
		key:       hkey,
	}
	hmac.Reset()
	return hmac
}

type boringHMAC struct {
	md          *C.GO_EVP_MD
	ctx         C.GO_HMAC_CTX
	ctx2        C.GO_HMAC_CTX
	size        int
	blockSize   int
	key         []byte
	sum         []byte
	needCleanup bool
}

func (h *boringHMAC) Reset() {
	if h.needCleanup {
		C._goboringcrypto_HMAC_CTX_cleanup(&h.ctx)
	} else {
		h.needCleanup = true
		// Note: Because of the finalizer, any time h.ctx is passed to cgo,
		// that call must be followed by a call to runtime.KeepAlive(h),
		// to make sure h is not collected (and finalized) before the cgo
		// call returns.
		runtime.SetFinalizer(h, (*boringHMAC).finalize)
	}
	C._goboringcrypto_HMAC_CTX_init(&h.ctx)

	if C._goboringcrypto_HMAC_Init(&h.ctx, unsafe.Pointer(base(h.key)), C.int(len(h.key)), h.md) == 0 {
		panic("boringcrypto: HMAC_Init failed")
	}
	if int(C._goboringcrypto_HMAC_size(&h.ctx)) != h.size {
		println("boringcrypto: HMAC size:", C._goboringcrypto_HMAC_size(&h.ctx), "!=", h.size)
		panic("boringcrypto: HMAC size mismatch")
	}
	runtime.KeepAlive(h) // Next line will keep h alive too; just making doubly sure.
	h.sum = nil
}

func (h *boringHMAC) finalize() {
	C._goboringcrypto_HMAC_CTX_cleanup(&h.ctx)
}

func (h *boringHMAC) Write(p []byte) (int, error) {
	if len(p) > 0 {
		C._goboringcrypto_HMAC_Update(&h.ctx, (*C.uint8_t)(unsafe.Pointer(&p[0])), C.size_t(len(p)))
	}
	runtime.KeepAlive(h)
	return len(p), nil
}

func (h *boringHMAC) Size() int {
	return h.size
}

func (h *boringHMAC) BlockSize() int {
	return h.blockSize
}

func (h *boringHMAC) Sum(in []byte) []byte {
	if h.sum == nil {
		size := h.Size()
		h.sum = make([]byte, size)
	}
	// Make copy of context because Go hash.Hash mandates
	// that Sum has no effect on the underlying stream.
	// In particular it is OK to Sum, then Write more, then Sum again,
	// and the second Sum acts as if the first didn't happen.
	C._goboringcrypto_HMAC_CTX_init(&h.ctx2)
	if C._goboringcrypto_HMAC_CTX_copy_ex(&h.ctx2, &h.ctx) == 0 {
		panic("boringcrypto: HMAC_CTX_copy_ex failed")
	}
	C._goboringcrypto_HMAC_Final(&h.ctx2, (*C.uint8_t)(unsafe.Pointer(&h.sum[0])), nil)
	C._goboringcrypto_HMAC_CTX_cleanup(&h.ctx2)
	return append(in, h.sum...)
}
