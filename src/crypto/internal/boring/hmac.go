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
	hkey := make([]byte, len(key))
	copy(hkey, key)
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
		runtime.SetFinalizer(h, (*boringHMAC).finalize)
	}
	C._goboringcrypto_HMAC_CTX_init(&h.ctx)

	if C._goboringcrypto_HMAC_Init(&h.ctx, unsafe.Pointer(&h.key[0]), C.int(len(h.key)), h.md) == 0 {
		panic("boringcrypto: HMAC_Init failed")
	}
	if int(C._goboringcrypto_HMAC_size(&h.ctx)) != h.size {
		println("boringcrypto: HMAC size:", C._goboringcrypto_HMAC_size(&h.ctx), "!=", h.size)
		panic("boringcrypto: HMAC size mismatch")
	}
	h.sum = nil
}

func (h *boringHMAC) finalize() {
	C._goboringcrypto_HMAC_CTX_cleanup(&h.ctx)
}

var badSum = make([]byte, 1)

func (h *boringHMAC) Write(p []byte) (int, error) {
	if h.sum != nil {
		h.sum = badSum
	} else if len(p) > 0 {
		C._goboringcrypto_HMAC_Update(&h.ctx, (*C.uint8_t)(unsafe.Pointer(&p[0])), C.size_t(len(p)))
	}
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
		C._goboringcrypto_HMAC_Final(&h.ctx, (*C.uint8_t)(unsafe.Pointer(&h.sum[0])), nil)

		// Clean up and disable finalizer since most of the time
		// there is no Reset coming. If we do get a Reset, we will
		// re-enable the finalizer. We have a saved copy of the key.
		C._goboringcrypto_HMAC_CTX_cleanup(&h.ctx)
		h.needCleanup = false
		runtime.SetFinalizer(h, nil)
	} else if &h.sum[0] == &badSum[0] {
		// crypto/tls's tls10.MAC method calls Write after Sum,
		// in an attempt to do more-like-constant-time checksums
		// during TLS 1.0 SHA1-based MACs. We can't support that,
		// so we ignore the Write in that case above, but we also
		// poison h.sum so that future Sum calls panic, to avoid
		// returning the pre-Write checksum.
		// We expect no code actually does Sum, Write, Sum.
		// Under FIPS restrictions, crypto/tls would not use
		// any SHA1-based MACs anyway.
		panic("boringcrypto: hmac used with Sum, Write, Sum")
	}
	return append(in, h.sum...)
}
