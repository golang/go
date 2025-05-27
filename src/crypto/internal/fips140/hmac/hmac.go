// Copyright 2009 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

// Package hmac implements HMAC according to [FIPS 198-1].
//
// [FIPS 198-1]: https://doi.org/10.6028/NIST.FIPS.198-1
package hmac

import (
	"crypto/internal/fips140"
	"crypto/internal/fips140/sha256"
	"crypto/internal/fips140/sha3"
	"crypto/internal/fips140/sha512"
	"errors"
	"hash"
)

// key is zero padded to the block size of the hash function
// ipad = 0x36 byte repeated for key length
// opad = 0x5c byte repeated for key length
// hmac = H([key ^ opad] H([key ^ ipad] text))

// marshalable is the combination of encoding.BinaryMarshaler and
// encoding.BinaryUnmarshaler. Their method definitions are repeated here to
// avoid a dependency on the encoding package.
type marshalable interface {
	MarshalBinary() ([]byte, error)
	UnmarshalBinary([]byte) error
}

type HMAC struct {
	// opad and ipad may share underlying storage with HMAC clones.
	opad, ipad   []byte
	outer, inner hash.Hash

	// If marshaled is true, then opad and ipad do not contain a padded
	// copy of the key, but rather the marshaled state of outer/inner after
	// opad/ipad has been fed into it.
	marshaled bool

	// forHKDF and keyLen are stored to inform the service indicator decision.
	forHKDF bool
	keyLen  int
}

func (h *HMAC) Sum(in []byte) []byte {
	// Per FIPS 140-3 IG C.M, key lengths below 112 bits are only allowed for
	// legacy use (i.e. verification only) and we don't support that. However,
	// HKDF uses the HMAC key for the salt, which is allowed to be shorter.
	if h.keyLen < 112/8 && !h.forHKDF {
		fips140.RecordNonApproved()
	}
	switch h.inner.(type) {
	case *sha256.Digest, *sha512.Digest, *sha3.Digest:
	default:
		fips140.RecordNonApproved()
	}

	origLen := len(in)
	in = h.inner.Sum(in)

	if h.marshaled {
		if err := h.outer.(marshalable).UnmarshalBinary(h.opad); err != nil {
			panic(err)
		}
	} else {
		h.outer.Reset()
		h.outer.Write(h.opad)
	}
	h.outer.Write(in[origLen:])
	return h.outer.Sum(in[:origLen])
}

func (h *HMAC) Write(p []byte) (n int, err error) {
	return h.inner.Write(p)
}

func (h *HMAC) Size() int      { return h.outer.Size() }
func (h *HMAC) BlockSize() int { return h.inner.BlockSize() }

func (h *HMAC) Reset() {
	if h.marshaled {
		if err := h.inner.(marshalable).UnmarshalBinary(h.ipad); err != nil {
			panic(err)
		}
		return
	}

	h.inner.Reset()
	h.inner.Write(h.ipad)

	// If the underlying hash is marshalable, we can save some time by saving a
	// copy of the hash state now, and restoring it on future calls to Reset and
	// Sum instead of writing ipad/opad every time.
	//
	// We do this on Reset to avoid slowing down the common single-use case.
	//
	// This is allowed by FIPS 198-1, Section 6: "Conceptually, the intermediate
	// results of the compression function on the B-byte blocks (K0 ⊕ ipad) and
	// (K0 ⊕ opad) can be precomputed once, at the time of generation of the key
	// K, or before its first use. These intermediate results can be stored and
	// then used to initialize H each time that a message needs to be
	// authenticated using the same key. [...] These stored intermediate values
	// shall be treated and protected in the same manner as secret keys."
	marshalableInner, innerOK := h.inner.(marshalable)
	if !innerOK {
		return
	}
	marshalableOuter, outerOK := h.outer.(marshalable)
	if !outerOK {
		return
	}

	imarshal, err := marshalableInner.MarshalBinary()
	if err != nil {
		return
	}

	h.outer.Reset()
	h.outer.Write(h.opad)
	omarshal, err := marshalableOuter.MarshalBinary()
	if err != nil {
		return
	}

	// Marshaling succeeded; save the marshaled state for later
	h.ipad = imarshal
	h.opad = omarshal
	h.marshaled = true
}

// Clone implements [hash.Cloner] if the underlying hash does.
// Otherwise, it returns [errors.ErrUnsupported].
func (h *HMAC) Clone() (hash.Cloner, error) {
	r := *h
	ic, ok := h.inner.(hash.Cloner)
	if !ok {
		return nil, errors.ErrUnsupported
	}
	oc, ok := h.outer.(hash.Cloner)
	if !ok {
		return nil, errors.ErrUnsupported
	}
	var err error
	r.inner, err = ic.Clone()
	if err != nil {
		return nil, errors.ErrUnsupported
	}
	r.outer, err = oc.Clone()
	if err != nil {
		return nil, errors.ErrUnsupported
	}
	return &r, nil
}

// New returns a new HMAC hash using the given [hash.Hash] type and key.
func New[H hash.Hash](h func() H, key []byte) *HMAC {
	hm := &HMAC{keyLen: len(key)}
	hm.outer = h()
	hm.inner = h()
	unique := true
	func() {
		defer func() {
			// The comparison might panic if the underlying types are not comparable.
			_ = recover()
		}()
		if hm.outer == hm.inner {
			unique = false
		}
	}()
	if !unique {
		panic("crypto/hmac: hash generation function does not produce unique values")
	}
	blocksize := hm.inner.BlockSize()
	hm.ipad = make([]byte, blocksize)
	hm.opad = make([]byte, blocksize)
	if len(key) > blocksize {
		// If key is too big, hash it.
		hm.outer.Write(key)
		key = hm.outer.Sum(nil)
	}
	copy(hm.ipad, key)
	copy(hm.opad, key)
	for i := range hm.ipad {
		hm.ipad[i] ^= 0x36
	}
	for i := range hm.opad {
		hm.opad[i] ^= 0x5c
	}
	hm.inner.Write(hm.ipad)

	return hm
}

// MarkAsUsedInKDF records that this HMAC instance is used as part of a KDF.
func MarkAsUsedInKDF(h *HMAC) {
	h.forHKDF = true
}
