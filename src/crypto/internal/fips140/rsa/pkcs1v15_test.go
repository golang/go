// Copyright 2024 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

package rsa

import (
	"bytes"
	"crypto"
	"crypto/x509/pkix"
	"encoding/asn1"
	"testing"
)

func TestHashPrefixes(t *testing.T) {
	prefixes := map[crypto.Hash]asn1.ObjectIdentifier{
		// RFC 3370, Section 2.1 and 2.2
		//
		// sha-1 OBJECT IDENTIFIER ::= { iso(1) identified-organization(3)
		//      oiw(14) secsig(3) algorithm(2) 26 }
		//
		// md5 OBJECT IDENTIFIER ::= { iso(1) member-body(2) us(840)
		// 	rsadsi(113549) digestAlgorithm(2) 5 }
		crypto.MD5:  {1, 2, 840, 113549, 2, 5},
		crypto.SHA1: {1, 3, 14, 3, 2, 26},

		// https://csrc.nist.gov/projects/computer-security-objects-register/algorithm-registration
		//
		// nistAlgorithms OBJECT IDENTIFIER ::= { joint-iso-ccitt(2) country(16) us(840)
		//          organization(1) gov(101) csor(3) nistAlgorithm(4) }
		//
		// hashAlgs OBJECT IDENTIFIER ::= { nistAlgorithms 2 }
		//
		// id-sha256 OBJECT IDENTIFIER ::= { hashAlgs 1 }
		// id-sha384 OBJECT IDENTIFIER ::= { hashAlgs 2 }
		// id-sha512 OBJECT IDENTIFIER ::= { hashAlgs 3 }
		// id-sha224 OBJECT IDENTIFIER ::= { hashAlgs 4 }
		// id-sha512-224 OBJECT IDENTIFIER ::= { hashAlgs 5 }
		// id-sha512-256 OBJECT IDENTIFIER ::= { hashAlgs 6 }
		// id-sha3-224 OBJECT IDENTIFIER ::= { hashAlgs 7 }
		// id-sha3-256 OBJECT IDENTIFIER ::= { hashAlgs 8 }
		// id-sha3-384 OBJECT IDENTIFIER ::= { hashAlgs 9 }
		// id-sha3-512 OBJECT IDENTIFIER ::= { hashAlgs 10 }
		crypto.SHA224:     {2, 16, 840, 1, 101, 3, 4, 2, 4},
		crypto.SHA256:     {2, 16, 840, 1, 101, 3, 4, 2, 1},
		crypto.SHA384:     {2, 16, 840, 1, 101, 3, 4, 2, 2},
		crypto.SHA512:     {2, 16, 840, 1, 101, 3, 4, 2, 3},
		crypto.SHA512_224: {2, 16, 840, 1, 101, 3, 4, 2, 5},
		crypto.SHA512_256: {2, 16, 840, 1, 101, 3, 4, 2, 6},
		crypto.SHA3_224:   {2, 16, 840, 1, 101, 3, 4, 2, 7},
		crypto.SHA3_256:   {2, 16, 840, 1, 101, 3, 4, 2, 8},
		crypto.SHA3_384:   {2, 16, 840, 1, 101, 3, 4, 2, 9},
		crypto.SHA3_512:   {2, 16, 840, 1, 101, 3, 4, 2, 10},
	}

	for h, oid := range prefixes {
		want, err := asn1.Marshal(struct {
			HashAlgorithm pkix.AlgorithmIdentifier
			Hash          []byte
		}{
			HashAlgorithm: pkix.AlgorithmIdentifier{
				Algorithm:  oid,
				Parameters: asn1.NullRawValue,
			},
			Hash: make([]byte, h.Size()),
		})
		if err != nil {
			t.Fatal(err)
		}
		want = want[:len(want)-h.Size()]
		got := hashPrefixes[h.String()]
		if !bytes.Equal(got, want) {
			t.Errorf("%s: got %x, want %x", h, got, want)
		}
	}
}
