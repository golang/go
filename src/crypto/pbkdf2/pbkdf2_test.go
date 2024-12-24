// Copyright 2012 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

package pbkdf2_test

import (
	"bytes"
	"crypto/internal/boring"
	"crypto/internal/fips140"
	"crypto/pbkdf2"
	"crypto/sha1"
	"crypto/sha256"
	"hash"
	"testing"
)

type testVector struct {
	password string
	salt     string
	iter     int
	output   []byte
}

// Test vectors from RFC 6070, http://tools.ietf.org/html/rfc6070
var sha1TestVectors = []testVector{
	{
		"password",
		"salt",
		1,
		[]byte{
			0x0c, 0x60, 0xc8, 0x0f, 0x96, 0x1f, 0x0e, 0x71,
			0xf3, 0xa9, 0xb5, 0x24, 0xaf, 0x60, 0x12, 0x06,
			0x2f, 0xe0, 0x37, 0xa6,
		},
	},
	{
		"password",
		"salt",
		2,
		[]byte{
			0xea, 0x6c, 0x01, 0x4d, 0xc7, 0x2d, 0x6f, 0x8c,
			0xcd, 0x1e, 0xd9, 0x2a, 0xce, 0x1d, 0x41, 0xf0,
			0xd8, 0xde, 0x89, 0x57,
		},
	},
	{
		"password",
		"salt",
		4096,
		[]byte{
			0x4b, 0x00, 0x79, 0x01, 0xb7, 0x65, 0x48, 0x9a,
			0xbe, 0xad, 0x49, 0xd9, 0x26, 0xf7, 0x21, 0xd0,
			0x65, 0xa4, 0x29, 0xc1,
		},
	},
	// // This one takes too long
	// {
	// 	"password",
	// 	"salt",
	// 	16777216,
	// 	[]byte{
	// 		0xee, 0xfe, 0x3d, 0x61, 0xcd, 0x4d, 0xa4, 0xe4,
	// 		0xe9, 0x94, 0x5b, 0x3d, 0x6b, 0xa2, 0x15, 0x8c,
	// 		0x26, 0x34, 0xe9, 0x84,
	// 	},
	// },
	{
		"passwordPASSWORDpassword",
		"saltSALTsaltSALTsaltSALTsaltSALTsalt",
		4096,
		[]byte{
			0x3d, 0x2e, 0xec, 0x4f, 0xe4, 0x1c, 0x84, 0x9b,
			0x80, 0xc8, 0xd8, 0x36, 0x62, 0xc0, 0xe4, 0x4a,
			0x8b, 0x29, 0x1a, 0x96, 0x4c, 0xf2, 0xf0, 0x70,
			0x38,
		},
	},
	{
		"pass\000word",
		"sa\000lt",
		4096,
		[]byte{
			0x56, 0xfa, 0x6a, 0xa7, 0x55, 0x48, 0x09, 0x9d,
			0xcc, 0x37, 0xd7, 0xf0, 0x34, 0x25, 0xe0, 0xc3,
		},
	},
}

// Test vectors from
// http://stackoverflow.com/questions/5130513/pbkdf2-hmac-sha2-test-vectors
var sha256TestVectors = []testVector{
	{
		"password",
		"salt",
		1,
		[]byte{
			0x12, 0x0f, 0xb6, 0xcf, 0xfc, 0xf8, 0xb3, 0x2c,
			0x43, 0xe7, 0x22, 0x52, 0x56, 0xc4, 0xf8, 0x37,
			0xa8, 0x65, 0x48, 0xc9,
		},
	},
	{
		"password",
		"salt",
		2,
		[]byte{
			0xae, 0x4d, 0x0c, 0x95, 0xaf, 0x6b, 0x46, 0xd3,
			0x2d, 0x0a, 0xdf, 0xf9, 0x28, 0xf0, 0x6d, 0xd0,
			0x2a, 0x30, 0x3f, 0x8e,
		},
	},
	{
		"password",
		"salt",
		4096,
		[]byte{
			0xc5, 0xe4, 0x78, 0xd5, 0x92, 0x88, 0xc8, 0x41,
			0xaa, 0x53, 0x0d, 0xb6, 0x84, 0x5c, 0x4c, 0x8d,
			0x96, 0x28, 0x93, 0xa0,
		},
	},
	{
		"passwordPASSWORDpassword",
		"saltSALTsaltSALTsaltSALTsaltSALTsalt",
		4096,
		[]byte{
			0x34, 0x8c, 0x89, 0xdb, 0xcb, 0xd3, 0x2b, 0x2f,
			0x32, 0xd8, 0x14, 0xb8, 0x11, 0x6e, 0x84, 0xcf,
			0x2b, 0x17, 0x34, 0x7e, 0xbc, 0x18, 0x00, 0x18,
			0x1c,
		},
	},
	{
		"pass\000word",
		"sa\000lt",
		4096,
		[]byte{
			0x89, 0xb6, 0x9d, 0x05, 0x16, 0xf8, 0x29, 0x89,
			0x3c, 0x69, 0x62, 0x26, 0x65, 0x0a, 0x86, 0x87,
		},
	},
}

func testHash(t *testing.T, h func() hash.Hash, hashName string, vectors []testVector) {
	for i, v := range vectors {
		o, err := pbkdf2.Key(h, v.password, []byte(v.salt), v.iter, len(v.output))
		if err != nil {
			t.Error(err)
		}
		if !bytes.Equal(o, v.output) {
			t.Errorf("%s %d: expected %x, got %x", hashName, i, v.output, o)
		}
	}
}

func TestWithHMACSHA1(t *testing.T) {
	testHash(t, sha1.New, "SHA1", sha1TestVectors)
}

func TestWithHMACSHA256(t *testing.T) {
	testHash(t, sha256.New, "SHA256", sha256TestVectors)
}

var sink uint8

func benchmark(b *testing.B, h func() hash.Hash) {
	var err error
	password := make([]byte, h().Size())
	salt := make([]byte, 8)
	for i := 0; i < b.N; i++ {
		password, err = pbkdf2.Key(h, string(password), salt, 4096, len(password))
		if err != nil {
			b.Error(err)
		}
	}
	sink += password[0]
}

func BenchmarkHMACSHA1(b *testing.B) {
	benchmark(b, sha1.New)
}

func BenchmarkHMACSHA256(b *testing.B) {
	benchmark(b, sha256.New)
}

func TestPBKDF2ServiceIndicator(t *testing.T) {
	if boring.Enabled {
		t.Skip("in BoringCrypto mode PBKDF2 is not from the Go FIPS module")
	}

	goodSalt := []byte{0x01, 0x02, 0x03, 0x04, 0x05, 0x06, 0x07, 0x08, 0x09, 0x0A, 0x0B, 0x0C, 0x0D, 0x0E, 0x0F, 0x10}

	fips140.ResetServiceIndicator()
	_, err := pbkdf2.Key(sha256.New, "password", goodSalt, 1, 32)
	if err != nil {
		t.Error(err)
	}
	if !fips140.ServiceIndicator() {
		t.Error("FIPS service indicator should be set")
	}

	// Salt too short
	fips140.ResetServiceIndicator()
	_, err = pbkdf2.Key(sha256.New, "password", goodSalt[:8], 1, 32)
	if err != nil {
		t.Error(err)
	}
	if fips140.ServiceIndicator() {
		t.Error("FIPS service indicator should not be set")
	}

	// Key length too short
	fips140.ResetServiceIndicator()
	_, err = pbkdf2.Key(sha256.New, "password", goodSalt, 1, 10)
	if err != nil {
		t.Error(err)
	}
	if fips140.ServiceIndicator() {
		t.Error("FIPS service indicator should not be set")
	}
}
