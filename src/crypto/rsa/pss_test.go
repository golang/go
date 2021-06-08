// Copyright 2013 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

package rsa_test

import (
	"bufio"
	"bytes"
	"compress/bzip2"
	"crypto"
	"crypto/rand"
	. "crypto/rsa"
	"crypto/sha1"
	"crypto/sha256"
	"encoding/hex"
	"math/big"
	"os"
	"strconv"
	"strings"
	"testing"
)

func TestEMSAPSS(t *testing.T) {
	// Test vector in file pss-int.txt from: ftp://ftp.rsasecurity.com/pub/pkcs/pkcs-1/pkcs-1v2-1-vec.zip
	msg := []byte{
		0x85, 0x9e, 0xef, 0x2f, 0xd7, 0x8a, 0xca, 0x00, 0x30, 0x8b,
		0xdc, 0x47, 0x11, 0x93, 0xbf, 0x55, 0xbf, 0x9d, 0x78, 0xdb,
		0x8f, 0x8a, 0x67, 0x2b, 0x48, 0x46, 0x34, 0xf3, 0xc9, 0xc2,
		0x6e, 0x64, 0x78, 0xae, 0x10, 0x26, 0x0f, 0xe0, 0xdd, 0x8c,
		0x08, 0x2e, 0x53, 0xa5, 0x29, 0x3a, 0xf2, 0x17, 0x3c, 0xd5,
		0x0c, 0x6d, 0x5d, 0x35, 0x4f, 0xeb, 0xf7, 0x8b, 0x26, 0x02,
		0x1c, 0x25, 0xc0, 0x27, 0x12, 0xe7, 0x8c, 0xd4, 0x69, 0x4c,
		0x9f, 0x46, 0x97, 0x77, 0xe4, 0x51, 0xe7, 0xf8, 0xe9, 0xe0,
		0x4c, 0xd3, 0x73, 0x9c, 0x6b, 0xbf, 0xed, 0xae, 0x48, 0x7f,
		0xb5, 0x56, 0x44, 0xe9, 0xca, 0x74, 0xff, 0x77, 0xa5, 0x3c,
		0xb7, 0x29, 0x80, 0x2f, 0x6e, 0xd4, 0xa5, 0xff, 0xa8, 0xba,
		0x15, 0x98, 0x90, 0xfc,
	}
	salt := []byte{
		0xe3, 0xb5, 0xd5, 0xd0, 0x02, 0xc1, 0xbc, 0xe5, 0x0c, 0x2b,
		0x65, 0xef, 0x88, 0xa1, 0x88, 0xd8, 0x3b, 0xce, 0x7e, 0x61,
	}
	expected := []byte{
		0x66, 0xe4, 0x67, 0x2e, 0x83, 0x6a, 0xd1, 0x21, 0xba, 0x24,
		0x4b, 0xed, 0x65, 0x76, 0xb8, 0x67, 0xd9, 0xa4, 0x47, 0xc2,
		0x8a, 0x6e, 0x66, 0xa5, 0xb8, 0x7d, 0xee, 0x7f, 0xbc, 0x7e,
		0x65, 0xaf, 0x50, 0x57, 0xf8, 0x6f, 0xae, 0x89, 0x84, 0xd9,
		0xba, 0x7f, 0x96, 0x9a, 0xd6, 0xfe, 0x02, 0xa4, 0xd7, 0x5f,
		0x74, 0x45, 0xfe, 0xfd, 0xd8, 0x5b, 0x6d, 0x3a, 0x47, 0x7c,
		0x28, 0xd2, 0x4b, 0xa1, 0xe3, 0x75, 0x6f, 0x79, 0x2d, 0xd1,
		0xdc, 0xe8, 0xca, 0x94, 0x44, 0x0e, 0xcb, 0x52, 0x79, 0xec,
		0xd3, 0x18, 0x3a, 0x31, 0x1f, 0xc8, 0x96, 0xda, 0x1c, 0xb3,
		0x93, 0x11, 0xaf, 0x37, 0xea, 0x4a, 0x75, 0xe2, 0x4b, 0xdb,
		0xfd, 0x5c, 0x1d, 0xa0, 0xde, 0x7c, 0xec, 0xdf, 0x1a, 0x89,
		0x6f, 0x9d, 0x8b, 0xc8, 0x16, 0xd9, 0x7c, 0xd7, 0xa2, 0xc4,
		0x3b, 0xad, 0x54, 0x6f, 0xbe, 0x8c, 0xfe, 0xbc,
	}

	hash := sha1.New()
	hash.Write(msg)
	hashed := hash.Sum(nil)

	encoded, err := EMSAPSSEncode(hashed, 1023, salt, sha1.New())
	if err != nil {
		t.Errorf("Error from emsaPSSEncode: %s\n", err)
	}
	if !bytes.Equal(encoded, expected) {
		t.Errorf("Bad encoding. got %x, want %x", encoded, expected)
	}

	if err = EMSAPSSVerify(hashed, encoded, 1023, len(salt), sha1.New()); err != nil {
		t.Errorf("Bad verification: %s", err)
	}
}

// TestPSSGolden tests all the test vectors in pss-vect.txt from
// ftp://ftp.rsasecurity.com/pub/pkcs/pkcs-1/pkcs-1v2-1-vec.zip
func TestPSSGolden(t *testing.T) {
	inFile, err := os.Open("testdata/pss-vect.txt.bz2")
	if err != nil {
		t.Fatalf("Failed to open input file: %s", err)
	}
	defer inFile.Close()

	// The pss-vect.txt file contains RSA keys and then a series of
	// signatures. A goroutine is used to preprocess the input by merging
	// lines, removing spaces in hex values and identifying the start of
	// new keys and signature blocks.
	const newKeyMarker = "START NEW KEY"
	const newSignatureMarker = "START NEW SIGNATURE"

	values := make(chan string)

	go func() {
		defer close(values)
		scanner := bufio.NewScanner(bzip2.NewReader(inFile))
		var partialValue string
		lastWasValue := true

		for scanner.Scan() {
			line := scanner.Text()
			switch {
			case len(line) == 0:
				if len(partialValue) > 0 {
					values <- strings.ReplaceAll(partialValue, " ", "")
					partialValue = ""
					lastWasValue = true
				}
				continue
			case strings.HasPrefix(line, "# ======") && lastWasValue:
				values <- newKeyMarker
				lastWasValue = false
			case strings.HasPrefix(line, "# ------") && lastWasValue:
				values <- newSignatureMarker
				lastWasValue = false
			case strings.HasPrefix(line, "#"):
				continue
			default:
				partialValue += line
			}
		}
		if err := scanner.Err(); err != nil {
			panic(err)
		}
	}()

	var key *PublicKey
	var hashed []byte
	hash := crypto.SHA1
	h := hash.New()
	opts := &PSSOptions{
		SaltLength: PSSSaltLengthEqualsHash,
	}

	for marker := range values {
		switch marker {
		case newKeyMarker:
			key = new(PublicKey)
			nHex, ok := <-values
			if !ok {
				continue
			}
			key.N = bigFromHex(nHex)
			key.E = intFromHex(<-values)
			// We don't care for d, p, q, dP, dQ or qInv.
			for i := 0; i < 6; i++ {
				<-values
			}
		case newSignatureMarker:
			msg := fromHex(<-values)
			<-values // skip salt
			sig := fromHex(<-values)

			h.Reset()
			h.Write(msg)
			hashed = h.Sum(hashed[:0])

			if err := VerifyPSS(key, hash, hashed, sig, opts); err != nil {
				t.Error(err)
			}
		default:
			t.Fatalf("unknown marker: " + marker)
		}
	}
}

// TestPSSOpenSSL ensures that we can verify a PSS signature from OpenSSL with
// the default options. OpenSSL sets the salt length to be maximal.
func TestPSSOpenSSL(t *testing.T) {
	hash := crypto.SHA256
	h := hash.New()
	h.Write([]byte("testing"))
	hashed := h.Sum(nil)

	// Generated with `echo -n testing | openssl dgst -sign key.pem -sigopt rsa_padding_mode:pss -sha256 > sig`
	sig := []byte{
		0x95, 0x59, 0x6f, 0xd3, 0x10, 0xa2, 0xe7, 0xa2, 0x92, 0x9d,
		0x4a, 0x07, 0x2e, 0x2b, 0x27, 0xcc, 0x06, 0xc2, 0x87, 0x2c,
		0x52, 0xf0, 0x4a, 0xcc, 0x05, 0x94, 0xf2, 0xc3, 0x2e, 0x20,
		0xd7, 0x3e, 0x66, 0x62, 0xb5, 0x95, 0x2b, 0xa3, 0x93, 0x9a,
		0x66, 0x64, 0x25, 0xe0, 0x74, 0x66, 0x8c, 0x3e, 0x92, 0xeb,
		0xc6, 0xe6, 0xc0, 0x44, 0xf3, 0xb4, 0xb4, 0x2e, 0x8c, 0x66,
		0x0a, 0x37, 0x9c, 0x69,
	}

	if err := VerifyPSS(&rsaPrivateKey.PublicKey, hash, hashed, sig, nil); err != nil {
		t.Error(err)
	}
}

func TestPSSNilOpts(t *testing.T) {
	hash := crypto.SHA256
	h := hash.New()
	h.Write([]byte("testing"))
	hashed := h.Sum(nil)

	SignPSS(rand.Reader, rsaPrivateKey, hash, hashed, nil)
}

func TestPSSSigning(t *testing.T) {
	var saltLengthCombinations = []struct {
		signSaltLength, verifySaltLength int
		good                             bool
	}{
		{PSSSaltLengthAuto, PSSSaltLengthAuto, true},
		{PSSSaltLengthEqualsHash, PSSSaltLengthAuto, true},
		{PSSSaltLengthEqualsHash, PSSSaltLengthEqualsHash, true},
		{PSSSaltLengthEqualsHash, 8, false},
		{PSSSaltLengthAuto, PSSSaltLengthEqualsHash, false},
		{8, 8, true},
		{PSSSaltLengthAuto, 42, true},
		{PSSSaltLengthAuto, 20, false},
		{PSSSaltLengthAuto, -2, false},
	}

	hash := crypto.SHA1
	h := hash.New()
	h.Write([]byte("testing"))
	hashed := h.Sum(nil)
	var opts PSSOptions

	for i, test := range saltLengthCombinations {
		opts.SaltLength = test.signSaltLength
		sig, err := SignPSS(rand.Reader, rsaPrivateKey, hash, hashed, &opts)
		if err != nil {
			t.Errorf("#%d: error while signing: %s", i, err)
			continue
		}

		opts.SaltLength = test.verifySaltLength
		err = VerifyPSS(&rsaPrivateKey.PublicKey, hash, hashed, sig, &opts)
		if (err == nil) != test.good {
			t.Errorf("#%d: bad result, wanted: %t, got: %s", i, test.good, err)
		}
	}
}

func TestPSS513(t *testing.T) {
	// See Issue 42741, and separately, RFC 8017: "Note that the octet length of
	// EM will be one less than k if modBits - 1 is divisible by 8 and equal to
	// k otherwise, where k is the length in octets of the RSA modulus n."
	key, err := GenerateKey(rand.Reader, 513)
	if err != nil {
		t.Fatal(err)
	}
	digest := sha256.Sum256([]byte("message"))
	signature, err := key.Sign(rand.Reader, digest[:], &PSSOptions{
		SaltLength: PSSSaltLengthAuto,
		Hash:       crypto.SHA256,
	})
	if err != nil {
		t.Fatal(err)
	}
	err = VerifyPSS(&key.PublicKey, crypto.SHA256, digest[:], signature, nil)
	if err != nil {
		t.Error(err)
	}
}

func bigFromHex(hex string) *big.Int {
	n, ok := new(big.Int).SetString(hex, 16)
	if !ok {
		panic("bad hex: " + hex)
	}
	return n
}

func intFromHex(hex string) int {
	i, err := strconv.ParseInt(hex, 16, 32)
	if err != nil {
		panic(err)
	}
	return int(i)
}

func fromHex(hexStr string) []byte {
	s, err := hex.DecodeString(hexStr)
	if err != nil {
		panic(err)
	}
	return s
}

func TestInvalidPSSSaltLength(t *testing.T) {
	key, err := GenerateKey(rand.Reader, 245)
	if err != nil {
		t.Fatal(err)
	}

	digest := sha256.Sum256([]byte("message"))
	// We don't check the exact error matches, because crypto/rsa and crypto/internal/boring
	// return two different error variables, which have the same content but are not equal.
	if _, err := SignPSS(rand.Reader, key, crypto.SHA256, digest[:], &PSSOptions{
		SaltLength: -2,
		Hash:       crypto.SHA256,
	}); err.Error() != InvalidSaltLenErr.Error() {
		t.Fatalf("SignPSS unexpected error: got %v, want %v", err, InvalidSaltLenErr)
	}

	// We don't check the specific error here, because crypto/rsa and crypto/internal/boring
	// return different errors, so we just check that _an error_ was returned.
	if err := VerifyPSS(&key.PublicKey, crypto.SHA256, []byte{1, 2, 3}, make([]byte, 31), &PSSOptions{
		SaltLength: -2,
	}); err == nil {
		t.Fatal("VerifyPSS unexpected success")
	}
}
