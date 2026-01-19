// Copyright 2013 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

package rsa_test

import (
	"bufio"
	"compress/bzip2"
	"crypto"
	"crypto/internal/fips140"
	"crypto/rand"
	. "crypto/rsa"
	"crypto/sha256"
	"crypto/sha512"
	"encoding/hex"
	"math/big"
	"os"
	"strconv"
	"strings"
	"testing"
)

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
			t.Fatalf("unknown marker: %s", marker)
		}
	}
}

// TestPSSOpenSSL ensures that we can verify a PSS signature from OpenSSL with
// the default options. OpenSSL sets the salt length to be maximal.
func TestPSSOpenSSL(t *testing.T) {
	t.Setenv("GODEBUG", "rsa1024min=0")

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

	if err := VerifyPSS(&test512Key.PublicKey, hash, hashed, sig, nil); err != nil {
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
		good, fipsGood                   bool
	}{
		{PSSSaltLengthAuto, PSSSaltLengthAuto, true, true},
		{PSSSaltLengthEqualsHash, PSSSaltLengthAuto, true, true},
		{PSSSaltLengthEqualsHash, PSSSaltLengthEqualsHash, true, true},
		{PSSSaltLengthEqualsHash, 8, false, false},
		{8, 8, true, true},
		{8, PSSSaltLengthAuto, true, true},
		{42, PSSSaltLengthAuto, true, true},
		// In FIPS mode, PSSSaltLengthAuto is capped at PSSSaltLengthEqualsHash.
		{PSSSaltLengthAuto, PSSSaltLengthEqualsHash, false, true},
		{PSSSaltLengthAuto, 106, true, false},
		{PSSSaltLengthAuto, 20, false, true},
		{PSSSaltLengthAuto, -2, false, false},
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
		good := test.good
		if fips140.Enabled {
			good = test.fipsGood
		}
		if (err == nil) != good {
			t.Errorf("#%d: bad result, wanted: %t, got: %s", i, test.good, err)
		}
	}
}

func TestPSS513(t *testing.T) {
	// See Issue 42741, and separately, RFC 8017: "Note that the octet length of
	// EM will be one less than k if modBits - 1 is divisible by 8 and equal to
	// k otherwise, where k is the length in octets of the RSA modulus n."
	t.Setenv("GODEBUG", "rsa1024min=0")
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
	t.Setenv("GODEBUG", "rsa1024min=0")
	key, err := GenerateKey(rand.Reader, 245)
	if err != nil {
		t.Fatal(err)
	}

	digest := sha256.Sum256([]byte("message"))
	if _, err := SignPSS(rand.Reader, key, crypto.SHA256, digest[:], &PSSOptions{
		SaltLength: -2,
		Hash:       crypto.SHA256,
	}); err.Error() != "crypto/rsa: invalid PSS salt length" {
		t.Fatalf("SignPSS unexpected error: got %v, want %v", err, "crypto/rsa: invalid PSS salt length")
	}

	// We don't check the specific error here, because crypto/rsa and crypto/internal/boring
	// return different errors, so we just check that _an error_ was returned.
	if err := VerifyPSS(&key.PublicKey, crypto.SHA256, []byte{1, 2, 3}, make([]byte, 31), &PSSOptions{
		SaltLength: -2,
	}); err == nil {
		t.Fatal("VerifyPSS unexpected success")
	}
}

func TestHashOverride(t *testing.T) {
	digest := sha512.Sum512([]byte("message"))
	// opts.Hash overrides the passed hash argument.
	sig, err := SignPSS(rand.Reader, test2048Key, crypto.SHA256, digest[:], &PSSOptions{Hash: crypto.SHA512})
	if err != nil {
		t.Fatalf("SignPSS unexpected error: got %v, want nil", err)
	}

	// VerifyPSS has the inverse behavior, opts.Hash is always ignored, check this is true.
	if err := VerifyPSS(&test2048Key.PublicKey, crypto.SHA512, digest[:], sig, &PSSOptions{Hash: crypto.SHA256}); err != nil {
		t.Fatalf("VerifyPSS unexpected error: got %v, want nil", err)
	}
}
