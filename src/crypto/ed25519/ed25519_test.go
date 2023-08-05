// Copyright 2016 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

package ed25519

import (
	"bufio"
	"bytes"
	"compress/gzip"
	"crypto"
	"crypto/internal/boring"
	"crypto/rand"
	"crypto/sha512"
	"encoding/hex"
	"internal/testenv"
	"log"
	"os"
	"strings"
	"testing"
)

func Example_ed25519ctx() {
	pub, priv, err := GenerateKey(nil)
	if err != nil {
		log.Fatal(err)
	}

	msg := []byte("The quick brown fox jumps over the lazy dog")

	sig, err := priv.Sign(nil, msg, &Options{
		Context: "Example_ed25519ctx",
	})
	if err != nil {
		log.Fatal(err)
	}

	if err := VerifyWithOptions(pub, msg, sig, &Options{
		Context: "Example_ed25519ctx",
	}); err != nil {
		log.Fatal("invalid signature")
	}
}

type zeroReader struct{}

func (zeroReader) Read(buf []byte) (int, error) {
	for i := range buf {
		buf[i] = 0
	}
	return len(buf), nil
}

func TestSignVerify(t *testing.T) {
	var zero zeroReader
	public, private, _ := GenerateKey(zero)

	message := []byte("test message")
	sig := Sign(private, message)
	if !Verify(public, message, sig) {
		t.Errorf("valid signature rejected")
	}

	wrongMessage := []byte("wrong message")
	if Verify(public, wrongMessage, sig) {
		t.Errorf("signature of different message accepted")
	}
}

func TestSignVerifyHashed(t *testing.T) {
	// From RFC 8032, Section 7.3
	key, _ := hex.DecodeString("833fe62409237b9d62ec77587520911e9a759cec1d19755b7da901b96dca3d42ec172b93ad5e563bf4932c70e1245034c35467ef2efd4d64ebf819683467e2bf")
	expectedSig, _ := hex.DecodeString("98a70222f0b8121aa9d30f813d683f809e462b469c7ff87639499bb94e6dae4131f85042463c2a355a2003d062adf5aaa10b8c61e636062aaad11c2a26083406")
	message, _ := hex.DecodeString("616263")

	private := PrivateKey(key)
	public := private.Public().(PublicKey)
	hash := sha512.Sum512(message)
	sig, err := private.Sign(nil, hash[:], crypto.SHA512)
	if err != nil {
		t.Fatal(err)
	}
	if !bytes.Equal(sig, expectedSig) {
		t.Error("signature doesn't match test vector")
	}
	sig, err = private.Sign(nil, hash[:], &Options{Hash: crypto.SHA512})
	if err != nil {
		t.Fatal(err)
	}
	if !bytes.Equal(sig, expectedSig) {
		t.Error("signature doesn't match test vector")
	}
	if err := VerifyWithOptions(public, hash[:], sig, &Options{Hash: crypto.SHA512}); err != nil {
		t.Errorf("valid signature rejected: %v", err)
	}

	if err := VerifyWithOptions(public, hash[:], sig, &Options{Hash: crypto.SHA256}); err == nil {
		t.Errorf("expected error for wrong hash")
	}

	wrongHash := sha512.Sum512([]byte("wrong message"))
	if VerifyWithOptions(public, wrongHash[:], sig, &Options{Hash: crypto.SHA512}) == nil {
		t.Errorf("signature of different message accepted")
	}

	sig[0] ^= 0xff
	if VerifyWithOptions(public, hash[:], sig, &Options{Hash: crypto.SHA512}) == nil {
		t.Errorf("invalid signature accepted")
	}
	sig[0] ^= 0xff
	sig[SignatureSize-1] ^= 0xff
	if VerifyWithOptions(public, hash[:], sig, &Options{Hash: crypto.SHA512}) == nil {
		t.Errorf("invalid signature accepted")
	}

	// The RFC provides no test vectors for Ed25519ph with context, so just sign
	// and verify something.
	sig, err = private.Sign(nil, hash[:], &Options{Hash: crypto.SHA512, Context: "123"})
	if err != nil {
		t.Fatal(err)
	}
	if err := VerifyWithOptions(public, hash[:], sig, &Options{Hash: crypto.SHA512, Context: "123"}); err != nil {
		t.Errorf("valid signature rejected: %v", err)
	}
	if err := VerifyWithOptions(public, hash[:], sig, &Options{Hash: crypto.SHA512, Context: "321"}); err == nil {
		t.Errorf("expected error for wrong context")
	}
	if err := VerifyWithOptions(public, hash[:], sig, &Options{Hash: crypto.SHA256, Context: "123"}); err == nil {
		t.Errorf("expected error for wrong hash")
	}
}

func TestSignVerifyContext(t *testing.T) {
	// From RFC 8032, Section 7.2
	key, _ := hex.DecodeString("0305334e381af78f141cb666f6199f57bc3495335a256a95bd2a55bf546663f6dfc9425e4f968f7f0c29f0259cf5f9aed6851c2bb4ad8bfb860cfee0ab248292")
	expectedSig, _ := hex.DecodeString("55a4cc2f70a54e04288c5f4cd1e45a7bb520b36292911876cada7323198dd87a8b36950b95130022907a7fb7c4e9b2d5f6cca685a587b4b21f4b888e4e7edb0d")
	message, _ := hex.DecodeString("f726936d19c800494e3fdaff20b276a8")
	context := "foo"

	private := PrivateKey(key)
	public := private.Public().(PublicKey)
	sig, err := private.Sign(nil, message, &Options{Context: context})
	if err != nil {
		t.Fatal(err)
	}
	if !bytes.Equal(sig, expectedSig) {
		t.Error("signature doesn't match test vector")
	}
	if err := VerifyWithOptions(public, message, sig, &Options{Context: context}); err != nil {
		t.Errorf("valid signature rejected: %v", err)
	}

	if VerifyWithOptions(public, []byte("bar"), sig, &Options{Context: context}) == nil {
		t.Errorf("signature of different message accepted")
	}
	if VerifyWithOptions(public, message, sig, &Options{Context: "bar"}) == nil {
		t.Errorf("signature with different context accepted")
	}

	sig[0] ^= 0xff
	if VerifyWithOptions(public, message, sig, &Options{Context: context}) == nil {
		t.Errorf("invalid signature accepted")
	}
	sig[0] ^= 0xff
	sig[SignatureSize-1] ^= 0xff
	if VerifyWithOptions(public, message, sig, &Options{Context: context}) == nil {
		t.Errorf("invalid signature accepted")
	}
}

func TestCryptoSigner(t *testing.T) {
	var zero zeroReader
	public, private, _ := GenerateKey(zero)

	signer := crypto.Signer(private)

	publicInterface := signer.Public()
	public2, ok := publicInterface.(PublicKey)
	if !ok {
		t.Fatalf("expected PublicKey from Public() but got %T", publicInterface)
	}

	if !bytes.Equal(public, public2) {
		t.Errorf("public keys do not match: original:%x vs Public():%x", public, public2)
	}

	message := []byte("message")
	var noHash crypto.Hash
	signature, err := signer.Sign(zero, message, noHash)
	if err != nil {
		t.Fatalf("error from Sign(): %s", err)
	}

	signature2, err := signer.Sign(zero, message, &Options{Hash: noHash})
	if err != nil {
		t.Fatalf("error from Sign(): %s", err)
	}
	if !bytes.Equal(signature, signature2) {
		t.Errorf("signatures keys do not match")
	}

	if !Verify(public, message, signature) {
		t.Errorf("Verify failed on signature from Sign()")
	}
}

func TestEqual(t *testing.T) {
	public, private, _ := GenerateKey(rand.Reader)

	if !public.Equal(public) {
		t.Errorf("public key is not equal to itself: %q", public)
	}
	if !public.Equal(crypto.Signer(private).Public()) {
		t.Errorf("private.Public() is not Equal to public: %q", public)
	}
	if !private.Equal(private) {
		t.Errorf("private key is not equal to itself: %q", private)
	}

	otherPub, otherPriv, _ := GenerateKey(rand.Reader)
	if public.Equal(otherPub) {
		t.Errorf("different public keys are Equal")
	}
	if private.Equal(otherPriv) {
		t.Errorf("different private keys are Equal")
	}
}

func TestGolden(t *testing.T) {
	// sign.input.gz is a selection of test cases from
	// https://ed25519.cr.yp.to/python/sign.input
	testDataZ, err := os.Open("testdata/sign.input.gz")
	if err != nil {
		t.Fatal(err)
	}
	defer testDataZ.Close()
	testData, err := gzip.NewReader(testDataZ)
	if err != nil {
		t.Fatal(err)
	}
	defer testData.Close()

	scanner := bufio.NewScanner(testData)
	lineNo := 0

	for scanner.Scan() {
		lineNo++

		line := scanner.Text()
		parts := strings.Split(line, ":")
		if len(parts) != 5 {
			t.Fatalf("bad number of parts on line %d", lineNo)
		}

		privBytes, _ := hex.DecodeString(parts[0])
		pubKey, _ := hex.DecodeString(parts[1])
		msg, _ := hex.DecodeString(parts[2])
		sig, _ := hex.DecodeString(parts[3])
		// The signatures in the test vectors also include the message
		// at the end, but we just want R and S.
		sig = sig[:SignatureSize]

		if l := len(pubKey); l != PublicKeySize {
			t.Fatalf("bad public key length on line %d: got %d bytes", lineNo, l)
		}

		var priv [PrivateKeySize]byte
		copy(priv[:], privBytes)
		copy(priv[32:], pubKey)

		sig2 := Sign(priv[:], msg)
		if !bytes.Equal(sig, sig2[:]) {
			t.Errorf("different signature result on line %d: %x vs %x", lineNo, sig, sig2)
		}

		if !Verify(pubKey, msg, sig2) {
			t.Errorf("signature failed to verify on line %d", lineNo)
		}

		priv2 := NewKeyFromSeed(priv[:32])
		if !bytes.Equal(priv[:], priv2) {
			t.Errorf("recreating key pair gave different private key on line %d: %x vs %x", lineNo, priv[:], priv2)
		}

		if pubKey2 := priv2.Public().(PublicKey); !bytes.Equal(pubKey, pubKey2) {
			t.Errorf("recreating key pair gave different public key on line %d: %x vs %x", lineNo, pubKey, pubKey2)
		}

		if seed := priv2.Seed(); !bytes.Equal(priv[:32], seed) {
			t.Errorf("recreating key pair gave different seed on line %d: %x vs %x", lineNo, priv[:32], seed)
		}
	}

	if err := scanner.Err(); err != nil {
		t.Fatalf("error reading test data: %s", err)
	}
}

func TestMalleability(t *testing.T) {
	// https://tools.ietf.org/html/rfc8032#section-5.1.7 adds an additional test
	// that s be in [0, order). This prevents someone from adding a multiple of
	// order to s and obtaining a second valid signature for the same message.
	msg := []byte{0x54, 0x65, 0x73, 0x74}
	sig := []byte{
		0x7c, 0x38, 0xe0, 0x26, 0xf2, 0x9e, 0x14, 0xaa, 0xbd, 0x05, 0x9a,
		0x0f, 0x2d, 0xb8, 0xb0, 0xcd, 0x78, 0x30, 0x40, 0x60, 0x9a, 0x8b,
		0xe6, 0x84, 0xdb, 0x12, 0xf8, 0x2a, 0x27, 0x77, 0x4a, 0xb0, 0x67,
		0x65, 0x4b, 0xce, 0x38, 0x32, 0xc2, 0xd7, 0x6f, 0x8f, 0x6f, 0x5d,
		0xaf, 0xc0, 0x8d, 0x93, 0x39, 0xd4, 0xee, 0xf6, 0x76, 0x57, 0x33,
		0x36, 0xa5, 0xc5, 0x1e, 0xb6, 0xf9, 0x46, 0xb3, 0x1d,
	}
	publicKey := []byte{
		0x7d, 0x4d, 0x0e, 0x7f, 0x61, 0x53, 0xa6, 0x9b, 0x62, 0x42, 0xb5,
		0x22, 0xab, 0xbe, 0xe6, 0x85, 0xfd, 0xa4, 0x42, 0x0f, 0x88, 0x34,
		0xb1, 0x08, 0xc3, 0xbd, 0xae, 0x36, 0x9e, 0xf5, 0x49, 0xfa,
	}

	if Verify(publicKey, msg, sig) {
		t.Fatal("non-canonical signature accepted")
	}
}

func TestAllocations(t *testing.T) {
	if boring.Enabled {
		t.Skip("skipping allocations test with BoringCrypto")
	}
	testenv.SkipIfOptimizationOff(t)

	if allocs := testing.AllocsPerRun(100, func() {
		seed := make([]byte, SeedSize)
		message := []byte("Hello, world!")
		priv := NewKeyFromSeed(seed)
		pub := priv.Public().(PublicKey)
		signature := Sign(priv, message)
		if !Verify(pub, message, signature) {
			t.Fatal("signature didn't verify")
		}
	}); allocs > 0 {
		t.Errorf("expected zero allocations, got %0.1f", allocs)
	}
}

func BenchmarkKeyGeneration(b *testing.B) {
	var zero zeroReader
	for i := 0; i < b.N; i++ {
		if _, _, err := GenerateKey(zero); err != nil {
			b.Fatal(err)
		}
	}
}

func BenchmarkNewKeyFromSeed(b *testing.B) {
	seed := make([]byte, SeedSize)
	for i := 0; i < b.N; i++ {
		_ = NewKeyFromSeed(seed)
	}
}

func BenchmarkSigning(b *testing.B) {
	var zero zeroReader
	_, priv, err := GenerateKey(zero)
	if err != nil {
		b.Fatal(err)
	}
	message := []byte("Hello, world!")
	b.ResetTimer()
	for i := 0; i < b.N; i++ {
		Sign(priv, message)
	}
}

func BenchmarkVerification(b *testing.B) {
	var zero zeroReader
	pub, priv, err := GenerateKey(zero)
	if err != nil {
		b.Fatal(err)
	}
	message := []byte("Hello, world!")
	signature := Sign(priv, message)
	b.ResetTimer()
	for i := 0; i < b.N; i++ {
		Verify(pub, message, signature)
	}
}
