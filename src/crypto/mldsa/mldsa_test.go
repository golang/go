// Copyright 2025 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

//go:build !fips140v1.0

package mldsa_test

import (
	"bytes"
	"crypto"
	"crypto/fips140"
	"crypto/internal/cryptotest"
	. "crypto/mldsa"
	"crypto/sha3"
	"encoding/hex"
	"flag"
	"math/rand/v2"
	"strings"
	"testing"
)

var _ crypto.Signer = (*PrivateKey)(nil)

var sixtyMillionFlag = flag.Bool("60million", false, "run 60M-iterations accumulated test")

// TestAccumulated accumulates 10k (or 100, or 60M) random vectors and checks
// the hash of the result, to avoid checking in megabytes of test vectors.
//
// 60M in particular is enough to give a 99.9% chance of hitting every value in
// the base field.
//
//	1-((q-1)/q)^60000000 ~= 0.9992
//
// If setting -60million, remember to also set -timeout 0.
func TestAccumulated(t *testing.T) {
	t.Run("ML-DSA-44/100", func(t *testing.T) {
		testAccumulated(t, MLDSA44(), 100,
			"d51148e1f9f4fa1a723a6cf42e25f2a99eb5c1b378b3d2dbbd561b1203beeae4")
	})
	t.Run("ML-DSA-65/100", func(t *testing.T) {
		testAccumulated(t, MLDSA65(), 100,
			"8358a1843220194417cadbc2651295cd8fc65125b5a5c1a239a16dc8b57ca199")
	})
	t.Run("ML-DSA-87/100", func(t *testing.T) {
		testAccumulated(t, MLDSA87(), 100,
			"8c3ad714777622b8f21ce31bb35f71394f23bc0fcf3c78ace5d608990f3b061b")
	})
	if !testing.Short() {
		t.Run("ML-DSA-44/10k", func(t *testing.T) {
			t.Parallel()
			testAccumulated(t, MLDSA44(), 10000,
				"e7fd21f6a59bcba60d65adc44404bb29a7c00e5d8d3ec06a732c00a306a7d143")
		})
		t.Run("ML-DSA-65/10k", func(t *testing.T) {
			t.Parallel()
			testAccumulated(t, MLDSA65(), 10000,
				"5ff5e196f0b830c3b10a9eb5358e7c98a3a20136cb677f3ae3b90175c3ace329")
		})
		t.Run("ML-DSA-87/10k", func(t *testing.T) {
			t.Parallel()
			testAccumulated(t, MLDSA87(), 10000,
				"80a8cf39317f7d0be0e24972c51ac152bd2a3e09bc0c32ce29dd82c4e7385e60")
		})
	}
	if *sixtyMillionFlag {
		t.Run("ML-DSA-44/60M", func(t *testing.T) {
			t.Parallel()
			testAccumulated(t, MLDSA44(), 60000000,
				"080b48049257f5cd30dee17d6aa393d6c42fe52a29099df84a460ebaf4b02330")
		})
		t.Run("ML-DSA-65/60M", func(t *testing.T) {
			t.Parallel()
			testAccumulated(t, MLDSA65(), 60000000,
				"0af0165db2b180f7a83dbecad1ccb758b9c2d834b7f801fc49dd572a9d4b1e83")
		})
		t.Run("ML-DSA-87/60M", func(t *testing.T) {
			t.Parallel()
			testAccumulated(t, MLDSA87(), 60000000,
				"011166e9d5032c9bdc5c9bbb5dbb6c86df1c3d9bf3570b65ebae942dd9830057")
		})
	}
}

func testAccumulated(t *testing.T, params Parameters, n int, expected string) {
	s := sha3.NewSHAKE128()
	o := sha3.NewSHAKE128()
	seed := make([]byte, PrivateKeySize)
	msg := make([]byte, 0)

	for i := 0; i < n; i++ {
		s.Read(seed)
		dk, err := NewPrivateKey(params, seed)
		if err != nil {
			t.Fatalf("NewPrivateKey: %v", err)
		}
		pk := dk.PublicKey().Bytes()
		o.Write(pk)
		sig, err := dk.SignDeterministic(msg, nil)
		if err != nil {
			t.Fatalf("SignDeterministic: %v", err)
		}
		o.Write(sig)
		pub, err := NewPublicKey(params, pk)
		if err != nil {
			t.Fatalf("NewPublicKey: %v", err)
		}
		if *pub != *dk.PublicKey() {
			t.Fatalf("public key mismatch")
		}
		if err := Verify(dk.PublicKey(), msg, sig, nil); err != nil {
			t.Fatalf("Verify: %v", err)
		}
	}

	sum := make([]byte, 32)
	o.Read(sum)
	got := hex.EncodeToString(sum)
	if got != expected {
		t.Errorf("got %s, expected %s", got, expected)
	}
}

func testAllParameters(t *testing.T, f func(*testing.T, Parameters)) {
	for _, params := range []Parameters{MLDSA44(), MLDSA65(), MLDSA87()} {
		t.Run(params.String(), func(t *testing.T) {
			f(t, params)
		})
	}
}

func TestGenerateKey(t *testing.T) {
	testAllParameters(t, testGenerateKey)
}

func testGenerateKey(t *testing.T, params Parameters) {
	k1, err := GenerateKey(params)
	if err != nil {
		t.Fatalf("GenerateKey: %v", err)
	}
	k2, err := GenerateKey(params)
	if err != nil {
		t.Fatalf("GenerateKey: %v", err)
	}
	if k1.Equal(k2) {
		t.Errorf("two generated keys are equal")
	}
	k1x, err := NewPrivateKey(params, k1.Bytes())
	if err != nil {
		t.Fatalf("NewPrivateKey: %v", err)
	}
	if !k1.Equal(k1x) {
		t.Errorf("generated key and re-parsed key are not equal")
	}
}

func TestAllocations(t *testing.T) {
	// We allocate
	//
	//   - the PrivateKey (k and kk) structs
	//   - their temporary inner structs (2x)
	//   - the public key (pkBytes) and signature (sig) byte slices
	//   - the Options argument to Sign
	//
	// on the heap. The structs are too large for the stack, the byte slices are
	// variable-sized, and Options is cast into an interface.
	//
	// Still, check we are not slipping more allocations in.
	var expected float64 = 7
	if fips140.Enabled() {
		// The PCT does a sign/verify cycle, which allocates a signature slice.
		expected += 1
	}
	if fips140.Version() == "v1.26.0" {
		// The v1.26.0 implementation precomputes PublicKey, making it large
		// enough to require heap allocation. Add pk, its inner struct, and the
		// return value of k.PublicKey().
		expected += 3
	}
	cryptotest.SkipTestAllocations(t)
	if allocs := testing.AllocsPerRun(100, func() {
		k, err := GenerateKey(MLDSA44())
		if err != nil {
			t.Fatalf("GenerateKey: %v", err)
		}
		seed := k.Bytes()
		kk, err := NewPrivateKey(MLDSA44(), seed)
		if err != nil {
			t.Fatalf("NewPrivateKey: %v", err)
		}
		if !k.Equal(kk) {
			t.Fatalf("keys not equal")
		}
		pkBytes := k.PublicKey().Bytes()
		pk, err := NewPublicKey(MLDSA44(), pkBytes)
		if err != nil {
			t.Fatalf("NewPublicKey: %v", err)
		}
		message := []byte("Hello, world!")
		context := "test"
		sig, err := k.Sign(nil, message, &Options{Context: context})
		if err != nil {
			t.Fatalf("Sign: %v", err)
		}
		if err := Verify(pk, message, sig, &Options{Context: context}); err != nil {
			t.Fatalf("Verify: %v", err)
		}
	}); allocs > expected {
		t.Errorf("expected %0.0f allocations, got %0.1f", expected, allocs)
	}
}

func TestParametersIdentity(t *testing.T) {
	// Per the MLDSA*() docs, repeated calls return the same value, suitable for
	// equality checks and switch statements.
	if MLDSA44() != MLDSA44() || MLDSA65() != MLDSA65() || MLDSA87() != MLDSA87() {
		t.Errorf("MLDSA*() returned different values across calls")
	}
	if MLDSA44() == MLDSA65() || MLDSA65() == MLDSA87() || MLDSA44() == MLDSA87() {
		t.Errorf("distinct parameter sets compare equal")
	}
}

// computeMu reproduces μ = SHAKE256(SHAKE256(pk, 64) || 0x00 || ctxlen || ctx ||
// msg, 64) per FIPS 204, used to drive the External μ signing path.
func computeMu(pk, ctx, msg []byte) []byte {
	tr := sha3.NewSHAKE256()
	tr.Write(pk)
	trOut := make([]byte, 64)
	tr.Read(trOut)

	h := sha3.NewSHAKE256()
	h.Write(trOut)
	h.Write([]byte{0x00, byte(len(ctx))})
	h.Write(ctx)
	h.Write(msg)
	out := make([]byte, 64)
	h.Read(out)
	return out
}

// fakeSignerOpts is a [crypto.SignerOpts] whose [HashFunc] returns h, used to
// exercise the opts-dispatch paths in [PrivateKey.Sign] without going through
// [Options].
type fakeSignerOpts struct{ h crypto.Hash }

func (f fakeSignerOpts) HashFunc() crypto.Hash { return f.h }

func TestSign(t *testing.T) {
	testAllParameters(t, func(t *testing.T, params Parameters) {
		sk, err := GenerateKey(params)
		if err != nil {
			t.Fatalf("GenerateKey: %v", err)
		}
		pk := sk.PublicKey()
		msg := []byte("test message")

		// nil opts and &Options{} must be equivalent (and both interoperable
		// with a nil/zero Verify opts).
		sig1, err := sk.Sign(nil, msg, nil)
		if err != nil {
			t.Fatalf("Sign(nil opts): %v", err)
		}
		if got := len(sig1); got != params.SignatureSize() {
			t.Errorf("len(sig) = %d, want %d", got, params.SignatureSize())
		}
		if err := Verify(pk, msg, sig1, nil); err != nil {
			t.Errorf("Verify of nil-opts signature with nil opts: %v", err)
		}
		if err := Verify(pk, msg, sig1, &Options{}); err != nil {
			t.Errorf("Verify of nil-opts signature with empty Options: %v", err)
		}

		sig2, err := sk.Sign(nil, msg, &Options{})
		if err != nil {
			t.Fatalf("Sign(&Options{}): %v", err)
		}
		if err := Verify(pk, msg, sig2, nil); err != nil {
			t.Errorf("Verify of empty-Options signature with nil opts: %v", err)
		}

		// A non-*Options crypto.SignerOpts whose HashFunc returns 0 must
		// also sign directly, with empty context.
		sig3, err := sk.Sign(nil, msg, fakeSignerOpts{h: 0})
		if err != nil {
			t.Fatalf("Sign(fakeSignerOpts{0}): %v", err)
		}
		if err := Verify(pk, msg, sig3, nil); err != nil {
			t.Errorf("Verify of fake-opts signature: %v", err)
		}

		// crypto.Hash(0) similarly: HashFunc returns 0.
		sig4, err := sk.Sign(nil, msg, crypto.Hash(0))
		if err != nil {
			t.Fatalf("Sign(crypto.Hash(0)): %v", err)
		}
		if err := Verify(pk, msg, sig4, nil); err != nil {
			t.Errorf("Verify of Hash(0)-opts signature: %v", err)
		}

		// A wrong HashFunc must produce errInvalidSignerOpts.
		if _, err := sk.Sign(nil, msg, crypto.SHA256); err == nil {
			t.Errorf("Sign with crypto.SHA256 opts: want error, got nil")
		}
		if _, err := sk.SignDeterministic(msg, crypto.SHA256); err == nil {
			t.Errorf("SignDeterministic with crypto.SHA256 opts: want error, got nil")
		}

		// SignDeterministic with nil and &Options{} must agree byte-for-byte.
		detA, err := sk.SignDeterministic(msg, nil)
		if err != nil {
			t.Fatalf("SignDeterministic(nil): %v", err)
		}
		detB, err := sk.SignDeterministic(msg, &Options{})
		if err != nil {
			t.Fatalf("SignDeterministic(&Options{}): %v", err)
		}
		if !bytes.Equal(detA, detB) {
			t.Errorf("SignDeterministic with nil and &Options{} differ")
		}

		// A different Context produces a different deterministic signature
		// and verification with a mismatched context must fail.
		detCtx, err := sk.SignDeterministic(msg, &Options{Context: "ctx"})
		if err != nil {
			t.Fatalf("SignDeterministic(ctx): %v", err)
		}
		if string(detCtx) == string(detA) {
			t.Errorf("SignDeterministic with empty and non-empty context match")
		}
		if err := Verify(pk, msg, detCtx, nil); err == nil {
			t.Errorf("Verify of context signature with empty context: want error, got nil")
		}
		if err := Verify(pk, msg, detCtx, &Options{Context: "ctx"}); err != nil {
			t.Errorf("Verify with matching context: %v", err)
		}

		// Context >255 bytes is rejected by the underlying implementation.
		longCtx := strings.Repeat("x", 256)
		if _, err := sk.Sign(nil, msg, &Options{Context: longCtx}); err == nil {
			t.Errorf("Sign with 256-byte context: want error, got nil")
		}
		if _, err := sk.SignDeterministic(msg, &Options{Context: longCtx}); err == nil {
			t.Errorf("SignDeterministic with 256-byte context: want error, got nil")
		}
		if err := Verify(pk, msg, detA, &Options{Context: longCtx}); err == nil {
			t.Errorf("Verify with 256-byte context: want error, got nil")
		}

		// Tampered signature must not verify.
		sigTampered := bytes.Clone(sig1)
		sigTampered[len(sigTampered)/2] ^= 0x01
		if err := Verify(pk, msg, sigTampered, nil); err == nil {
			t.Errorf("Verify of tampered signature: want error, got nil")
		}

		// Modified message must not verify against the original signature.
		msgTampered := bytes.Clone(msg)
		msgTampered[0] ^= 0x01
		if err := Verify(pk, msgTampered, sig1, nil); err == nil {
			t.Errorf("Verify of modified message: want error, got nil")
		}

		// Signature from a different key must not verify.
		skOther, err := GenerateKey(params)
		if err != nil {
			t.Fatalf("GenerateKey: %v", err)
		}
		sigOther, err := skOther.SignDeterministic(msg, nil)
		if err != nil {
			t.Fatalf("SignDeterministic: %v", err)
		}
		if err := Verify(pk, msg, sigOther, nil); err == nil {
			t.Errorf("Verify of signature from a different key: want error, got nil")
		}
	})
}

func TestExternalMu(t *testing.T) {
	testAllParameters(t, func(t *testing.T, params Parameters) {
		sk, err := GenerateKey(params)
		if err != nil {
			t.Fatalf("GenerateKey: %v", err)
		}
		pk := sk.PublicKey()
		pkBytes := pk.Bytes()
		msg := []byte("hello mu")

		for _, ctx := range []string{"", "ctx"} {
			μ := computeMu(pkBytes, []byte(ctx), msg)
			sig, err := sk.Sign(nil, μ, crypto.MLDSAMu)
			if err != nil {
				t.Fatalf("Sign(MLDSAMu, ctx=%q): %v", ctx, err)
			}
			if err := Verify(pk, msg, sig, &Options{Context: ctx}); err != nil {
				t.Errorf("Verify of MLDSAMu signature, ctx=%q: %v", ctx, err)
			}

			detSig, err := sk.SignDeterministic(μ, crypto.MLDSAMu)
			if err != nil {
				t.Fatalf("SignDeterministic(MLDSAMu, ctx=%q): %v", ctx, err)
			}
			if err := Verify(pk, msg, detSig, &Options{Context: ctx}); err != nil {
				t.Errorf("Verify of deterministic MLDSAMu signature, ctx=%q: %v", ctx, err)
			}
			detSig2, err := sk.SignDeterministic(μ, crypto.MLDSAMu)
			if err != nil {
				t.Fatalf("SignDeterministic(MLDSAMu) second call: %v", err)
			}
			if string(detSig) != string(detSig2) {
				t.Errorf("SignDeterministic(MLDSAMu) is not deterministic")
			}
		}

		// Cross-context: μ computed under one ctx must not verify under another.
		μA := computeMu(pkBytes, []byte("a"), msg)
		sigA, err := sk.Sign(nil, μA, crypto.MLDSAMu)
		if err != nil {
			t.Fatalf("Sign(MLDSAMu, ctx=a): %v", err)
		}
		if err := Verify(pk, msg, sigA, &Options{Context: "b"}); err == nil {
			t.Errorf("Verify of MLDSAMu(ctx=a) signature with ctx=b: want error, got nil")
		}

		// Tampered MLDSAMu signature must not verify.
		sigTampered := bytes.Clone(sigA)
		sigTampered[len(sigTampered)/2] ^= 0x01
		if err := Verify(pk, msg, sigTampered, &Options{Context: "a"}); err == nil {
			t.Errorf("Verify of tampered MLDSAMu signature: want error, got nil")
		}

		// Wrong-length μ must be rejected.
		if _, err := sk.Sign(nil, make([]byte, 32), crypto.MLDSAMu); err == nil {
			t.Errorf("Sign(MLDSAMu) with 32-byte input: want error, got nil")
		}
		if _, err := sk.SignDeterministic(make([]byte, 32), crypto.MLDSAMu); err == nil {
			t.Errorf("SignDeterministic(MLDSAMu) with 32-byte input: want error, got nil")
		}
	})
}

func TestPublicKey(t *testing.T) {
	cases := []struct {
		params  Parameters
		name    string
		pkSize  int
		sigSize int
	}{
		{MLDSA44(), "ML-DSA-44", MLDSA44PublicKeySize, MLDSA44SignatureSize},
		{MLDSA65(), "ML-DSA-65", MLDSA65PublicKeySize, MLDSA65SignatureSize},
		{MLDSA87(), "ML-DSA-87", MLDSA87PublicKeySize, MLDSA87SignatureSize},
	}
	for _, tc := range cases {
		t.Run(tc.name, func(t *testing.T) {
			if got := tc.params.String(); got != tc.name {
				t.Errorf("Parameters.String() = %q, want %q", got, tc.name)
			}
			if got := tc.params.PublicKeySize(); got != tc.pkSize {
				t.Errorf("Parameters.PublicKeySize() = %d, want %d", got, tc.pkSize)
			}
			if got := tc.params.SignatureSize(); got != tc.sigSize {
				t.Errorf("Parameters.SignatureSize() = %d, want %d", got, tc.sigSize)
			}

			sk, err := GenerateKey(tc.params)
			if err != nil {
				t.Fatalf("GenerateKey: %v", err)
			}
			pk := sk.PublicKey()
			if got := pk.Parameters(); got != tc.params {
				t.Errorf("PublicKey.Parameters() = %v, want %v", got, tc.params)
			}
			if got := len(pk.Bytes()); got != tc.params.PublicKeySize() {
				t.Errorf("len(PublicKey.Bytes()) = %d, want %d", got, tc.params.PublicKeySize())
			}
			if got := len(sk.Bytes()); got != PrivateKeySize {
				t.Errorf("len(PrivateKey.Bytes()) = %d, want %d", got, PrivateKeySize)
			}

			// Public() returns the same key as PublicKey().
			anyPub := sk.Public()
			pub2, ok := anyPub.(*PublicKey)
			if !ok {
				t.Fatalf("PrivateKey.Public() = %T, want *PublicKey", anyPub)
			}
			if !pk.Equal(pub2) {
				t.Errorf("PrivateKey.Public() does not equal PublicKey()")
			}

			// Round-trip via NewPrivateKey/NewPublicKey.
			sk2, err := NewPrivateKey(tc.params, sk.Bytes())
			if err != nil {
				t.Fatalf("NewPrivateKey round-trip: %v", err)
			}
			if !sk.Equal(sk2) {
				t.Errorf("PrivateKey round-trip not equal")
			}
			pk2, err := NewPublicKey(tc.params, pk.Bytes())
			if err != nil {
				t.Fatalf("NewPublicKey round-trip: %v", err)
			}
			if !pk.Equal(pk2) {
				t.Errorf("PublicKey round-trip not equal")
			}
		})
	}
}

func TestEqualWrongType(t *testing.T) {
	sk, err := GenerateKey(MLDSA44())
	if err != nil {
		t.Fatalf("GenerateKey: %v", err)
	}
	if sk.Equal("not a key") {
		t.Errorf("PrivateKey.Equal(string) = true, want false")
	}
	if sk.Equal((*PublicKey)(nil)) {
		t.Errorf("PrivateKey.Equal(*PublicKey) = true, want false")
	}
	if sk.PublicKey().Equal("not a key") {
		t.Errorf("PublicKey.Equal(string) = true, want false")
	}
	if sk.PublicKey().Equal((*PrivateKey)(nil)) {
		t.Errorf("PublicKey.Equal(*PrivateKey) = true, want false")
	}

	// Distinct keys are not Equal.
	sk2, err := GenerateKey(MLDSA44())
	if err != nil {
		t.Fatalf("GenerateKey: %v", err)
	}
	if sk.Equal(sk2) {
		t.Errorf("two random PrivateKeys are Equal")
	}
	if sk.PublicKey().Equal(sk2.PublicKey()) {
		t.Errorf("two random PublicKeys are Equal")
	}
}

func TestInvalidParameters(t *testing.T) {
	var zero Parameters
	if _, err := GenerateKey(zero); err == nil {
		t.Errorf("GenerateKey(zero Parameters): want error, got nil")
	}
	if _, err := NewPrivateKey(zero, make([]byte, PrivateKeySize)); err == nil {
		t.Errorf("NewPrivateKey(zero Parameters): want error, got nil")
	}
	if _, err := NewPublicKey(zero, make([]byte, MLDSA44PublicKeySize)); err == nil {
		t.Errorf("NewPublicKey(zero Parameters): want error, got nil")
	}
}

func TestInvalidSize(t *testing.T) {
	testAllParameters(t, func(t *testing.T, params Parameters) {
		if _, err := NewPrivateKey(params, make([]byte, PrivateKeySize-1)); err == nil {
			t.Errorf("NewPrivateKey with short seed: want error, got nil")
		}
		if _, err := NewPrivateKey(params, make([]byte, PrivateKeySize+1)); err == nil {
			t.Errorf("NewPrivateKey with long seed: want error, got nil")
		}
		if _, err := NewPublicKey(params, make([]byte, params.PublicKeySize()-1)); err == nil {
			t.Errorf("NewPublicKey with short encoding: want error, got nil")
		}
		if _, err := NewPublicKey(params, make([]byte, params.PublicKeySize()+1)); err == nil {
			t.Errorf("NewPublicKey with long encoding: want error, got nil")
		}

		sk, err := GenerateKey(params)
		if err != nil {
			t.Fatalf("GenerateKey: %v", err)
		}
		msg := []byte("test message")
		sig, err := sk.SignDeterministic(msg, nil)
		if err != nil {
			t.Fatalf("SignDeterministic: %v", err)
		}
		if err := Verify(sk.PublicKey(), msg, sig[:len(sig)-1], nil); err == nil {
			t.Errorf("Verify with short signature: want error, got nil")
		}
		if err := Verify(sk.PublicKey(), msg, append(sig, 0), nil); err == nil {
			t.Errorf("Verify with long signature: want error, got nil")
		}
	})

	// Cross-parameter mismatch: an MLDSA65 public key encoding is rejected by
	// MLDSA44 (and vice versa), because the lengths differ.
	sk65, err := GenerateKey(MLDSA65())
	if err != nil {
		t.Fatalf("GenerateKey(MLDSA65): %v", err)
	}
	if _, err := NewPublicKey(MLDSA44(), sk65.PublicKey().Bytes()); err == nil {
		t.Errorf("NewPublicKey(MLDSA44, MLDSA65 encoding): want error, got nil")
	}
}

func BenchmarkSign(b *testing.B) {
	// Signing works by rejection sampling, which introduces massive variance in
	// individual signing times. To get stable but correct results, we benchmark
	// a series of representative operations, engineered to have the same
	// distribution of rejection counts and reasons as the average case. See also
	// https://words.filippo.io/rsa-keygen-bench/ for a similar approach.
	b.Run("ML-DSA-44", func(b *testing.B) {
		benchmarkSign(b, MLDSA44(), benchmarkMessagesMLDSA44)
	})
	b.Run("ML-DSA-65", func(b *testing.B) {
		benchmarkSign(b, MLDSA65(), benchmarkMessagesMLDSA65)
	})
	b.Run("ML-DSA-87", func(b *testing.B) {
		benchmarkSign(b, MLDSA87(), benchmarkMessagesMLDSA87)
	})
}

func benchmarkSign(b *testing.B, params Parameters, messages []string) {
	seed := make([]byte, 32)
	priv, err := NewPrivateKey(params, seed)
	if err != nil {
		b.Fatalf("NewPrivateKey: %v", err)
	}
	rand.Shuffle(len(messages), func(i, j int) {
		messages[i], messages[j] = messages[j], messages[i]
	})
	i := 0
	for b.Loop() {
		msg := messages[i]
		if i++; i >= len(messages) {
			i = 0
		}
		priv.SignDeterministic([]byte(msg), nil)
	}
}

func BenchmarkVerify(b *testing.B) {
	b.Run("ML-DSA-44", func(b *testing.B) {
		benchmarkVerify(b, MLDSA44())
	})
	b.Run("ML-DSA-65", func(b *testing.B) {
		benchmarkVerify(b, MLDSA65())
	})
	b.Run("ML-DSA-87", func(b *testing.B) {
		benchmarkVerify(b, MLDSA87())
	})
}

func benchmarkVerify(b *testing.B, params Parameters) {
	priv, err := GenerateKey(params)
	if err != nil {
		b.Fatalf("GenerateKey: %v", err)
	}
	msg := make([]byte, 128)
	sig, err := priv.SignDeterministic(msg, &Options{Context: "context"})
	if err != nil {
		b.Fatalf("SignDeterministic: %v", err)
	}
	pub := priv.PublicKey().Bytes()

	// "Whole" runs both public key parsing and signature verification,
	// since pre-computation can be easily moved between the two, but in practice
	// most uses of verification are for fresh public keys (unlike signing).
	b.Run("Whole", func(b *testing.B) {
		for b.Loop() {
			pk, err := NewPublicKey(params, pub)
			if err != nil {
				b.Fatalf("NewPublicKey: %v", err)
			}
			if err := Verify(pk, msg, sig, &Options{Context: "context"}); err != nil {
				b.Fatalf("Verify: %v", err)
			}
		}
	})

	// "Precomputed" runs only Verify with a pre-parsed public key.
	b.Run("Precomputed", func(b *testing.B) {
		pk, err := NewPublicKey(params, pub)
		if err != nil {
			b.Fatalf("NewPublicKey: %v", err)
		}
		for b.Loop() {
			if err := Verify(pk, msg, sig, &Options{Context: "context"}); err != nil {
				b.Fatalf("Verify: %v", err)
			}
		}
	})
}

func BenchmarkKeygen(b *testing.B) {
	b.Run("ML-DSA-44", func(b *testing.B) {
		for b.Loop() {
			NewPrivateKey(MLDSA44(), make([]byte, 32))
		}
	})
	b.Run("ML-DSA-65", func(b *testing.B) {
		for b.Loop() {
			NewPrivateKey(MLDSA65(), make([]byte, 32))
		}
	})
	b.Run("ML-DSA-87", func(b *testing.B) {
		for b.Loop() {
			NewPrivateKey(MLDSA87(), make([]byte, 32))
		}
	})
}

var benchmarkMessagesMLDSA44 = []string{
	"BUS7IAZWYOZ4JHJQYDWRTJL4V7",
	"MK5HFFNP4TB5S6FM4KUFZSIXPD",
	"DBFETUV4O56J57FXTXTIVCDIAR",
	"I4FCMZ7UNLYAE2VVPKTE5ETXKL",
	"56U76XRPOVFX3AU7MB2JHAP6JX",
	"3ER6UPKIIDGCXLGLPU7KI3ODTN",
	"JPQDX2IL3W5CYAFRZ4XUJOHQ3G",
	"6AJOEI33Z3MLEBVC2Q67AYWK5L",
	"WE3U36HYOPJ72RN3C74F6IOTTJ",
	"NMPF5I3B2BKQG5RK26LMPQECCX",
	"JRGAN2FA6IY7ESFGZ7PVI2RGWA",
	"UIKLF6KNSIUHIIVNRKNUFRNR4W",
	"HA252APFYUWHSZZFKP7CWGIBRY",
	"JFY774TXRITQ6CIR56P2ZOTOL6",
	"ZASYLW5Y3RAOC5NDZ2NCH5A4UY",
	"42X4JXNPXMFRCFAE5AKR7XTFO7",
	"YAHQUWUH534MUI2TYEKQR7VR3A",
	"HBP7FGEXGSOZ5HNOVRGXZJU2KG",
	"HG4O7DCRMYMQXASFLMYQ6NMIXK",
	"2KPQMDZKS65CLJU4DHTMVV5WI3",
	"G6YSUTEX4HHL44ISK2JVVK45BV",
	"PUJGPEQUBQM3IK2EXDQFJ2WGBG",
	"PNS6HMQAWA3RORSMSNEUAINMIR",
	"L35MZS4XYIJK453OFXCZG4WHIK",
	"CRY54YZMFRF6JTB3FPNNBWPUOG",
	"Y25TSZBWGU4HJCRMWZHAWXQ2DN",
	"23W64TW3AKZPKCM4HMKEHFI6VQ",
	"PWQAOZ24B4VLNEQR4XKN7LZHDI",
	"YINPDR3ZSAKPPXP6J6VAXHIPYO",
	"JDBB52ZRAB3PYBPNE7P4COY5PJ",
	"4DYU52LQLVG3LTREOTLBCJK3XC",
	"AB45MV6RKUGPCW4EUK7DX23MJX",
	"HEJSITE5K7J6YJ74OEATVTCERV",
	"ZKI5QCFCGM26UK7F5KYTENXKD2",
	"VH5G3ZLF5XC22QAEJ6JDGOBE5Y",
	"HYGXFHH3JW5SENG26MXLL54IGV",
	"MJUCRL36JZ757UYHBFPCJBPZRH",
	"IBH3T6NAVLCJQBYSVHAQFUITYA",
	"VMWCS7JMIMFQB6TPRAMOUXIKWD",
	"SXRPGPNNW2MMBKQS3HJURIQ3XV",
	"YPPYMJZW6WYXPSCZIPI57NTP5L",
	"N3SH6DUH6UOPU7YMQ6BJJEQSPI",
	"Q243DGA6VC6CW66FFUAB5V3VLB",
	"OUUBXEU4NJBRN5XZJ7YQUPIZLA",
	"H5TWHVGC7FXG6MCKJQURD3RNWG",
	"OONG2ZZ7H3P5BREEEURNJHBBQG",
	"HWROSSRTBCQOAIQAY5S4EQG4FX",
	"AJW6PW62JQNU72VKGIQMPBX64C",
	"OXECVUVAWBBBXGGQGQBTYVEP4S",
	"M5XN6V2LQJDEIN3G4Z6WJO6AVT",
	"NHGJUX3WGRTEIRPFWC2I467ST4",
	"SEOADTJDKAYYLDSC4VAES2CRDJ",
	"J5AT674S577ZFGEURNIAGYOHKW",
	"VJQVNMGHG4ITFX2XSPSDEWVZWD",
	"ZWY3KJPXTAVWWVHNAJDUXZ52TG",
	"HY46PBUGP4EMH34C6Q56MO7CJP",
	"MQTUO7CF6R6CRJPVV6F673M6VW",
	"35Z2Z5KV2RBJPQ7OZ24ZJE6BKR",
	"OVUEVXBLCU2BBY25QP5WJACDIX",
	"LNJX7PCLYL35WYJBW6CTXENPUU",
	"IH7E766LCENOQ5ZKZVCMLEPACU",
	"T2HZFGDDSFQ6YADB52NIFLBFEV",
	"RHQUJMN4MB5SYY4FP4ARZH52QJ",
	"W7GZC5ZM63UF2EJ7OC4WJM3OTH",
	"T2NHNFVOMICY33AQZSR53HXFQ6",
	"7ZVB4Y4K4Y2VAM5NC7HHAJNZIB",
	"UX2I4VF62XJGP2XTNN6LDKXTOH",
	"HJAMJR5RQTQW7JMW7ZLPRBZE7E",
	"HKWSKX7MB5346PHYNWNBAYDSYK",
	"BVWSB75HFLLE45MWA6EPHPTCFR",
	"YDH2J6NMM7UINHGUOPIUI7PSSR",
	"SYQPZLK52HMUAQFMVHGRJYKBEY",
	"7AA6UQFGSPBGNUDPLWXSGNKKPP",
	"AYXRJGRWZ5S3QOEDVWYHHCICHV",
	"KFJYAWO7IATSBCSTDUAA5EPFAN",
	"3JABTLB6T2ICHGVT3HXZZ3OAIT",
	"WCM3IBOCQJ36WSG627CCNK3QA7",
	"5FB5H3BZN2J4RGR2DUW7M37NKZ",
	"VKDDAD3BVOMPSNEDGIRHKX5S6R",
	"LFH5HVUR726OSFD3YVYM3ZHEIH",
	"Y4ETQB2KZVFB4M7SALLCTHX2FB",
	"E6SAU3C25MO2WBBVBKCKP2N4ZE",
	"3JA54Q3NEKURB5EAPL2FOFIESD",
	"FZPBW7BIQIW3FTKQD4TLKNWLMD",
	"LY5W6XFA2ZRI53FTUJYGWZ5RX6",
	"QID236JY3ICR55O5YRED33O7YT",
	"HDRU3L6MFEBCBQFNLF5IRPMOAL",
	"232ANKJBDBG4TSKQ7GJMWTHT23",
	"CDWE3CELZM5AOJGYEFHMUNSP5O",
	"7LNJRBOKN6W7RXUU34MDJ2SNKL",
	"S3IZOADTW2A6E5IGRO5WKX7FVH",
	"ZAISTLXC55EBMTN6KZ6QX5S7OS",
	"4Z5ZIVCMFR2PY2PY4Z47T4YPYA",
	"NE36L53Z6AMYQU7Q5REFUF76MK",
	"WND5UP5M6KWPBRFP5WIWTOWV3I",
	"7OC54DLFWMADJEMKEJ3Y2FMMZS",
	"BWJVZHGEN43ULNIOZCPZOB64HG",
	"VDFPQSR7RE54A75GT4JDZY5JK2",
	"HFCD5EPBZBSVMXIDA47DZ6MRD6",
	"RNBVFIUUJUM7EHRE3VNWSTORGO",
	"VO5NLQJBR22CRRYUETGTU6JLMR",
	"RZOMNFHBTL6HMGWH4PEEDASK7U",
	"QL73UBTOLK5O2TW43YWAIKS6T3",
	"NE3QVSMWS5G3W5C3BMKTJNMI2L",
	"YHI6EYQ4GZMB2QPGHPUG2ZUOEL",
	"6MBATW7MFNRUQBFD3GM35B7YPM",
	"AIYRY6P5T4XU44CGVPEV6W43FR",
	"MIAQ2FHXMAPY5NXSS45VRDPRMG",
	"2SNLHQYKK2K6NSWOF6KPGZ3CPC",
	"RVBHIQO5LH77ZWEAO3SVL72M2V",
	"XXTGJCJNRSNLE7ARAH2UU6LVKR",
	"DQMGILY5IDMWN5OYQYYXH26ZGR",
	"627VTXXMM455KMTFNUUTKNFXPY",
	"HC7IBFGLZCWGUR4K7REPMPW6W4",
	"CHL6JRQUS7D4NML3PFT37PPZAA",
	"Y767HXJAGJ75KE3JLO4DTLQIXC",
	"NTIODXI5I7TF2KXXWXOAYGT7G4",
	"PKZYEK2WAI4D4HEYYZH6H5IOMP",
	"FG6J6G7HZDEDF4JQBQOTC7RQGZ",
	"3VHM2VZU77Y25E3UUYZJLB2QLA",
	"WRZQJQW7ARH4DXYHVLCJ4HRTTB",
	"LQXKV5HD2AZHENSJ2VFLJ5YU5L",
	"MF6Q4OA2EN6TG6BUDK7RWCQNPU",
	"3USKYKPC5CB3EC4ZRMZVE3R2UO",
	"3WICO2GVS3IRBFUHNDLNKWVP7N",
	"P6ZR2UZZOVUZKT4KUS5WICW5XE",
	"PYPZUU76RYVOUZGUUX33HLDKYA",
	"2FTSURHV34VYTVIUU7W6V5C3NK",
	"YABDYMGXS2MD2CYF3S4ALG4FLG",
	"MHIBDH25RRPWV3P4VAWT6SAX3I",
	"OINSMWJQ2UTOOKZ3X6ICXXBQR7",
	"PFTQS7JNU2Q3Q6L4CGBXVLOYNE",
	"A4MZ7CCVYQUDJ2AFHNXBBQ3D24",
	"CPUB5R3ORTCMSMCLUQURE6AN5O",
	"NF5E7U3DFTXWFFXXHUXTEP4VZQ",
	"AWB5WDFERWSSJG53YGJMDORQKR",
	"U5JQUILKD6SEL6LXAMNFZP6VSW",
	"M45NLOAFLO74EJKG5EXNET6J5Y",
	"P2KTEUMZ5DZZMYSPOHDR2WJXAN",
	"KVO7AXZNFBUBPYLOTZQQ42TFNS",
	"WGJJ7SAEV6SBBWWYS4BTLD63WM",
	"Y6GURVDV4ESRBPWSTV25T4PE4K",
	"ESK7MPFPUZ5ZAQ52RP4SQIYCCC",
	"623M3CIABZ3RANERQ2IREXAVYO",
	"OQ4CQCFO42RS4BMMSGSDLUTOQO",
	"AMFHRDVGM6G2TIR3TKIFGFSDVM",
	"7VVSGGCVC53PLOYG7YHPFUJM5X",
	"Z3HMESVL7EZUSZNZ33WXEBHA2N",
	"AWWVRQD5W7IBSQPS26XOJVDV5H",
	"OQBZ5ZST3U3NZYHSIWRNROIG6L",
	"II573BW7DJLBYJSPSYIABQWDZD",
	"MOKXOQFOCUCLQQH4UKH2DPE7VN",
	"XR54NGUOU6BBUUTINNWBPJ35HX",
	"DNK36COZGFXI6DY7WLCNUETIRT",
	"R5M2PV7E3EHEM3TLGRCL3HSFMC",
	"ITKENZQYDQMZFCUPOT7VF3BMU7",
	"5GDCB74PPPHEP5N5G3DVRCYT7R",
	"ZMKXVRPLI5PY5BDVEPOA3NQZGN",
	"GBLIALWTHTUDTOMDERQFVB77CS",
	"VKRTTXUTFOK4PJAQQZCCT7TV3T",
	"ZJBUJJ4SW62BXOID3XO2W2M2PF",
	"SKWT5T6QJTCD3FCINIK22KMVBJ",
	"EHINNU6L33HRLOOJ3A2XFJSYQL",
	"N4HRQJEFPAT5SU3YPO74WSMQIR",
	"TGPTZ3ENMFWB5CZKJFR5WHIRI4",
	"O4HNFTAUJJ2LZPQXPXRAXOVABA",
	"4JVB5STP2YG5GYOXDWIF4KCKFB",
	"MY554X3YZHBECLHNNZ7A3SPJTU",
	"ASCJMAH7VCQAD2QJSWXPSVSM3H",
	"NBNGL5DZ623KCG2JNZFGZMZ7KD",
	"KGMZSW35AEQOJ6FA7IR7BHZI52",
	"Q7QUHHS4OJFMJ4I3FY6TDKSMZQ",
	"MZAE7TOEXAS76T7KIC73FEYRU4",
	"2BVESR3REAWADCGYOYM7T646RG",
	"EK3L2ORP4LT3HU3EMXDSQWFOKJ",
	"3X4A6VMGMIDLVK72FZSDHSERWY",
	"I3UHWI6M6HQFRBSQ6W2SABUNUP",
	"REKPXW4DIB4MTKMPHN3RBVHVME",
	"W37FNFZE35NX65Z7CVQ7L5U4L5",
	"4AGYK6U2KP6RAOADCBUDDCBECV",
	"IXM4SFQUDW2NOTXZIPWTNGET3F",
	"6YE4G3VELF27MN3Z5B4VIQ3XYK",
	"LPOZCPZAG3MD47MIWGR4FIOCDH",
	"WGREKUL2LD7C7SYGKH7APIY2A6",
	"WWW277FKTKUXQMP4BECSRHLWJI",
	"UYE4IQPMSTXVQG7EJALKWWEGDN",
	"TIV2L5Z6K7SNGNUVWSNKTAF4UE",
	"I3FQOAW3PINUK26P62HCX657FO",
}

var benchmarkMessagesMLDSA65 = []string{
	"NDGEUBUDWGRJJ3A4UNZZQOEKNL",
	"ACGYQUXN4POOFUENCLNCIPHFAZ",
	"Z3XETEYKROVJH7SIHOIAYCTO42",
	"DXWCVCEFULV7XHRWHJWSEXWES7",
	"BCR2D5PNLGFYX6B3QFQFV23JZP",
	"2DVP5HNG54ES64QK4D37PWUYTJ",
	"UJM4ADPJLURAIQH4XA6QYUGNJ6",
	"B5WRCIPK5IVZW52R6TJOKNPKZH",
	"7QNL6JTSP62IGX6RCM2NHRMTKK",
	"EJSZQYLM7G7AJCGIEVBV2UW7NN",
	"UFNA2NKJ3QFWNHHL5CXZ4R5H46",
	"QZAXRTT3E4DOGVTJCOTBG3WXQV",
	"KH2ETOYZO5UHIHIKATWJMUVG27",
	"V5HVVQTOWRXZ2PB4XWXSEKXUN5",
	"5LA7NAFI2LESMH533XY45QVCQW",
	"SMF4TWPTMJA2Z4F4OVETTLVRAY",
	"FWZ5OJAFMLTQRREPYF4VDRPPGI",
	"OK3QMNO3OZSKSR6Q4BFVOVRWTH",
	"NQOVN6F6AOBOEGMJTVMF67KTIJ",
	"CCLC4Y6YT3AQ3HGT2QNSYAUGNV",
	"CAZJHCHBUYQ6OKZ7DMWMDDLIZQ",
	"LVW5XDTHPKOW5D452SYD7AFO6Q",
	"EYA6O6FTYPC6TRKZPRPX5N2KQ4",
	"Z6SGAEZ2SAAZHPQO7GL7CUMBAG",
	"FKUCKW6JQVF4WQYXUSXYZQMAVY",
	"LN2KDF4DANPE4SC4GKJ4BES3IZ",
	"AVCRTWB6ALOQHY34XI7NTMP2JH",
	"A5WHIS6CBWPCYIEC6N2MBAOEZ6",
	"JC2BH476BXUQFIDA6UCR5V4G4F",
	"NU6XH6VLSSFHVSRZCYXPFYKYCD",
	"GSUXVZBDDYSZYFGXNP6AZW3PTC",
	"XJPRNJ26XP4MIYH2Q7M7MPZ73M",
	"INUTUP3IRFWIIT23DNFTIYKCFY",
	"T4KH7HKLEYGXHBIRFGFCRUZCC4",
	"GGQX4JFVWZHE5Y73YTLMSSOXNS",
	"BUA4Q3TQZGLVHMMJU62GQOSHLV",
	"WXW3SJXLSZO2MYF4YFIMXL2IQP",
	"Q32XBVVGFQTSXAIDJE6XSEPRZG",
	"6TEXT6SA7INRCTDSCSVZJEQ2YG",
	"ZBN4UL43C3SJIG4HYR236PXCVS",
	"TVWPLLC7NROBREWOM75VA3XCR3",
	"CCDGL2FURLBABQ4IJBYCB75JFR",
	"XBZGCOVTZHCPAARBTMAKPIE6GJ",
	"TPRAENJ7I54XRIVH6LL6FDIA3I",
	"RKOM3PHFILPIIQZL4ILQWGRYWI",
	"CEEZIZ2WUXHQQFATYYGQ3ZDBTI",
	"SLKOVAP6WLIVJBVU7VZG3ZGEOW",
	"TWMCLJJSWEEQQPQGGDKEJ5SU2R",
	"IFMUXXCD2LC7IGQLZ2QEK5UOQ2",
	"C7IWFEBHW2CXN4XBJS7VLWH3VK",
	"7KJYUEW3F264727TM4LE6RMGDO",
	"BPG2XAPBMBTA4VMPUM7IZVZPK3",
	"Y5X577BWRZNPLNUHJVSKGMUXYB",
	"ZCKMKM23E4IUPTNQDFN2LTLZVX",
	"4RKK223JNBDAP4G5DOAHHZ3VNO",
	"5UZ3TQZHZT22ISTB4WJEVO6MC4",
	"YMVS4HFSJ32CRZRL23PXZUEJFJ",
	"UQEUJUTPSZLZARNBXWMCTMHPFF",
	"CZAAZ5WK7EIPMW7NA3EZNNBF45",
	"227PBHH23WM7F2QLEZSPFYXVW4",
	"YUYS2J5CRFXZ4J4KJT2ZKIZVW3",
	"MFLHZJOZV44SN4AH6OJ3QZWM2O",
	"H2B3CRBCXYN7QWDGYUPHQZP23A",
	"T4L6YWQUQ3CTACENAJ5WUXZWFH",
	"N723H6MUGPZSRZ72C635OD4BP7",
	"NI4TUMVA6LQPQV2TXPN4QOIGBZ",
	"CQI3S4LSTQASSJJVZXEFPOVW7K",
	"ANPY4HJ64LLSB3GK2R4C6WDBS3",
	"RGWQCZKQLMT5FZRDE4B3VMASVK",
	"Q3WCCF2HA3CA4WWRJBMGBW7WI7",
	"2AKJRXFHXLUQPOXPTLSZN5PW4A",
	"IJWOOTI4N7RWXJIHAPXN6KEWEN",
	"4D53T6N6ATOVTD4LKSTAAWBJMU",
	"B4G5HDD6RITG6NIH6FXCRZDYZM",
	"TJCDFKMRUY2OG6KRSMNVCGQFUP",
	"PB33IHQKALAY6H6GVBVLI6ZRXK",
	"SCCWGW2J5S4WL4FTTMQ435F6DB",
	"ZVJH2HSMTLHGXMGPMXLJCKCLLE",
	"62LG37U6JXR77YRZQQCDSBHVCS",
	"BU4CBWOXQ352TEOKIXO245ID4O",
	"UEZOH7KEIODSEVRUF6GMWGA2RB",
	"IPJWROME4GM66CGLUWP5BJ4SX6",
	"355GDC7TG64AZJ7IJX6K62KZCZ",
	"AHTFKX3V7XUB3EWOMQVCGZYGUE",
	"N4RV2GKXJ4SPHHJ52Z7K5EGLER",
	"ZY7V7NE5F66XHDHWM6YNFEWZA6",
	"DIKFO5KAVT4WAP7BOEFM56ZUSR",
	"4TDFOFKDAPIOM3MU5GD7NPXNWQ",
	"AD7YZO756HDK6YWFILAKW3JWA7",
	"NUA53JS2ZK2BGHH3A7BJTJZYW7",
	"QLCNC3AQNKLRMSYR62WQSQP5VI",
	"SJ7OBS7ZYXSGXOYXPE5KW2XKN6",
	"44HBMOGMIMJS63CEXQU7FCXE2E",
	"KCK3J7ZL6QF4SLHHSWTJURK7PG",
	"HLH4CLUGBSOOBSS3BPO62N5MC3",
	"3FNS4GITO6OEUBAVDDXK4WOBTD",
	"IAC3K3I4AQGY3G6UHG7PL2N6TE",
	"KUKLNH74POJI5DYAEWUD7RABTQ",
	"ETM6N7VU3GBSQ7P5MCD6UF3E3S",
	"IZITM5NYBGJZLSI3BI4VEMW43U",
	"46OPQU4LL6N3Z2U7KYPKUMBAGI",
	"EV7YZ5DMAV7VKYJQUFSRD37GPP",
	"AV7W2PGYDJIAKLFVEBL6BXQSGC",
	"M2FOX5QZEZKV4QXKPI5XUZDHEM",
	"R4IFPLVMOVYCHRTR6LXAUGP3LL",
	"JGH6XJUMP4DRVAM27P2JNOKXVO",
	"D2XN3ZLLU6VFPMDYM7NBHSQEOI",
	"2PO3BYENOMQK6SHQDCFSRPJQI3",
	"IBVQ7U3QEUC6PQRE4PV53JTZTK",
	"ZBCOX4P7NG2IXXFB2R43MG2SLV",
	"5NJDPQVVDO7ADNZ2CV7L6QBNGZ",
	"V7ASFIIYUMXFGW4B7ZM6LOGUTE",
	"PX5IJZ7W2LUPKM6YN4PMZ43ZLM",
	"AYK7SZ23DHC7Q56MWAJXBG76LB",
	"UYCAPXJM4HNGKLIDSZ4NCEDJLN",
	"UWMDZ3C2ODLACKGJPGETNQ3TA4",
	"Q6OI6R3WYYJ4CCZCDJBQMCRCZR",
	"LCMJHLP7354APCEGPKE7HHWTWB",
	"N7T7ZKOYPAMEYTTDOWZNCN6PRD",
	"UZADPU4UNHAF7L7LQDMTKA2EQH",
	"DC2OEPQDECVLRVNNCS6BMH4CRA",
	"37IZ427XHUMZ66EJ62U2YEZDAC",
	"6BCZDQZDPZLS5OGESKNUBPSSFV",
	"ST2LEMJ4OLQ32TJTLH2WCWT4WA",
	"GA2TL4SFLEW4G2B5PQMIKJT5XG",
	"L7PPBIET26EH7LQTLEFC4I4EIA",
	"6YSM7MC2W4DEV6ULAHMX27LH56",
	"QL26Z5KZ4YRRG2BXXGDRRLV357",
	"677TWRAJ5NSNHCE243POQPEG7K",
	"66MEBQJLGAGVXDX3KZ2YFTTVJM",
	"6D4VUWAQD6R65ICSDLFAATC67V",
	"7GXLD5CNU3TDUQSSW42SHL7B5D",
	"RQETUMEBG2ZM2NF2EZAQHGHWWE",
	"DCRX5ANWDMXZFIDVAXYLQZYMRN",
	"5SDWT7YAF7L4WWANAGYINZAYXH",
	"PZILRV7I2S6WKUSHKYRLA2JQY3",
	"2G66TK2PZ5MOTAZDN7BFS3LAIH",
	"QOLJ3WGJ6JS3FMMXBNTNAIKXVK",
	"FMAL67YTHDCCYVZ5CRMN2XJPDN",
	"UOTZDXTJKQ3YAIRKHTYNX6G55P",
	"X3DLNPJ3V62LRHGEY4DTT35H3R",
	"DKU7CHNXPB5QRZVGIQZW46XCKC",
	"RAKBD4LQKEDTVDSK3DVTRWG23B",
	"INTRA7BWHLVQMBRKBJNUSMF7MU",
	"AUYRBNVCOYYHOHUYOOFIZ2FWMD",
	"22EJVDEQ7PASLBAMTVKXOQP5RJ",
	"3S6NATWA57SFTZEW7UZUOUYAEU",
}

var benchmarkMessagesMLDSA87 = []string{
	"LQQPGPNUME6QDNDTQTS4BA7I7M",
	"PTYEEJ7RMI6MXNN6PZH222Y6QI",
	"R6DTHAADKNMEADDK5ECPNOTOAT",
	"S2QM7VDC6UKRQNRETZMNAZ6SJT",
	"EYULPTSJORQJCNYNYVHDFN4N3F",
	"YETZNHZ75SXFU672VQ5WXYEPV2",
	"KTSND3JGA4AN3PCMG4455JEXGR",
	"JGE6HK37O6XMWZQZCHFUPNUEXP",
	"CRYB2FZD2BYNANBFFO2HRZEHGZ",
	"7MLNDZJ7OIEPBJZOMULOMQH2BA",
	"4WQCNTIFVSX2DNALMWUKZRA6CI",
	"Y5NK4OBDSDWC5WLL27CEEXYYOT",
	"C4SSWSPBVCDAWJXH2CDMXR36LH",
	"THDBKXRTKWJUGJMAAYTWTFMX7Z",
	"NWXPUD4DAA6QOREW4AFFYQYQNG",
	"3RQIJXMO7WYHBEBL3G6EOLNZNQ",
	"R7JEOHFP2C7O4AVPRPRELXWOMM",
	"LU6MWR7SZXVIKS54BY62X67NPA",
	"FG2FFM4F2ECKHCSJ75KXK632JP",
	"BF76ZDSVVUSYS5KK4FFD22YPS7",
	"HCLBWZRLHEMYZLFWHLAN2BKCZ7",
	"HGFVS4QC7AWXYPVRSWAK77KTQF",
	"LUZ3C53PUUHBWCDJ7WAHK2UT3K",
	"Y3WR6SMDUBW34N3MUT7EQYIJCV",
	"F2X35AQTXVZBMPXTWNAAH4ZX2W",
	"6MKFFDYWD6ZAKS3C6GRCRLZLRF",
	"AFMZYYFRHKMQRNKU5UTSKQ74H6",
	"TDTN7J3O367OVPWLESRNPLN4M2",
	"WYMLD2X6N4CZ2RDOKF5CFTSYTG",
	"UNPTSBLJ6HZRNR72T2VEEHCFX2",
	"SNCM4R2P27AJOXBS67RMCARS3U",
	"OU7QBE5QOXO7CIYTBJR3KOW2WK",
	"2NNQOBQKZ2OD4ZAXI3SNEURYUP",
	"YQTUPOYBT67XPCHIGKSGSKC3BZ",
	"HGB4ZM3G76IXYWWCMVT3HONRIS",
	"WZC6QUKRZZ2TOVA277JYKQITEW",
	"XO2WT46A5HYL6CUJF7SGJ6YWOG",
	"4QJA35PMYQIDRZ7ZHG7RLZJVGF",
	"BMJZELWZ4I2UWXESU3NR6ATC4M",
	"XWLFB7FN6D5PRY6YUXC5JUIBFM",
	"WRAFFF27AVTIOYIBYA2IPTXI3R",
	"VOXUTYTN2XZ362OJFO2R53UCUF",
	"UHN73ARJ737WUJ6QYEI7U46OPO",
	"3Y3K5E2A4ML3VYVNAFWEEIXTSN",
	"QMU4322NKPRLE7JBGYFGS36H2S",
	"NJAQTNCXPVDICTDVUKTPRCD2AX",
	"OC373ZFBNV2H46T6OY3XRPSUHG",
	"UBLAS6CDWE3A662MLKP7QDEOCC",
	"BKFDLAL2RTPMERYVW3B7UJ5W3H",
	"QFKFGXKGW5SAKLBAWQXUWW77OS",
	"EJNUQHTLLOVB4ARETOGLY4WUTJ",
	"N243OCMVLLAO6I2XLCYOIMQYGY",
	"YRRFLWK7ZASUKYX7ZLQMW2PJ6X",
	"3DGVPBWD2BIK6KQE65K72DNJNM",
	"TJRYMNOAIW33VIHKLJG4GXAVUK",
	"6DSRINAYXL34U54U355U7IVFGS",
	"6CHA4MX7LVS77XKRWG7IYC3XVL",
	"GM2CEGBEPBOHAPIOBUWJ4MJNTG",
	"VJKHGBY33VUIJFEQLX3JVUNQBD",
	"DTOHAD5M2KL46IZHE4TPLJWHTI",
	"IYFG3UDN7ROOY2ZFSLM2BU2LMQ",
	"A5OGJHPOE4PW6QSZYHZ5TKPGIC",
	"FX4BCN67AEGCLUTLFPNDL3SQU5",
	"MWIZQVOZOHTTBUXC3BEX62MNI5",
	"BYHVJHBLK4O6LFSKEIQ3CAAKU7",
	"QJU7P6KWSSKAA5GVA6RH4OV7MX",
	"I3T3XM5Z5TAJHAYDQHFA2ZV7PU",
	"L46MQCHV3TJ6FYIQQ2FCJXES74",
	"QXZRQIYAJMXYR6PU3VDYGCIT5W",
	"MFS53RR2XEYS22NYOJLGTHVTTM",
	"FRWIWJRP4AQMXWX4WJ4WYVKM3E",
	"X6GK6IGVLJWYSHLKHGXSW3TJDP",
	"L5LPJ2HIWA4UY6G6FMZXGDEDAM",
	"GD6FYOYUGDHXEQ5S2KLJEGNSN7",
	"ODAL7ZRKXSPAAN5DVRBWJQCFQX",
	"CV3QFBDXBPT3SCPJGUYSMDN6ZS",
	"IGSLSACRZ6XID466KQIB4YNGYO",
	"WZ2EACBN26RAML2S52YXRYP2OF",
	"LB76VEVNOBYFMKFZ7SDFCBCHQE",
	"TLFA7EU3JJFAP6EMUKNV2ZXRBM",
	"SIIJF6OXAKRP25CBUYFBRCDDVP",
	"TEPNI7TJ7HASJWIQMBS4VFLRQC",
	"VK2JINYWEDV7IQFWH4OTAD4W5O",
	"GILUH5AMVE4TM7EKPXJBZGT6EJ",
	"DV7ALFRAW3TI4WMQQLDTO6RNHN",
	"CAIB5G3NXC5ASPLFIWAFPVHS5B",
	"MLFJXZUOAGN7EGPMXOOVTB2CL4",
	"6MZYT3ANWHBOS67WGHZI3QPEAP",
	"LVJDQB52C2PERSSQJRMRCJ4UBF",
	"QY4VKAZAYQIZOX2L2VO2QHAQVC",
	"UAA5SST2XA76JPKM3XOZ5RUHFI",
	"VLZWF53JSQ6SCRUFDKVPXWAS4L",
	"NX2DZIKMJIYXUNSAHFP23FHTBU",
	"F5OAKDDDA34A2RPIKDPM5CYPMZ",
	"E5PEP3ANIK2L4VLOST4NIYNKBD",
	"IPBGFLHSMP4UFXF6XJX42T6CAL",
	"XHPU7DBFTZB2TX5K34AD6DJTK3",
	"2ZU7EJN2DG2UMT6HX5KGS2RFT6",
	"SD5S7U34WSE4GBPKVDUDZLBIEH",
	"WZFFL3BTQAV4VQMSAGCS45SGG3",
	"QE7ZT2LI4CA5DLSVMHV6CP3E3V",
	"YIWMS6AS72Z5N2ALZNFGCYC5QL",
	"A4QJ5FNY54THAKBOB65K2JBIV7",
	"6LORQGA3QO7TNADHEIINQZEE26",
	"5V45M6RAKOZDMONYY4DIH3ZBL2",
	"SVP7UYIZ5RTLWRKFLCWHAQV3Y2",
	"C2UYQL2BBE4VLUJ3IFNFMHAN7O",
	"P4DS44LGP2ERZB3OB7JISQKBXA",
	"A6B4O5MWALOEHLILSVDOIXHQ4Z",
	"DKQJTW5QF7KDZA3IR4X5R5F3CG",
	"H6QFQX2C2QTH3YKEOO57SQS23J",
	"DIF373ML2RWZMEOIVUHFXKUG7O",
	"Z5PPIA3GJ74QXFFCOSUAQMN5YN",
	"PM6XIDECSS5S77UXMB55VZHZSE",
}
