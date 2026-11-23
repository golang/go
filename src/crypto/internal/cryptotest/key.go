// Copyright 2026 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

package cryptotest

import (
	"bytes"
	"crypto"
	"reflect"
	"testing"
)

// KeyBytes returns the encoding of a key.
//
// It is usually a method expression like (*ecdh.PrivateKey).Bytes.
type KeyBytes[K any] interface {
	func(K) []byte | func(K) ([]byte, error)
}

// privateKey is the interface documented in crypto.PrivateKey.
type privateKey interface {
	Public() crypto.PublicKey
	Equal(x crypto.PrivateKey) bool
}

// publicKey is the interface documented in crypto.PublicKey.
type publicKey interface {
	Equal(x crypto.PublicKey) bool
}

// TestPrivateKey performs a set of tests on a private key type, checking the
// Equal and Public methods documented in crypto.PrivateKey, and its encoding
// like [TestKeyEncoding].
//
// k1 and k2 must be different keys.
func TestPrivateKey[K privateKey, B KeyBytes[K]](t *testing.T, k1, k2 K, newKey func([]byte) (K, error), keyBytes B) {
	parsed := parseKey(t, newKey, encodeKey(t, keyBytes, k1))
	t.Run("Equal", func(t *testing.T) {
		testKeyEqual(t, k1, k2, parsed, func(k K, x any) bool { return k.Equal(x) })
	})
	t.Run("Public", func(t *testing.T) {
		public := func(k K) publicKey {
			t.Helper()
			p, ok := k.Public().(publicKey)
			if !ok {
				t.Fatalf("Public returned a %T, which has no Equal method", k.Public())
			}
			return p
		}
		p1, p2, pp := public(k1), public(k2), public(parsed)
		if !p1.Equal(public(k1)) {
			t.Errorf("public keys of the same private key are not Equal")
		}
		if !p1.Equal(pp) || !pp.Equal(p1) {
			t.Errorf("public key of a re-encoded private key is not Equal to the original")
		}
		if p1.Equal(p2) || p2.Equal(p1) {
			t.Errorf("public keys of different private keys are Equal")
		}
		if k1.Equal(p1) {
			t.Errorf("private key is Equal to its public key")
		}
		if p1.Equal(k1) {
			t.Errorf("public key is Equal to its private key")
		}
		if zero := reflect.Zero(reflect.TypeOf(p1)).Interface(); k1.Equal(zero) {
			t.Errorf("private key is Equal to a zero %T", zero)
		}
		if zero := *new(K); p1.Equal(zero) {
			t.Errorf("public key is Equal to a zero %T", zero)
		}
	})
	TestKeyEncoding(t, k1, newKey, keyBytes)
}

// TestPublicKey performs a set of tests on a public key type, checking the
// Equal method documented in crypto.PublicKey, and its encoding like
// [TestKeyEncoding].
//
// k1 and k2 must be different keys.
func TestPublicKey[K publicKey, B KeyBytes[K]](t *testing.T, k1, k2 K, newKey func([]byte) (K, error), keyBytes B) {
	parsed := parseKey(t, newKey, encodeKey(t, keyBytes, k1))
	t.Run("Equal", func(t *testing.T) {
		testKeyEqual(t, k1, k2, parsed, func(k K, x any) bool { return k.Equal(x) })
	})
	TestKeyEncoding(t, k1, newKey, keyBytes)
}

// testKeyEqual checks the Equal method of k1, where parsed is the result of
// encoding and parsing k1, and k2 is a different key.
func testKeyEqual[K any](t *testing.T, k1, k2, parsed K, equal func(k K, x any) bool) {
	if !equal(k1, k1) {
		t.Errorf("key is not Equal to itself")
	}
	if !equal(k1, parsed) || !equal(parsed, k1) {
		t.Errorf("re-encoded key is not Equal to the original")
	}
	if equal(k1, k2) || equal(k2, k1) {
		t.Errorf("different keys are Equal")
	}
	if equal(k1, nil) {
		t.Errorf("key is Equal to nil")
	}
	if equal(k1, "not a key") {
		t.Errorf("key is Equal to a string")
	}
}

// TestKeyEncoding performs a set of tests on the fixed-length encoding of a key
// type, checking that it round-trips, that inputs of the wrong length are
// rejected, and that keys don't share memory with the input of newKey or with
// the output of keyBytes.
//
// newKey parses the encoding and keyBytes produces it, like NewPrivateKey and
// PrivateKey.Bytes in crypto/ecdh. k should come from a function other than
// newKey, such as GenerateKey, so that both are tested.
func TestKeyEncoding[K any, B KeyBytes[K]](t *testing.T, k K, newKey func([]byte) (K, error), keyBytes B) {
	enc := bytes.Clone(encodeKey(t, keyBytes, k))

	t.Run("Roundtrip", func(t *testing.T) {
		in := bytes.Clone(enc)
		parsed := parseKey(t, newKey, in)
		if !bytes.Equal(in, enc) {
			t.Errorf("newKey modified its input")
		}
		if got := encodeKey(t, keyBytes, parsed); !bytes.Equal(got, enc) {
			t.Errorf("re-encoded key mismatch: got %x, want %x", got, enc)
		}
	})

	t.Run("WrongLength", func(t *testing.T) {
		for _, in := range [][]byte{
			nil, {}, enc[:len(enc)-1], enc[:len(enc)/2],
			append(bytes.Clone(enc), 0), append([]byte{0}, enc...),
			append(bytes.Clone(enc), enc...),
		} {
			if _, err := newKey(in); err == nil {
				t.Errorf("newKey accepted an input of length %d instead of %d", len(in), len(enc))
			}
		}
	})

	t.Run("InputNotRetained", func(t *testing.T) {
		in := bytes.Clone(enc)
		parsed := parseKey(t, newKey, in)
		scribble(in)
		if got := encodeKey(t, keyBytes, parsed); !bytes.Equal(got, enc) {
			t.Errorf("modifying the input of newKey changed the key")
		}
	})

	t.Run("OutputNotAliased", func(t *testing.T) {
		outputNotAliased := func(t *testing.T, k K) {
			b1, b2 := encodeKey(t, keyBytes, k), encodeKey(t, keyBytes, k)
			scribble(b1)
			if !bytes.Equal(b2, enc) {
				t.Errorf("modifying the output of keyBytes changed another output")
			}
			if got := encodeKey(t, keyBytes, k); !bytes.Equal(got, enc) {
				t.Errorf("modifying the output of keyBytes changed the key")
			}
		}
		t.Run("Parsed", func(t *testing.T) {
			outputNotAliased(t, parseKey(t, newKey, bytes.Clone(enc)))
		})
		// This runs last because a failure leaves k modified.
		t.Run("Original", func(t *testing.T) {
			outputNotAliased(t, k)
		})
	})
}

func parseKey[K any](t *testing.T, newKey func([]byte) (K, error), in []byte) K {
	t.Helper()
	k, err := newKey(in)
	if err != nil {
		t.Fatal(err)
	}
	return k
}

func encodeKey[K any, B KeyBytes[K]](t *testing.T, keyBytes B, k K) []byte {
	t.Helper()
	switch keyBytes := any(keyBytes).(type) {
	case func(K) []byte:
		return keyBytes(k)
	case func(K) ([]byte, error):
		b, err := keyBytes(k)
		if err != nil {
			t.Fatal(err)
		}
		return b
	default:
		panic("unreachable")
	}
}

// scribble changes every byte of b, including its spare capacity, which a
// caller could write to with append. Scribbling twice doesn't restore b.
func scribble(b []byte) {
	b = b[:cap(b)]
	for i := range b {
		b[i]++
	}
}
