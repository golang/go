// Copyright 2025 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

//go:build !fips140v1.0

package cryptotest

import (
	"bytes"
	"crypto/ecdsa"
	"crypto/elliptic"
	"crypto/mlkem"
	"crypto/rand"
	"encoding/hex"
	"testing"
)

func TestSetGlobalRandom(t *testing.T) {
	seed1, _ := hex.DecodeString("6ae6783f4fbde91b6eb88b73a48ed247dbe5882e2579683432c1bfc525454add" +
		"0cd87274d67084caaf0e0d36c8496db7fef55fe0e125750aa608d5e20ffc2d12")

	t.Run("rand.Read", func(t *testing.T) {
		buf := make([]byte, 64)

		t.Run("seed 1", func(t *testing.T) {
			SetGlobalRandom(t, 1)
			rand.Read(buf)
			if !bytes.Equal(buf, seed1) {
				t.Errorf("rand.Read with seed 1 = %x; want %x", buf, seed1)
			}

			rand.Read(buf)
			if bytes.Equal(buf, seed1) {
				t.Errorf("rand.Read with seed 1 returned same output twice: %x", buf)
			}

			SetGlobalRandom(t, 1)
			rand.Read(buf)
			if !bytes.Equal(buf, seed1) {
				t.Errorf("rand.Read with seed 1 after reset = %x; want %x", buf, seed1)
			}

			SetGlobalRandom(t, 1)
		})

		rand.Read(buf)
		if bytes.Equal(buf, seed1) {
			t.Errorf("rand.Read returned seeded output after test end")
		}

		t.Run("seed 2", func(t *testing.T) {
			SetGlobalRandom(t, 2)
			rand.Read(buf)
			if bytes.Equal(buf, seed1) {
				t.Errorf("rand.Read with seed 2 = %x; want different from %x", buf, seed1)
			}
		})
	})

	t.Run("rand.Reader", func(t *testing.T) {
		buf := make([]byte, 64)

		t.Run("seed 1", func(t *testing.T) {
			SetGlobalRandom(t, 1)
			rand.Reader.Read(buf)
			if !bytes.Equal(buf, seed1) {
				t.Errorf("rand.Reader.Read with seed 1 = %x; want %x", buf, seed1)
			}

			SetGlobalRandom(t, 1)
		})

		rand.Reader.Read(buf)
		if bytes.Equal(buf, seed1) {
			t.Errorf("rand.Reader.Read returned seeded output after test end")
		}

		oldReader := rand.Reader
		t.Cleanup(func() { rand.Reader = oldReader })
		rand.Reader = bytes.NewReader(bytes.Repeat([]byte{5}, 64))

		t.Run("seed 1 again", func(t *testing.T) {
			SetGlobalRandom(t, 1)
			rand.Reader.Read(buf)
			if !bytes.Equal(buf, seed1) {
				t.Errorf("rand.Reader.Read with seed 1 = %x; want %x", buf, seed1)
			}
		})

		rand.Reader.Read(buf)
		if !bytes.Equal(buf, bytes.Repeat([]byte{5}, 64)) {
			t.Errorf("rand.Reader not restored")
		}
	})

	// A direct internal use of drbg.Read.
	t.Run("mlkem.GenerateKey768", func(t *testing.T) {
		exp, err := mlkem.NewDecapsulationKey768(seed1)
		if err != nil {
			t.Fatalf("mlkem.NewDecapsulationKey768: %v", err)
		}

		SetGlobalRandom(t, 1)
		got, err := mlkem.GenerateKey768()
		if err != nil {
			t.Fatalf("mlkem.GenerateKey768: %v", err)
		}

		if gotBytes := got.Bytes(); !bytes.Equal(gotBytes, exp.Bytes()) {
			t.Errorf("mlkem.GenerateKey768 with seed 1 = %x; want %x", gotBytes, exp.Bytes())
		}
	})

	// An ignored passed-in Reader.
	t.Run("ecdsa.GenerateKey", func(t *testing.T) {
		exp, err := ecdsa.ParseRawPrivateKey(elliptic.P384(), seed1[:48])
		if err != nil {
			t.Fatalf("ecdsa.ParseRawPrivateKey: %v", err)
		}

		SetGlobalRandom(t, 1)
		got, err := ecdsa.GenerateKey(elliptic.P384(), bytes.NewReader([]byte("this reader is ignored")))
		if err != nil {
			t.Fatalf("ecdsa.GenerateKey: %v", err)
		}

		if !got.Equal(exp) {
			t.Errorf("ecdsa.GenerateKey with seed 1 = %x; want %x", got.D.Bytes(), exp.D.Bytes())
		}
	})

	// The passed-in Reader is used if cryptocustomrand=1 is set,
	// and MaybeReadByte is called on it.
	t.Run("cryptocustomrand=1", func(t *testing.T) {
		t.Setenv("GODEBUG", "cryptocustomrand=1")

		buf := make([]byte, 49)
		buf[0] = 42
		for i := 2; i < 49; i++ {
			buf[i] = 1
		}

		exp1, err := ecdsa.ParseRawPrivateKey(elliptic.P384(), buf[:48])
		if err != nil {
			t.Fatalf("ecdsa.ParseRawPrivateKey: %v", err)
		}
		exp2, err := ecdsa.ParseRawPrivateKey(elliptic.P384(), buf[1:49])
		if err != nil {
			t.Fatalf("ecdsa.ParseRawPrivateKey: %v", err)
		}

		seen := [2]bool{}
		for i := 0; i < 1000; i++ {
			r := bytes.NewReader(buf)
			got, err := ecdsa.GenerateKey(elliptic.P384(), r)
			if err != nil {
				t.Fatalf("ecdsa.GenerateKey: %v", err)
			}
			switch {
			case got.Equal(exp1):
				seen[0] = true
			case got.Equal(exp2):
				seen[1] = true
			default:
				t.Fatalf("ecdsa.GenerateKey with custom reader = %x; want %x or %x", got.D.Bytes(), exp1.D.Bytes(), exp2.D.Bytes())
			}
			if seen[0] && seen[1] {
				break
			}
		}
		if !seen[0] || !seen[1] {
			t.Errorf("ecdsa.GenerateKey with custom reader did not produce both expected keys")
		}

		// Again, with SetGlobalRandom.
		SetGlobalRandom(t, 1)

		seen = [2]bool{}
		for i := 0; i < 1000; i++ {
			r := bytes.NewReader(buf)
			got, err := ecdsa.GenerateKey(elliptic.P384(), r)
			if err != nil {
				t.Fatalf("ecdsa.GenerateKey: %v", err)
			}
			switch {
			case got.Equal(exp1):
				seen[0] = true
			case got.Equal(exp2):
				seen[1] = true
			default:
				t.Fatalf("ecdsa.GenerateKey with custom reader and SetGlobalRandom = %x; want %x or %x", got.D.Bytes(), exp1.D.Bytes(), exp2.D.Bytes())
			}
			if seen[0] && seen[1] {
				break
			}
		}
		if !seen[0] || !seen[1] {
			t.Errorf("ecdsa.GenerateKey with custom reader and SetGlobalRandom did not produce both expected keys")
		}
	})
}
