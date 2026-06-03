// Copyright 2026 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

package mlkem_test

import (
	"bytes"
	"crypto/internal/cryptotest/wycheproof"
	"crypto/internal/fips140/mlkem"
	. "crypto/mlkem"
	"crypto/mlkem/mlkemtest"
	"testing"
)

func TestKeyGenWycheproof(t *testing.T) {
	for _, file := range []string{
		// mlkem_512_keygen_seed_test omitted - no ML-KEM 512 support.
		"mlkem_768_keygen_seed_test.json",
		"mlkem_1024_keygen_seed_test.json",
	} {
		var testdata wycheproof.MlkemKeygenSeedTestSchemaJson
		wycheproof.LoadVectorFile(t, file, &testdata)

		for _, tg := range testdata.TestGroups {
			for _, tv := range tg.Tests {
				t.Run(wycheproof.TestName(file, tv), func(t *testing.T) {
					t.Parallel()
					runKeyGenTest(t, tg.ParameterSet, tv)
				})
			}
		}
	}
}

func runKeyGenTest(t *testing.T, paramSet wycheproof.MLKEMKeyGenTestGroupParameterSet, tv wycheproof.MLKEMKeyGenTestGroupTestsElem) {
	t.Helper()

	seed := wycheproof.MustDecodeHex(tv.Seed)
	expectedEk := wycheproof.MustDecodeHex(tv.Ek)
	expectedDk := wycheproof.MustDecodeHex(tv.Dk)

	switch paramSet {
	case wycheproof.MLKEMKeyGenTestGroupParameterSetMLKEM768:
		dk, err := mlkem.NewDecapsulationKey768(seed)
		if err != nil {
			if tv.Result == "valid" {
				t.Fatalf("NewDecapsulationKey768: %v", err)
			}
			return
		}
		if !bytes.Equal(dk.Bytes(), seed) {
			t.Errorf("decapsulation key seed roundtrip mismatch")
		}
		ek := dk.EncapsulationKey()
		if !bytes.Equal(ek.Bytes(), expectedEk) {
			t.Errorf("encapsulation key mismatch")
		}
		if !bytes.Equal(mlkem.TestingOnlyExpandedBytes768(dk), expectedDk) {
			t.Errorf("expanded decapsulation key mismatch")
		}
		ek2, err := mlkem.NewEncapsulationKey768(expectedEk)
		if err != nil {
			t.Fatalf("NewEncapsulationKey768: %v", err)
		}
		if !bytes.Equal(ek2.Bytes(), expectedEk) {
			t.Errorf("encapsulation key roundtrip mismatch")
		}
		k, c := ek.Encapsulate()
		k2, err := dk.Decapsulate(c)
		if err != nil {
			t.Fatalf("Decapsulate: %v", err)
		}
		if !bytes.Equal(k, k2) {
			t.Errorf("encaps/decaps roundtrip key mismatch")
		}

	case wycheproof.MLKEMKeyGenTestGroupParameterSetMLKEM1024:
		dk, err := mlkem.NewDecapsulationKey1024(seed)
		if err != nil {
			if tv.Result == "valid" {
				t.Fatalf("NewDecapsulationKey1024: %v", err)
			}
			return
		}
		if !bytes.Equal(dk.Bytes(), seed) {
			t.Errorf("decapsulation key seed roundtrip mismatch")
		}
		ek := dk.EncapsulationKey()
		if !bytes.Equal(ek.Bytes(), expectedEk) {
			t.Errorf("encapsulation key mismatch")
		}
		if !bytes.Equal(mlkem.TestingOnlyExpandedBytes1024(dk), expectedDk) {
			t.Errorf("expanded decapsulation key mismatch")
		}
		ek2, err := mlkem.NewEncapsulationKey1024(expectedEk)
		if err != nil {
			t.Fatalf("NewEncapsulationKey1024: %v", err)
		}
		if !bytes.Equal(ek2.Bytes(), expectedEk) {
			t.Errorf("encapsulation key roundtrip mismatch")
		}
		k, c := ek.Encapsulate()
		k2, err := dk.Decapsulate(c)
		if err != nil {
			t.Fatalf("Decapsulate: %v", err)
		}
		if !bytes.Equal(k, k2) {
			t.Errorf("encaps/decaps roundtrip key mismatch")
		}

	default:
		t.Fatalf("parameter set %s unsupported", paramSet)
	}
}

func TestMLKEMEncapsWycheproof(t *testing.T) {
	for _, file := range []string{
		// mlkem_512_encaps_test omitted - no ML-KEM 512 support.
		"mlkem_768_encaps_test.json",
		"mlkem_1024_encaps_test.json",
	} {
		var testdata wycheproof.MlkemEncapsTestSchemaJson
		wycheproof.LoadVectorFile(t, file, &testdata)

		for _, tg := range testdata.TestGroups {
			for _, tv := range tg.Tests {
				t.Run(wycheproof.TestName(file, tv), func(t *testing.T) {
					t.Parallel()
					runEncapsTest(t, tg.ParameterSet, tv)
				})
			}
		}
	}
}

func runEncapsTest(t *testing.T, paramSet wycheproof.MLKEMEncapsTestGroupParameterSet, tv wycheproof.MLKEMEncapsTestGroupTestsElem) {
	t.Helper()

	shouldPass := wycheproof.ShouldPass(t, tv.Result, tv.Flags, nil)
	ekBytes := wycheproof.MustDecodeHex(tv.Ek)
	m := wycheproof.MustDecodeHex(tv.M)
	expectedC := wycheproof.MustDecodeHex(tv.C)
	expectedK := wycheproof.MustDecodeHex(tv.K)

	switch paramSet {
	case wycheproof.MLKEMEncapsTestGroupParameterSetMLKEM768:
		ek, err := NewEncapsulationKey768(ekBytes)
		if err != nil {
			if shouldPass {
				t.Fatalf("NewEncapsulationKey768: %v", err)
			}
			return
		}
		if !bytes.Equal(ek.Bytes(), ekBytes) {
			t.Errorf("encapsulation key roundtrip mismatch")
		}
		k, c, err := mlkemtest.Encapsulate768(ek, m)
		if err != nil {
			if shouldPass {
				t.Fatalf("Encapsulate768: %v", err)
			}
			return
		}
		if !shouldPass {
			t.Errorf("Encapsulate768 unexpectedly succeeded")
			return
		}
		if !bytes.Equal(c, expectedC) {
			t.Errorf("ciphertext mismatch")
		}
		if !bytes.Equal(k, expectedK) {
			t.Errorf("shared key mismatch")
		}

	case wycheproof.MLKEMEncapsTestGroupParameterSetMLKEM1024:
		ek, err := NewEncapsulationKey1024(ekBytes)
		if err != nil {
			if shouldPass {
				t.Fatalf("NewEncapsulationKey1024: %v", err)
			}
			return
		}
		if !bytes.Equal(ek.Bytes(), ekBytes) {
			t.Errorf("encapsulation key roundtrip mismatch")
		}
		k, c, err := mlkemtest.Encapsulate1024(ek, m)
		if err != nil {
			if shouldPass {
				t.Fatalf("Encapsulate1024: %v", err)
			}
			return
		}
		if !shouldPass {
			t.Errorf("Encapsulate1024 unexpectedly succeeded")
			return
		}
		if !bytes.Equal(c, expectedC) {
			t.Errorf("ciphertext mismatch")
		}
		if !bytes.Equal(k, expectedK) {
			t.Errorf("shared key mismatch")
		}

	default:
		t.Fatalf("parameter set %s unsupported", paramSet)
	}
}

func TestMLKEMDecapsWycheproof(t *testing.T) {
	for _, file := range []string{
		// mlkem_512_test omitted - no ML-KEM 512 support.
		"mlkem_768_test.json",
		"mlkem_1024_test.json",
	} {
		var testdata wycheproof.MlkemTestSchemaJson
		wycheproof.LoadVectorFile(t, file, &testdata)

		for _, tg := range testdata.TestGroups {
			for _, tv := range tg.Tests {
				t.Run(wycheproof.TestName(file, tv), func(t *testing.T) {
					t.Parallel()
					runDecapsTest(t, tg.ParameterSet, tv)
				})
			}
		}
	}
}

func runDecapsTest(t *testing.T, paramSet wycheproof.MLKEMTestGroupParameterSet, tv wycheproof.MLKEMTestGroupTestsElem) {
	t.Helper()

	shouldPass := wycheproof.ShouldPass(t, tv.Result, tv.Flags, nil)
	seed := wycheproof.MustDecodeHex(tv.Seed)
	ciphertext := wycheproof.MustDecodeHex(tv.C)
	expectedK := wycheproof.MustDecodeHex(tv.K)

	switch paramSet {
	case wycheproof.MLKEMTestGroupParameterSetMLKEM768:
		dk, err := NewDecapsulationKey768(seed)
		if err != nil {
			if shouldPass {
				t.Fatalf("NewDecapsulationKey768: %v", err)
			}
			return
		}
		if !bytes.Equal(dk.Bytes(), seed) {
			t.Errorf("decapsulation key seed roundtrip mismatch")
		}
		if tv.Ek != nil {
			expectedEk := wycheproof.MustDecodeHex(*tv.Ek)
			if !bytes.Equal(dk.EncapsulationKey().Bytes(), expectedEk) {
				t.Errorf("encapsulation key mismatch")
			}
		}
		k, err := dk.Decapsulate(ciphertext)
		if err != nil {
			if shouldPass {
				t.Fatalf("Decapsulate: %v", err)
			}
			return
		}
		if shouldPass {
			if !bytes.Equal(k, expectedK) {
				t.Errorf("shared key mismatch: got %x, want %x", k, expectedK)
			}
			kFresh, cFresh := dk.EncapsulationKey().Encapsulate()
			kRT, err := dk.Decapsulate(cFresh)
			if err != nil {
				t.Fatalf("Decapsulate of fresh ciphertext: %v", err)
			}
			if !bytes.Equal(kFresh, kRT) {
				t.Errorf("encaps/decaps roundtrip key mismatch")
			}
		}

	case wycheproof.MLKEMTestGroupParameterSetMLKEM1024:
		dk, err := NewDecapsulationKey1024(seed)
		if err != nil {
			if shouldPass {
				t.Fatalf("NewDecapsulationKey1024: %v", err)
			}
			return
		}
		if !bytes.Equal(dk.Bytes(), seed) {
			t.Errorf("decapsulation key seed roundtrip mismatch")
		}
		if tv.Ek != nil {
			expectedEk := wycheproof.MustDecodeHex(*tv.Ek)
			if !bytes.Equal(dk.EncapsulationKey().Bytes(), expectedEk) {
				t.Errorf("encapsulation key mismatch")
			}
		}
		k, err := dk.Decapsulate(ciphertext)
		if err != nil {
			if shouldPass {
				t.Fatalf("Decapsulate: %v", err)
			}
			return
		}
		if shouldPass {
			if !bytes.Equal(k, expectedK) {
				t.Errorf("shared key mismatch: got %x, want %x", k, expectedK)
			}
			kFresh, cFresh := dk.EncapsulationKey().Encapsulate()
			kRT, err := dk.Decapsulate(cFresh)
			if err != nil {
				t.Fatalf("Decapsulate of fresh ciphertext: %v", err)
			}
			if !bytes.Equal(kFresh, kRT) {
				t.Errorf("encaps/decaps roundtrip key mismatch")
			}
		}

	default:
		t.Fatalf("parameter set %s unsupported", paramSet)
	}
}

func TestMLKEMSemiExpandedDecapsWycheproof(t *testing.T) {
	for _, file := range []string{
		// mlkem_512_semi_expanded_decaps_test omitted - no ML-KEM 512 support.
		"mlkem_768_semi_expanded_decaps_test.json",
		"mlkem_1024_semi_expanded_decaps_test.json",
	} {
		var testdata wycheproof.MlkemSemiExpandedDecapsTestSchemaJson
		wycheproof.LoadVectorFile(t, file, &testdata)

		for _, tg := range testdata.TestGroups {
			for _, tv := range tg.Tests {
				t.Run(wycheproof.TestName(file, tv), func(t *testing.T) {
					t.Parallel()
					runSemiExpandedDecapsTest(t, tg.ParameterSet, tv)
				})
			}
		}
	}
}

func runSemiExpandedDecapsTest(t *testing.T, paramSet wycheproof.MLKEMDecapsTestGroupParameterSet, tv wycheproof.MLKEMDecapsTestGroupTestsElem) {
	t.Helper()

	shouldPass := wycheproof.ShouldPass(t, tv.Result, tv.Flags, nil)
	dkBytes := wycheproof.MustDecodeHex(tv.Dk)
	ciphertext := wycheproof.MustDecodeHex(tv.C)

	switch paramSet {
	case wycheproof.MLKEMDecapsTestGroupParameterSetMLKEM768:
		dk, err := mlkem.TestingOnlyNewDecapsulationKey768(dkBytes)
		if err != nil {
			if shouldPass {
				t.Fatalf("TestingOnlyNewDecapsulationKey768: %v", err)
			}
			return
		}
		if !bytes.Equal(mlkem.TestingOnlyExpandedBytes768(dk), dkBytes) {
			t.Errorf("expanded decapsulation key roundtrip mismatch")
		}
		k, err := dk.Decapsulate(ciphertext)
		if err != nil {
			if shouldPass {
				t.Fatalf("Decapsulate: %v", err)
			}
			return
		}
		if !shouldPass {
			t.Errorf("Decapsulate unexpectedly succeeded")
			return
		}
		if len(k) != SharedKeySize {
			t.Errorf("shared key has wrong length: got %d, want %d", len(k), SharedKeySize)
		}
		kFresh, cFresh := dk.EncapsulationKey().Encapsulate()
		kRT, err := dk.Decapsulate(cFresh)
		if err != nil {
			t.Fatalf("Decapsulate of fresh ciphertext: %v", err)
		}
		if !bytes.Equal(kFresh, kRT) {
			t.Errorf("encaps/decaps roundtrip key mismatch")
		}

	case wycheproof.MLKEMDecapsTestGroupParameterSetMLKEM1024:
		dk, err := mlkem.TestingOnlyNewDecapsulationKey1024(dkBytes)
		if err != nil {
			if shouldPass {
				t.Fatalf("TestingOnlyNewDecapsulationKey1024: %v", err)
			}
			return
		}
		if !bytes.Equal(mlkem.TestingOnlyExpandedBytes1024(dk), dkBytes) {
			t.Errorf("expanded decapsulation key roundtrip mismatch")
		}
		k, err := dk.Decapsulate(ciphertext)
		if err != nil {
			if shouldPass {
				t.Fatalf("Decapsulate: %v", err)
			}
			return
		}
		if !shouldPass {
			t.Errorf("Decapsulate unexpectedly succeeded")
			return
		}
		if len(k) != SharedKeySize {
			t.Errorf("shared key has wrong length: got %d, want %d", len(k), SharedKeySize)
		}
		kFresh, cFresh := dk.EncapsulationKey().Encapsulate()
		kRT, err := dk.Decapsulate(cFresh)
		if err != nil {
			t.Fatalf("Decapsulate of fresh ciphertext: %v", err)
		}
		if !bytes.Equal(kFresh, kRT) {
			t.Errorf("encaps/decaps roundtrip key mismatch")
		}

	default:
		t.Fatalf("parameter set %s unsupported", paramSet)
	}
}
