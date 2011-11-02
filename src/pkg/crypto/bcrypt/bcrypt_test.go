// Copyright 2011 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

package bcrypt

import (
	"bytes"
	"testing"
)

func TestBcryptingIsEasy(t *testing.T) {
	pass := []byte("mypassword")
	hp, err := GenerateFromPassword(pass, 0)
	if err != nil {
		t.Fatalf("GenerateFromPassword error: %s", err)
	}

	if CompareHashAndPassword(hp, pass) != nil {
		t.Errorf("%v should hash %s correctly", hp, pass)
	}

	notPass := "notthepass"
	err = CompareHashAndPassword(hp, []byte(notPass))
	if err != MismatchedHashAndPasswordError {
		t.Errorf("%v and %s should be mismatched", hp, notPass)
	}
}

func TestBcryptingIsCorrect(t *testing.T) {
	pass := []byte("allmine")
	salt := []byte("XajjQvNhvvRt5GSeFk1xFe")
	expectedHash := []byte("$2a$10$XajjQvNhvvRt5GSeFk1xFeyqRrsxkhBkUiQeg0dt.wU1qD4aFDcga")

	hash, err := bcrypt(pass, 10, salt)
	if err != nil {
		t.Fatalf("bcrypt blew up: %v", err)
	}
	if !bytes.HasSuffix(expectedHash, hash) {
		t.Errorf("%v should be the suffix of %v", hash, expectedHash)
	}

	h, err := newFromHash(expectedHash)
	if err != nil {
		t.Errorf("Unable to parse %s: %v", string(expectedHash), err)
	}

	// This is not the safe way to compare these hashes. We do this only for
	// testing clarity. Use bcrypt.CompareHashAndPassword()
	if err == nil && !bytes.Equal(expectedHash, h.Hash()) {
		t.Errorf("Parsed hash %v should equal %v", h.Hash(), expectedHash)
	}
}

func TestTooLongPasswordsWork(t *testing.T) {
	salt := []byte("XajjQvNhvvRt5GSeFk1xFe")
	// One byte over the usual 56 byte limit that blowfish has
	tooLongPass := []byte("012345678901234567890123456789012345678901234567890123456")
	tooLongExpected := []byte("$2a$10$XajjQvNhvvRt5GSeFk1xFe5l47dONXg781AmZtd869sO8zfsHuw7C")
	hash, err := bcrypt(tooLongPass, 10, salt)
	if err != nil {
		t.Fatalf("bcrypt blew up on long password: %v", err)
	}
	if !bytes.HasSuffix(tooLongExpected, hash) {
		t.Errorf("%v should be the suffix of %v", hash, tooLongExpected)
	}
}

type InvalidHashTest struct {
	err  error
	hash []byte
}

var invalidTests = []InvalidHashTest{
	{HashTooShortError, []byte("$2a$10$fooo")},
	{HashTooShortError, []byte("$2a")},
	{HashVersionTooNewError('3'), []byte("$3a$10$sssssssssssssssssssssshhhhhhhhhhhhhhhhhhhhhhhhhhhhhhh")},
	{InvalidHashPrefixError('%'), []byte("%2a$10$sssssssssssssssssssssshhhhhhhhhhhhhhhhhhhhhhhhhhhhhhh")},
	{InvalidCostError(32), []byte("$2a$32$sssssssssssssssssssssshhhhhhhhhhhhhhhhhhhhhhhhhhhhhhh")},
}

func TestInvalidHashErrors(t *testing.T) {
	check := func(name string, expected, err error) {
		if err == nil {
			t.Errorf("%s: Should have returned an error", name)
		}
		if err != nil && err != expected {
			t.Errorf("%s gave err %v but should have given %v", name, err, expected)
		}
	}
	for _, iht := range invalidTests {
		_, err := newFromHash(iht.hash)
		check("newFromHash", iht.err, err)
		err = CompareHashAndPassword(iht.hash, []byte("anything"))
		check("CompareHashAndPassword", iht.err, err)
	}
}

func TestUnpaddedBase64Encoding(t *testing.T) {
	original := []byte{101, 201, 101, 75, 19, 227, 199, 20, 239, 236, 133, 32, 30, 109, 243, 30}
	encodedOriginal := []byte("XajjQvNhvvRt5GSeFk1xFe")

	encoded := base64Encode(original)

	if !bytes.Equal(encodedOriginal, encoded) {
		t.Errorf("Encoded %v should have equaled %v", encoded, encodedOriginal)
	}

	decoded, err := base64Decode(encodedOriginal)
	if err != nil {
		t.Fatalf("base64Decode blew up: %s", err)
	}

	if !bytes.Equal(decoded, original) {
		t.Errorf("Decoded %v should have equaled %v", decoded, original)
	}
}

func TestCost(t *testing.T) {
	if testing.Short() {
		return
	}

	pass := []byte("mypassword")

	for c := 0; c < MinCost; c++ {
		p, _ := newFromPassword(pass, c)
		if p.cost != uint32(DefaultCost) {
			t.Errorf("newFromPassword should default costs below %d to %d, but was %d", MinCost, DefaultCost, p.cost)
		}
	}

	p, _ := newFromPassword(pass, 14)
	if p.cost != 14 {
		t.Errorf("newFromPassword should default cost to 14, but was %d", p.cost)
	}

	hp, _ := newFromHash(p.Hash())
	if p.cost != hp.cost {
		t.Errorf("newFromHash should maintain the cost at %d, but was %d", p.cost, hp.cost)
	}

	_, err := newFromPassword(pass, 32)
	if err == nil {
		t.Fatalf("newFromPassword: should return a cost error")
	}
	if err != InvalidCostError(32) {
		t.Errorf("newFromPassword: should return cost error, got %#v", err)
	}
}

func TestCostReturnsWithLeadingZeroes(t *testing.T) {
	hp, _ := newFromPassword([]byte("abcdefgh"), 7)
	cost := hp.Hash()[4:7]
	expected := []byte("07$")

	if !bytes.Equal(expected, cost) {
		t.Errorf("single digit costs in hash should have leading zeros: was %v instead of %v", cost, expected)
	}
}

func TestMinorNotRequired(t *testing.T) {
	noMinorHash := []byte("$2$10$XajjQvNhvvRt5GSeFk1xFeyqRrsxkhBkUiQeg0dt.wU1qD4aFDcga")
	h, err := newFromHash(noMinorHash)
	if err != nil {
		t.Fatalf("No minor hash blew up: %s", err)
	}
	if h.minor != 0 {
		t.Errorf("Should leave minor version at 0, but was %d", h.minor)
	}

	if !bytes.Equal(noMinorHash, h.Hash()) {
		t.Errorf("Should generate hash %v, but created %v", noMinorHash, h.Hash())
	}
}

func BenchmarkEqual(b *testing.B) {
	b.StopTimer()
	passwd := []byte("somepasswordyoulike")
	hash, _ := GenerateFromPassword(passwd, 10)
	b.StartTimer()
	for i := 0; i < b.N; i++ {
		CompareHashAndPassword(hash, passwd)
	}
}

func BenchmarkGeneration(b *testing.B) {
	b.StopTimer()
	passwd := []byte("mylongpassword1234")
	b.StartTimer()
	for i := 0; i < b.N; i++ {
		GenerateFromPassword(passwd, 10)
	}
}
