// Copyright 2024 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

package gcm_test

import (
	"bytes"
	"crypto/internal/cryptotest"
	"crypto/internal/fips/aes"
	"crypto/internal/fips/aes/gcm"
	"encoding/hex"
	"strings"
	"testing"
)

var sink byte

func TestAllocations(t *testing.T) {
	cryptotest.SkipTestAllocations(t)
	if allocs := testing.AllocsPerRun(10, func() {
		b, err := aes.New(make([]byte, 16))
		if err != nil {
			t.Fatal(err)
		}
		c := gcm.NewCMAC(b)
		sink ^= c.MAC(make([]byte, 16))[0]
	}); allocs > 0 {
		t.Errorf("expected zero allocations, got %0.1f", allocs)
	}
}

func TestCMAC(t *testing.T) {
	// https://csrc.nist.gov/CSRC/media/Projects/Cryptographic-Standards-and-Guidelines/documents/examples/AES_CMAC.pdf
	key := "2B7E1516 28AED2A6 ABF71588 09CF4F3C"
	tests := []struct {
		in, out string
	}{
		{
			"",
			"BB1D6929 E9593728 7FA37D12 9B756746",
		},
		{
			"6BC1BEE2 2E409F96 E93D7E11 7393172A",
			"070A16B4 6B4D4144 F79BDD9D D04A287C",
		},
		{
			"6BC1BEE2 2E409F96 E93D7E11 7393172A AE2D8A57",
			"7D85449E A6EA19C8 23A7BF78 837DFADE",
		},
	}

	b, err := aes.New(decodeHex(t, key))
	if err != nil {
		t.Fatal(err)
	}
	c := gcm.NewCMAC(b)
	for i, test := range tests {
		in := decodeHex(t, test.in)
		out := decodeHex(t, test.out)
		got := c.MAC(in)
		if !bytes.Equal(got[:], out) {
			t.Errorf("test %d: got %x, want %x", i, got, out)
		}
	}
}

func decodeHex(t *testing.T, s string) []byte {
	t.Helper()
	s = strings.ReplaceAll(s, " ", "")
	b, err := hex.DecodeString(s)
	if err != nil {
		t.Fatal(err)
	}
	return b
}
