// Copyright 2024 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

package fipstest

import (
	"bytes"
	"crypto/internal/fips/aes"
	"crypto/internal/fips/aes/gcm"
	"testing"
)

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
