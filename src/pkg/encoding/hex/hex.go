// Copyright 2009 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

// This package implements hexadecimal encoding and decoding.
package hex

import (
	"os"
	"strconv"
)

const hextable = "0123456789abcdef"

// EncodedLen returns the length of an encoding of n source bytes.
func EncodedLen(n int) int { return n * 2 }

// Encode encodes src into EncodedLen(len(src))
// bytes of dst.  As a convenience, it returns the number
// of bytes written to dst, but this value is always EncodedLen(len(src)).
// Encode implements hexadecimal encoding.
func Encode(dst, src []byte) int {
	for i, v := range src {
		dst[i*2] = hextable[v>>4]
		dst[i*2+1] = hextable[v&0x0f]
	}

	return len(src) * 2
}

// OddLengthInputError results from decoding an odd length slice.
type OddLengthInputError struct{}

func (OddLengthInputError) String() string { return "odd length hex string" }

// InvalidHexCharError results from finding an invalid character in a hex string.
type InvalidHexCharError byte

func (e InvalidHexCharError) String() string {
	return "invalid hex char: " + strconv.Itoa(int(e))
}


func DecodedLen(x int) int { return x / 2 }

// Decode decodes src into DecodedLen(len(src)) bytes, returning the actual
// number of bytes written to dst.
//
// If Decode encounters invalid input, it returns an OddLengthInputError or an
// InvalidHexCharError.
func Decode(dst, src []byte) (int, os.Error) {
	if len(src)%2 == 1 {
		return 0, OddLengthInputError{}
	}

	for i := 0; i < len(src)/2; i++ {
		a, ok := fromHexChar(src[i*2])
		if !ok {
			return 0, InvalidHexCharError(src[i*2])
		}
		b, ok := fromHexChar(src[i*2+1])
		if !ok {
			return 0, InvalidHexCharError(src[i*2+1])
		}
		dst[i] = (a << 4) | b
	}

	return len(src) / 2, nil
}

// fromHexChar converts a hex character into its value and a success flag.
func fromHexChar(c byte) (byte, bool) {
	switch {
	case '0' <= c && c <= '9':
		return c - '0', true
	case 'a' <= c && c <= 'f':
		return c - 'a' + 10, true
	case 'A' <= c && c <= 'F':
		return c - 'A' + 10, true
	}

	return 0, false
}

// EncodeToString returns the hexadecimal encoding of src.
func EncodeToString(src []byte) string {
	dst := make([]byte, EncodedLen(len(src)))
	Encode(dst, src)
	return string(dst)
}

// DecodeString returns the bytes represented by the hexadecimal string s.
func DecodeString(s string) ([]byte, os.Error) {
	src := []byte(s)
	dst := make([]byte, DecodedLen(len(src)))
	_, err := Decode(dst, src)
	if err != nil {
		return nil, err
	}
	return dst, nil
}
