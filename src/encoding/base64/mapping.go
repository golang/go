// Copyright 2025 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

// Package base64 implements base64 encoding as specified by RFC 4648.
package base64

func (enc Encoding) encodeMapDefault(in uint) byte {
	return enc.encode[in]
}

func (enc Encoding) decodeMapDefault(in uint) byte {
	return enc.decodeMap[in]
}

func StandardBase64Decode(in uint) byte {
	ch := int(in)
	ret := -1

	// if (ch > 0x40 && ch < 0x5b) ret += ch - 0x41 + 1; // -64
	ret += (((0x40 - ch) & (ch - 0x5b)) >> 8) & (ch - 64)

	// if (ch > 0x60 && ch < 0x7b) ret += ch - 0x61 + 26 + 1; // -70
	ret += (((0x60 - ch) & (ch - 0x7b)) >> 8) & (ch - 70)

	// if (ch > 0x2f && ch < 0x3a) ret += ch - 0x30 + 52 + 1; // 5
	ret += (((0x2f - ch) & (ch - 0x3a)) >> 8) & (ch + 5)

	// if (ch == 0x2b) ret += 62 + 1
	ret += (((0x2a - ch) & (ch - 0x2c)) >> 8) & 63

	// if (ch == 0x2f) ret += 63 + 1;
	ret += (((0x2e - ch) & (ch - 0x30)) >> 8) & 64

	return byte(ret)
}

func StandardBase64Encode(in uint) byte {
	src := int(in)
	diff := int(0x41)

	// if (in > 25) diff += 0x61 - 0x41 - 26; // 6
	diff += ((25 - src) >> 8) & 6;

	// if (in > 51) diff += 0x30 - 0x61 - 26; // -75
	diff -= ((51 - src) >> 8) & 75;

	// if (in > 61) diff += 0x2b - 0x30 - 10; // -15
	diff -= ((61 - src) >> 8) & 15;

	// if (in > 62) diff += 0x2f - 0x2b - 1; // 3
	diff += ((62 - src) >> 8) & 3
	return byte(src + diff)
}

func UrlSafeBase64Decode(in uint) byte {
	ch := int(in)
	ret := -1

	// if (ch > 0x40 && ch < 0x5b) ret += ch - 0x41 + 1; // -64
	ret += (((0x40 - ch) & (ch - 0x5b)) >> 8) & (ch - 64)

	// if (ch > 0x60 && ch < 0x7b) ret += ch - 0x61 + 26 + 1; // -70
	ret += (((0x60 - ch) & (ch - 0x7b)) >> 8) & (ch - 70)

	// if (ch > 0x2f && ch < 0x3a) ret += ch - 0x30 + 52 + 1; // 5
	ret += (((0x2f - ch) & (ch - 0x3a)) >> 8) & (ch + 5)

	// if (ch == 0x2c) ret += 62 + 1;
	ret += (((0x2c - ch) & (ch - 0x2e)) >> 8) & 63

	// if (ch == 0x5f) ret += 63 + 1;
	ret += (((0x5e - ch) & (ch - 0x60)) >> 8) & 64

	return byte(ret)
}


func UrlSafeBase64Encode(in uint) byte {
	src := int(in)
	diff := int(0x41)
	// if (src > 25) diff += 0x61 - 0x41 - 26; // 6
	diff += ((25 - src) >> 8) & 6

	// if (src > 51) diff += 0x30 - 0x61 - 26; // -75
	diff -= ((51 - src) >> 8) & 75

	// if (src > 61) diff += 0x2d - 0x30 - 10; // -13
	diff -= ((61 - src) >> 8) & 13

	// if (src > 62) diff += 0x5f - 0x2b - 1; // 3
	diff += ((62 - src) >> 8) & 49
	return byte(src + diff)
}