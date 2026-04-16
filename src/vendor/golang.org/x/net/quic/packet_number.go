// Copyright 2023 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

package quic

// A packetNumber is a QUIC packet number.
// Packet numbers are integers in the range [0, 2^62-1].
//
// https://www.rfc-editor.org/rfc/rfc9000.html#section-12.3
type packetNumber int64

const maxPacketNumber = 1<<62 - 1 // https://www.rfc-editor.org/rfc/rfc9000.html#section-17.1-1

// decodePacketNumber decodes a truncated packet number, given
// the largest acknowledged packet number in this number space,
// the truncated number received in a packet, and the size of the
// number received in bytes.
//
// https://www.rfc-editor.org/rfc/rfc9000.html#section-17.1
// https://www.rfc-editor.org/rfc/rfc9000.html#section-a.3
func decodePacketNumber(largest, truncated packetNumber, numLenInBytes int) packetNumber {
	expected := largest + 1
	win := packetNumber(1) << (uint(numLenInBytes) * 8)
	hwin := win / 2
	mask := win - 1
	candidate := (expected &^ mask) | truncated
	if candidate <= expected-hwin && candidate < (1<<62)-win {
		return candidate + win
	}
	if candidate > expected+hwin && candidate >= win {
		return candidate - win
	}
	return candidate
}

// appendPacketNumber appends an encoded packet number to b.
// The packet number must be larger than the largest acknowledged packet number.
// When no packets have been acknowledged yet, largestAck is -1.
//
// https://www.rfc-editor.org/rfc/rfc9000.html#section-17.1-5
func appendPacketNumber(b []byte, pnum, largestAck packetNumber) []byte {
	switch packetNumberLength(pnum, largestAck) {
	case 1:
		return append(b, byte(pnum))
	case 2:
		return append(b, byte(pnum>>8), byte(pnum))
	case 3:
		return append(b, byte(pnum>>16), byte(pnum>>8), byte(pnum))
	default:
		return append(b, byte(pnum>>24), byte(pnum>>16), byte(pnum>>8), byte(pnum))
	}
}

// packetNumberLength returns the minimum length, in bytes, needed to encode
// a packet number given the largest acknowledged packet number.
// The packet number must be larger than the largest acknowledged packet number.
//
// https://www.rfc-editor.org/rfc/rfc9000.html#section-17.1-5
func packetNumberLength(pnum, largestAck packetNumber) int {
	d := pnum - largestAck
	switch {
	case d < 0x80:
		return 1
	case d < 0x8000:
		return 2
	case d < 0x800000:
		return 3
	default:
		return 4
	}
}
