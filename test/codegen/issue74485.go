// asmcheck

// Copyright 2025 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

package codegen

func divUint64(b uint64) uint64 {
	// amd64:"SHRQ [$]63, AX"
	return b / 9223372036854775808
}

func divUint32(b uint32) uint32 {
	// amd64:"SHRL [$]31, AX"
	return b / 2147483648
}

func divUint16(b uint16) uint16 {
	// amd64:"SHRW [$]15, AX"
	return b / 32768
}

func divUint8(b uint8) uint8 {
	// amd64:"SHRB [$]7, AL"
	return b / 128
}

func modUint64(b uint64) uint64 {
	// amd64:"BTRQ [$]63, AX"
	return b % 9223372036854775808
}

func modUint32(b uint32) uint32 {
	// amd64:"ANDL [$]2147483647, AX"
	return b % 2147483648
}

func modUint16(b uint16) uint16 {
	// amd64:"ANDL [$]32767, AX"
	return b % 32768
}

func modUint8(b uint8) uint8 {
	// amd64:"ANDL [$]127, AX"
	return b % 128
}
