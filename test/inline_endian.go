// errorcheckwithauto -0 -m -d=inlfuncswithclosures=1

//go:build (386 || amd64 || arm64 || ppc64le || s390x) && !gcflags_noopt

// Copyright 2021 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

// Similar to inline.go, but only for architectures that can merge loads.

package foo

import (
	"encoding/binary"
)

// Ensure that simple encoding/binary functions are cheap enough
// that functions using them can also be inlined (issue 42958).
func endian(b []byte) uint64 { // ERROR "can inline endian" "b does not escape"
	return binary.LittleEndian.Uint64(b) + binary.BigEndian.Uint64(b) // ERROR "inlining call to binary.littleEndian.Uint64" "inlining call to binary.bigEndian.Uint64"
}

func appendLittleEndian(b []byte) []byte { // ERROR "can inline appendLittleEndian" "leaking param: b to result ~r0 level=0"
	b = binary.LittleEndian.AppendUint64(b, 64) // ERROR "inlining call to binarylite.littleEndian.AppendUint64"
	b = binary.LittleEndian.AppendUint32(b, 32) // ERROR "inlining call to binarylite.littleEndian.AppendUint32"
	b = binary.LittleEndian.AppendUint16(b, 16) // ERROR "inlining call to binarylite.littleEndian.AppendUint16"
	return b
}

func appendBigEndian(b []byte) []byte { // ERROR "can inline appendBigEndian" "leaking param: b to result ~r0 level=0"
	b = binary.BigEndian.AppendUint64(b, 64) // ERROR "inlining call to binarylite.bigEndian.AppendUint64"
	b = binary.BigEndian.AppendUint32(b, 32) // ERROR "inlining call to binarylite.bigEndian.AppendUint32"
	b = binary.BigEndian.AppendUint16(b, 16) // ERROR "inlining call to binarylite.bigEndian.AppendUint16"
	return b
}
