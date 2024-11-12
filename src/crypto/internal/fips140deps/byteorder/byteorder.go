// Copyright 2024 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

package byteorder

import (
	"internal/byteorder"
)

func LEUint16(b []byte) uint16 {
	return byteorder.LEUint16(b)
}

func BEUint32(b []byte) uint32 {
	return byteorder.BEUint32(b)
}

func BEUint64(b []byte) uint64 {
	return byteorder.BEUint64(b)
}

func LEUint64(b []byte) uint64 {
	return byteorder.LEUint64(b)
}

func BEPutUint16(b []byte, v uint16) {
	byteorder.BEPutUint16(b, v)
}

func BEPutUint32(b []byte, v uint32) {
	byteorder.BEPutUint32(b, v)
}

func BEPutUint64(b []byte, v uint64) {
	byteorder.BEPutUint64(b, v)
}

func LEPutUint64(b []byte, v uint64) {
	byteorder.LEPutUint64(b, v)
}

func BEAppendUint16(b []byte, v uint16) []byte {
	return byteorder.BEAppendUint16(b, v)
}

func BEAppendUint32(b []byte, v uint32) []byte {
	return byteorder.BEAppendUint32(b, v)
}

func BEAppendUint64(b []byte, v uint64) []byte {
	return byteorder.BEAppendUint64(b, v)
}
