// Copyright 2024 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

package byteorder

import (
	"internal/byteorder"
)

func LEUint16(b []byte) uint16 {
	return byteorder.LeUint16(b)
}

func BEUint32(b []byte) uint32 {
	return byteorder.BeUint32(b)
}

func BEUint64(b []byte) uint64 {
	return byteorder.BeUint64(b)
}

func LEUint64(b []byte) uint64 {
	return byteorder.LeUint64(b)
}

func BEPutUint16(b []byte, v uint16) {
	byteorder.BePutUint16(b, v)
}

func BEPutUint32(b []byte, v uint32) {
	byteorder.BePutUint32(b, v)
}

func BEPutUint64(b []byte, v uint64) {
	byteorder.BePutUint64(b, v)
}

func LEPutUint64(b []byte, v uint64) {
	byteorder.LePutUint64(b, v)
}

func BEAppendUint16(b []byte, v uint16) []byte {
	return byteorder.BeAppendUint16(b, v)
}

func BEAppendUint32(b []byte, v uint32) []byte {
	return byteorder.BeAppendUint32(b, v)
}

func BEAppendUint64(b []byte, v uint64) []byte {
	return byteorder.BeAppendUint64(b, v)
}
