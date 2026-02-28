// Copyright 2016 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

//go:build darwin || dragonfly || freebsd || netbsd || openbsd

package routebsd

import "internal/byteorder"

// This file contains duplicates of encoding/binary package.
//
// This package is supposed to be used by the net package of standard
// library. Therefore the package set used in the package must be the
// same as net package.

var (
	littleEndian binaryLittleEndian
	bigEndian    binaryBigEndian
)

type binaryByteOrder interface {
	Uint16([]byte) uint16
	Uint32([]byte) uint32
	Uint64([]byte) uint64
}

type binaryLittleEndian struct{}

func (binaryLittleEndian) Uint16(b []byte) uint16 {
	return byteorder.LEUint16(b)
}

func (binaryLittleEndian) Uint32(b []byte) uint32 {
	return byteorder.LEUint32(b)
}

func (binaryLittleEndian) Uint64(b []byte) uint64 {
	return byteorder.LEUint64(b)
}

type binaryBigEndian struct{}

func (binaryBigEndian) Uint16(b []byte) uint16 {
	return byteorder.BEUint16(b)
}

func (binaryBigEndian) Uint32(b []byte) uint32 {
	return byteorder.BEUint32(b)
}

func (binaryBigEndian) Uint64(b []byte) uint64 {
	return byteorder.BEUint64(b)
}
