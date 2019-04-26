// Copyright 2017 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

// Functions to access/create device major and minor numbers matching the
// encoding used in NetBSD's sys/types.h header.

package unix

// Major returns the major component of a NetBSD device number.
func Major(dev uint64) uint32 {
	return uint32((dev & 0x000fff00) >> 8)
}

// Minor returns the minor component of a NetBSD device number.
func Minor(dev uint64) uint32 {
	minor := uint32((dev & 0x000000ff) >> 0)
	minor |= uint32((dev & 0xfff00000) >> 12)
	return minor
}

// Mkdev returns a NetBSD device number generated from the given major and minor
// components.
func Mkdev(major, minor uint32) uint64 {
	dev := (uint64(major) << 8) & 0x000fff00
	dev |= (uint64(minor) << 12) & 0xfff00000
	dev |= (uint64(minor) << 0) & 0x000000ff
	return dev
}
