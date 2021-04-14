// Copyright 2020 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

//go:generate go run generate_zipdata.go

// Package tzdata provides an embedded copy of the timezone database.
// If this package is imported anywhere in the program, then if
// the time package cannot find tzdata files on the system,
// it will use this embedded information.
//
// Importing this package will increase the size of a program by about
// 450 KB.
//
// This package should normally be imported by a program's main package,
// not by a library. Libraries normally shouldn't decide whether to
// include the timezone database in a program.
//
// This package will be automatically imported if you build with
// -tags timetzdata.
package tzdata

// The test for this package is time/tzdata_test.go.

import (
	"errors"
	"syscall"
	_ "unsafe" // for go:linkname
)

// registerLoadFromEmbeddedTZData is defined in package time.
//go:linkname registerLoadFromEmbeddedTZData time.registerLoadFromEmbeddedTZData
func registerLoadFromEmbeddedTZData(func(string) (string, error))

func init() {
	registerLoadFromEmbeddedTZData(loadFromEmbeddedTZData)
}

// get4s returns the little-endian 32-bit value at the start of s.
func get4s(s string) int {
	if len(s) < 4 {
		return 0
	}
	return int(s[0]) | int(s[1])<<8 | int(s[2])<<16 | int(s[3])<<24
}

// get2s returns the little-endian 16-bit value at the start of s.
func get2s(s string) int {
	if len(s) < 2 {
		return 0
	}
	return int(s[0]) | int(s[1])<<8
}

// loadFromEmbeddedTZData returns the contents of the file with the given
// name in an uncompressed zip file, where the contents of the file can
// be found in embeddedTzdata.
// This is similar to time.loadTzinfoFromZip.
func loadFromEmbeddedTZData(name string) (string, error) {
	const (
		zecheader = 0x06054b50
		zcheader  = 0x02014b50
		ztailsize = 22

		zheadersize = 30
		zheader     = 0x04034b50
	)

	z := zipdata

	idx := len(z) - ztailsize
	n := get2s(z[idx+10:])
	idx = get4s(z[idx+16:])

	for i := 0; i < n; i++ {
		// See time.loadTzinfoFromZip for zip entry layout.
		if get4s(z[idx:]) != zcheader {
			break
		}
		meth := get2s(z[idx+10:])
		size := get4s(z[idx+24:])
		namelen := get2s(z[idx+28:])
		xlen := get2s(z[idx+30:])
		fclen := get2s(z[idx+32:])
		off := get4s(z[idx+42:])
		zname := z[idx+46 : idx+46+namelen]
		idx += 46 + namelen + xlen + fclen
		if zname != name {
			continue
		}
		if meth != 0 {
			return "", errors.New("unsupported compression for " + name + " in embedded tzdata")
		}

		// See time.loadTzinfoFromZip for zip per-file header layout.
		idx = off
		if get4s(z[idx:]) != zheader ||
			get2s(z[idx+8:]) != meth ||
			get2s(z[idx+26:]) != namelen ||
			z[idx+30:idx+30+namelen] != name {
			return "", errors.New("corrupt embedded tzdata")
		}
		xlen = get2s(z[idx+28:])
		idx += 30 + namelen + xlen
		return z[idx : idx+size], nil
	}

	return "", syscall.ENOENT
}
