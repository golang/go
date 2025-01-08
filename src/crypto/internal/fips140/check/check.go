// Copyright 2024 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

// Package check implements the FIPS 140 load-time code+data verification.
// Every FIPS package providing cryptographic functionality except hmac and sha256
// must import crypto/internal/fips140/check, so that the verification happens
// before initialization of package global variables.
// The hmac and sha256 packages are used by this package, so they cannot import it.
// Instead, those packages must be careful not to change global variables during init.
// (If necessary, we could have check call a PostCheck function in those packages
// after the check has completed.)
package check

import (
	"crypto/internal/fips140"
	"crypto/internal/fips140/hmac"
	"crypto/internal/fips140/sha256"
	"crypto/internal/fips140deps/byteorder"
	"crypto/internal/fips140deps/godebug"
	"io"
	"unsafe"
)

// Verified is set when verification succeeded. It can be expected to always be
// true when [fips140.Enabled] is true, or init would have panicked.
var Verified bool

// Linkinfo holds the go:fipsinfo symbol prepared by the linker.
// See cmd/link/internal/ld/fips.go for details.
//
//go:linkname Linkinfo go:fipsinfo
var Linkinfo struct {
	Magic [16]byte
	Sum   [32]byte
	Self  uintptr
	Sects [4]struct {
		// Note: These must be unsafe.Pointer, not uintptr,
		// or else checkptr panics about turning uintptrs
		// into pointers into the data segment during
		// go test -race.
		Start unsafe.Pointer
		End   unsafe.Pointer
	}
}

// "\xff"+fipsMagic is the expected linkinfo.Magic.
// We avoid writing that explicitly so that the string does not appear
// elsewhere in normal binaries, just as a precaution.
const fipsMagic = " Go fipsinfo \xff\x00"

var zeroSum [32]byte

func init() {
	if !fips140.Enabled {
		return
	}

	if err := fips140.Supported(); err != nil {
		panic("fips140: " + err.Error())
	}

	if Linkinfo.Magic[0] != 0xff || string(Linkinfo.Magic[1:]) != fipsMagic || Linkinfo.Sum == zeroSum {
		panic("fips140: no verification checksum found")
	}

	h := hmac.New(sha256.New, make([]byte, 32))
	w := io.Writer(h)

	/*
		// Uncomment for debugging.
		// Commented (as opposed to a const bool flag)
		// to avoid import "os" in default builds.
		f, err := os.Create("fipscheck.o")
		if err != nil {
			panic(err)
		}
		w = io.MultiWriter(h, f)
	*/

	w.Write([]byte("go fips object v1\n"))

	var nbuf [8]byte
	for _, sect := range Linkinfo.Sects {
		n := uintptr(sect.End) - uintptr(sect.Start)
		byteorder.BEPutUint64(nbuf[:], uint64(n))
		w.Write(nbuf[:])
		w.Write(unsafe.Slice((*byte)(sect.Start), n))
	}
	sum := h.Sum(nil)

	if [32]byte(sum) != Linkinfo.Sum {
		panic("fips140: verification mismatch")
	}

	// "The temporary value(s) generated during the integrity test of the
	// module’s software or firmware shall [05.10] be zeroised from the module
	// upon completion of the integrity test"
	clear(sum)
	clear(nbuf[:])
	h.Reset()

	if godebug.Value("#fips140") == "debug" {
		println("fips140: verified code+data")
	}

	Verified = true
}
