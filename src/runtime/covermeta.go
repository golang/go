// Copyright 2022 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

package runtime

import (
	"internal/coverage/rtcov"
	"unsafe"
)

// covMeta is the top-level container for bits of state related to
// code coverage meta-data in the runtime.
var covMeta struct {
	// metaList contains the list of currently registered meta-data
	// blobs for the running program.
	metaList []rtcov.CovMetaBlob

	// pkgMap records mappings from hard-coded package IDs to
	// slots in the covMetaList above.
	pkgMap map[int]int

	// Set to true if we discover a package mapping glitch.
	hardCodedListNeedsUpdating bool
}

func reportErrorInHardcodedList(slot int32, pkgId int32) {
	println("internal error in coverage meta-data tracking:")
	println("encountered bad pkg ID ", pkgId, " at slot ", slot)
	println("list of hard-coded runtime package IDs needs revising.")
	println("[see the comment on the 'rtPkgs' var in ")
	println(" <goroot>/src/internal/coverage/pkid.go]")
	println("registered list:")
	for k, b := range covMeta.metaList {
		print("slot: ", k, " path='", b.PkgPath, "' ")
		if b.PkgID != -1 {
			print(" hard-coded id: ", b.PkgID)
		}
		println("")
	}
	println("remap table:")
	for from, to := range covMeta.pkgMap {
		println("from ", from, " to ", to)
	}
}

// addCovMeta is invoked during package "init" functions by the
// compiler when compiling for coverage instrumentation; here 'p' is a
// meta-data blob of length 'dlen' for the package in question, 'hash'
// is a compiler-computed md5.sum for the blob, 'pkpath' is the
// package path, 'pkid' is the hard-coded ID that the compiler is
// using for the package (or -1 if the compiler doesn't think a
// hard-coded ID is needed), and 'cmode'/'cgran' are the coverage
// counter mode and granularity requested by the user. Return value is
// the ID for the package for use by the package code itself.
func addCovMeta(p unsafe.Pointer, dlen uint32, hash [16]byte, pkpath string, pkid int, cmode uint8, cgran uint8) uint32 {
	slot := len(covMeta.metaList)
	covMeta.metaList = append(covMeta.metaList,
		rtcov.CovMetaBlob{
			P:                  (*byte)(p),
			Len:                dlen,
			Hash:               hash,
			PkgPath:            pkpath,
			PkgID:              pkid,
			CounterMode:        cmode,
			CounterGranularity: cgran,
		})
	if pkid != -1 {
		if covMeta.pkgMap == nil {
			covMeta.pkgMap = make(map[int]int)
		}
		if _, ok := covMeta.pkgMap[pkid]; ok {
			throw("runtime.addCovMeta: coverage package map collision")
		}
		// Record the real slot (position on meta-list) for this
		// package; we'll use the map to fix things up later on.
		covMeta.pkgMap[pkid] = slot
	}

	// ID zero is reserved as invalid.
	return uint32(slot + 1)
}
