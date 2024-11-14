// Copyright 2024 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

// Package fips implements support for the GOFIPS140 build setting.
//
// The GOFIPS140 build setting controls two aspects of the build:
//
//   - Whether binaries are built to default to running in FIPS-140 mode,
//     meaning whether they default to GODEBUG=fips140=on or =off.
//
//   - Which copy of the crypto/internal/fips source code to use.
//     The default is obviously GOROOT/src/crypto/internal/fips,
//     but earlier snapshots that have differing levels of external
//     validation and certification are stored in GOROOT/lib/fips140
//     and can be substituted into the build instead.
//
// This package provides the logic needed by the rest of the go command
// to make those decisions and implement the resulting policy.
//
// [Init] must be called to initialize the FIPS logic. It may fail and
// call base.Fatalf.
//
// When GOFIPS140=off, [Enabled] returns false, and the build is
// unchanged from its usual behaviors.
//
// When GOFIPS140 is anything else, [Enabled] returns true, and the build
// sets the default GODEBUG to include fips140=on. This will make
// binaries change their behavior at runtime to confirm to various
// FIPS-140 details. [cmd/go/internal/load.defaultGODEBUG] calls
// [fips.Enabled] when preparing the default settings.
//
// For all builds, FIPS code and data is laid out in contiguous regions
// that are conceptually concatenated into a "fips object file" that the
// linker hashes and then binaries can re-hash at startup to detect
// corruption of those symbols. When [Enabled] is true, the link step
// passes -fipso={a.Objdir}/fips.o to the linker to save a copy of the
// fips.o file. Since the first build target always uses a.Objdir set to
// $WORK/b001, a build like
//
//	GOFIPS140=latest go build -work my/binary
//
// will leave fips.o behind in $WORK/b001. Auditors like to be able to
// see that file. Accordingly, when [Enabled] returns true,
// [cmd/go/internal/work.Builder.useCache] arranges never to cache linker
// output, so that the link step always runs, and fips.o is always left
// behind in the link step. If this proves too slow, we could always
// cache fips.o as an extra link output and then restore it when -work is
// set, but we went a very long time never caching link steps at all, so
// not caching them in FIPS mode seems perfectly fine.
//
// When GOFIPS140 is set to something besides off and latest, [Snapshot]
// returns true, indicating that the build should replace the latest copy
// of crypto/internal/fips with an earlier snapshot. The reason to do
// this is to use a copy that has been through additional lab validation
// (an "in-process" module) or NIST certification (a "certified" module).
// This functionality is not yet implemented.
package fips

import (
	"cmd/go/internal/base"
	"cmd/go/internal/cfg"
)

// Init initializes the FIPS settings.
// It must be called before using any other functions in this package.
// If initialization fails, Init calls base.Fatalf.
func Init() {
	if initDone {
		return
	}
	initDone = true
	initVersion()
}

var initDone bool

// checkInit panics if Init has not been called.
func checkInit() {
	if !initDone {
		panic("fips: not initialized")
	}
}

// Version reports the GOFIPS140 version in use,
// which is either "off", "latest", or a version like "v1.2.3".
// If GOFIPS140 is set to an alias like "inprocess" or "certified",
// Version returns the underlying version.
func Version() string {
	checkInit()
	return version
}

// Enabled reports whether FIPS mode is enabled at all.
// That is, it reports whether GOFIPS140 is set to something besides "off".
func Enabled() bool {
	checkInit()
	return version != "off"
}

// Snapshot reports whether FIPS mode is using a source snapshot
// rather than $GOROOT/src/crypto/internal/fips.
// That is, it reports whether GOFIPS140 is set to something besides "latest" or "off".
func Snapshot() bool {
	checkInit()
	return version != "latest" && version != "off"
}

var version string

func initVersion() {
	// For off and latest, use the local source tree.
	v := cfg.GOFIPS140
	if v == "off" || v == "" {
		version = "off"
		return
	}
	if v == "latest" {
		version = "latest"
		return
	}

	base.Fatalf("go: unknown GOFIPS140 version %q", v)
}
