// Copyright 2024 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

package fips140

import (
	"crypto/internal/fips140"
	"crypto/internal/fips140/check"
	"internal/godebug"
)

var fips140GODEBUG = godebug.New("#fips140")

// Enabled reports whether the cryptography libraries are operating in FIPS
// 140-3 mode.
//
// It can be controlled at runtime using the GODEBUG setting "fips140". If set
// to "on", FIPS 140-3 mode is enabled. If set to "only", non-approved
// cryptography functions will additionally return errors or panic.
//
// This can't be changed after the program has started.
func Enabled() bool {
	godebug := fips140GODEBUG.Value()
	currentlyEnabled := godebug == "on" || godebug == "only" || godebug == "debug"
	if currentlyEnabled != fips140.Enabled {
		panic("crypto/fips140: GODEBUG setting changed after program start")
	}
	if fips140.Enabled && !check.Verified {
		panic("crypto/fips140: FIPS 140-3 mode enabled, but integrity check didn't pass")
	}
	return fips140.Enabled
}
