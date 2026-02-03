// Copyright 2024 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

package fips140

import (
	"crypto/internal/fips140"
	"crypto/internal/fips140/check"
)

// Enabled reports whether the cryptography libraries are operating in FIPS
// 140-3 mode.
//
// It can be controlled at runtime using the GODEBUG setting "fips140". If set
// to "on", FIPS 140-3 mode is enabled. If set to "only", non-approved
// cryptography functions will additionally return errors or panic.
//
// This can't be changed after the program has started.
func Enabled() bool {
	if fips140.Enabled && !check.Verified {
		panic("crypto/fips140: FIPS 140-3 mode enabled, but integrity check didn't pass")
	}
	return fips140.Enabled
}

// Version returns the FIPS 140-3 Go Cryptographic Module version (such as
// "v1.0.0"), as referenced in the Security Policy for the module, if building
// against a frozen module with GOFIPS140. Otherwise, it returns "latest". If an
// alias is in use (such as "inprogress") the actual resolved version is
// returned.
//
// The returned version may not uniquely identify the frozen module which was
// used to build the program, if there are multiple copies of the frozen module
// at the same version. The uniquely identifying version suffix can be found by
// checking the value of the GOFIPS140 setting in
// runtime/debug.BuildInfo.Settings.
func Version() string {
	return fips140.Version()
}
