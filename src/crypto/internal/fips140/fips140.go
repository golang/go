// Copyright 2024 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

package fips140

import (
	"crypto/internal/fips140deps/godebug"
	"errors"
	"runtime"
)

var Enabled bool

var debug bool

func init() {
	v := godebug.Value("#fips140")
	switch v {
	case "on", "only":
		Enabled = true
	case "debug":
		Enabled = true
		debug = true
	case "off", "":
	default:
		panic("fips140: unknown GODEBUG setting fips140=" + v)
	}
}

// Supported returns an error if FIPS 140-3 mode can't be enabled.
func Supported() error {
	// Keep this in sync with fipsSupported in cmd/dist/test.go.

	// ASAN disapproves of reading swaths of global memory in fips140/check.
	// One option would be to expose runtime.asanunpoison through
	// crypto/internal/fips140deps and then call it to unpoison the range
	// before reading it, but it is unclear whether that would then cause
	// false negatives. For now, FIPS+ASAN doesn't need to work.
	if asanEnabled {
		return errors.New("FIPS 140-3 mode is incompatible with ASAN")
	}

	// See EnableFIPS in cmd/internal/obj/fips.go for commentary.
	switch {
	case runtime.GOARCH == "wasm",
		runtime.GOOS == "windows" && runtime.GOARCH == "386",
		runtime.GOOS == "windows" && runtime.GOARCH == "arm",
		runtime.GOOS == "openbsd", // due to -fexecute-only, see #70880
		runtime.GOOS == "aix":
		return errors.New("FIPS 140-3 mode is not supported on " + runtime.GOOS + "-" + runtime.GOARCH)
	}

	if boringEnabled {
		return errors.New("FIPS 140-3 mode is incompatible with GOEXPERIMENT=boringcrypto")
	}

	return nil
}

func Name() string {
	return "Go Cryptographic Module"
}

// Version returns the formal version (such as "v1.0") if building against a
// frozen module with GOFIPS140. Otherwise, it returns "latest".
func Version() string {
	// This return value is replaced by mkzip.go, it must not be changed or
	// moved to a different file.
	return "latest" //mkzip:version
}
