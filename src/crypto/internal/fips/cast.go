// Copyright 2024 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

package fips

import (
	"errors"
	"internal/godebug"
	"strings"
	_ "unsafe" // for go:linkname
)

// fatal is [runtime.fatal], pushed via linkname.
//
//go:linkname fatal crypto/internal/fips.fatal
func fatal(string)

// failfipscast is a GODEBUG key allowing simulation of a Cryptographic Algorithm
// Self-Test (CAST) failure, as required during FIPS 140-3 functional testing.
// The value is a substring of the target CAST name.
var failfipscast = godebug.New("#failfipscast")

// testingOnlyCASTHook is called during tests with each CAST name.
var testingOnlyCASTHook func(string)

// CAST runs the named Cryptographic Algorithm Self-Test (if operated in FIPS
// mode) and aborts the program (stopping the module input/output and entering
// the "error state") if the self-test fails.
//
// These are mandatory self-checks that must be performed by FIPS 140-3 modules
// before the algorithm is used. See Implementation Guidance 10.3.A.
//
// The name must not contain commas, colons, hashes, or equal signs.
//
// When calling this function, also add the calling package to cast_external_test.go.
func CAST(name string, f func() error) {
	if strings.ContainsAny(name, ",#=:") {
		panic("fips: invalid self-test name: " + name)
	}
	if testingOnlyCASTHook != nil {
		testingOnlyCASTHook(name)
	}
	if !Enabled {
		return
	}

	err := f()
	if failfipscast.Value() != "" && strings.Contains(name, failfipscast.Value()) {
		err = errors.New("simulated CAST failure")
	}
	if err != nil {
		fatal("FIPS 140-3 self-test failed: " + name + ": " + err.Error())
		panic("unreachable")
	}
}
