// Copyright 2024 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

package fips

import (
	"crypto/internal/fipsdeps/godebug"
	"errors"
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
var failfipscast = godebug.Value("#failfipscast")

// CAST runs the named Cryptographic Algorithm Self-Test or Pairwise Consistency
// Test (if operated in FIPS mode) and aborts the program (stopping the module
// input/output and entering the "error state") if the self-test fails.
//
// CASTs are mandatory self-checks that must be performed by FIPS 140-3 modules
// before the algorithm is used. See Implementation Guidance 10.3.A. PCTs are
// mandatory for every key pair that is generated/imported, including ephemeral
// keys (which effectively doubles the cost of key establishment). See
// Implementation Guidance 10.3.A Additional Comment 1.
//
// The name must not contain commas, colons, hashes, or equal signs.
//
// When calling this function from init(), also import the calling package from
// crypto/internal/fipstest, while if calling it from key generation/importing, add
// an invocation to fipstest.TestPCTs.
func CAST(name string, f func() error) {
	if strings.ContainsAny(name, ",#=:") {
		panic("fips: invalid self-test name: " + name)
	}
	if !Enabled {
		return
	}

	err := f()
	if failfipscast != "" && strings.Contains(name, failfipscast) {
		err = errors.New("simulated CAST/PCT failure")
	}
	if err != nil {
		fatal("FIPS 140-3 self-test failed: " + name + ": " + err.Error())
		panic("unreachable")
	}
	if debug {
		println("FIPS 140-3 self-test passed:", name)
	}
}
