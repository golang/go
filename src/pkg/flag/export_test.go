// Copyright 2010 The Go Authors.  All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

package flag

import "os"

// Additional routines compiled into the package only during testing.

// ResetForTesting clears all flag state and sets the usage function as directed.
// After calling ResetForTesting, parse errors in flag handling will panic rather
// than exit the program.
func ResetForTesting(usage func()) {
	flags = &allFlags{make(map[string]*Flag), make(map[string]*Flag), os.Args[1:]}
	Usage = usage
	panicOnError = true
}

// ParseForTesting parses the flag state using the provided arguments. It
// should be called after 1) ResetForTesting and 2) setting up the new flags.
// The return value reports whether the parse was error-free.
func ParseForTesting(args []string) (result bool) {
	defer func() {
		if recover() != nil {
			result = false
		}
	}()
	os.Args = args
	Parse()
	return true
}
