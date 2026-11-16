// Copyright 2023 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

package toolchain

import (
	"cmd/go/internal/base"
	"fmt"
	"os"
)

// execGoToolchain execs the Go toolchain with the given name (gotoolchain),
// GOROOT directory, and go command executable.
// The GOROOT directory is empty if we are invoking a command named
// gotoolchain found in $PATH.
func execGoToolchain(gotoolchain, dir, exe string) {
	os.Setenv(targetEnv, gotoolchain)
	if dir == "" {
		os.Unsetenv("GOROOT")
	} else {
		os.Setenv("GOROOT", dir)
	}
	if toolchainTrace {
		if dir == "" {
			fmt.Fprintf(os.Stderr, "go: using %s toolchain located in system PATH (%s)\n", gotoolchain, exe)
		} else {
			fmt.Fprintf(os.Stderr, "go: using %s toolchain from cache located at %s\n", gotoolchain, exe)
		}
	}

	base.Exec(gotoolchain, exe)
}
