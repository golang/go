// Copyright 2026 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

// Package sharedtoolentry provides an exit path for the vendored analysis
// driver that can return through a shared-library entry point.
package sharedtoolentry

import (
	"log"
	"os"
)

type exitStatus int

func (s exitStatus) SharedExitCode() int { return int(s) }

// Exit exits a native tool process or unwinds a shared-tool invocation.
func Exit(code int) {
	if enabled {
		panic(exitStatus(code))
	}
	os.Exit(code)
}

// Fatal writes the arguments and exits with status 1.
func Fatal(args ...any) {
	log.Print(args...)
	Exit(1)
}

// Fatalf writes a formatted message and exits with status 1.
func Fatalf(format string, args ...any) {
	log.Printf(format, args...)
	Exit(1)
}
