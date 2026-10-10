// Copyright 2026 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

// Package sharedentry contains process-state setup shared by c-shared command
// entry points. It is only used by files guarded by the sharedtool build tag.
package sharedentry

import (
	"fmt"
	"log"
	"os"
	"runtime/debug"
	"sync"
	"cmd/internal/sharedentry"
)

type exitStatus int

var invokeMu sync.Mutex

// Exit ends a command invocation. In sharedtool builds it unwinds to Invoke,
// which turns the status into the C entry point's return value. Normal command
// builds retain the usual os.Exit behavior.
func Exit(code int) {
	if enabled {
		panic(exitStatus(code))
	}
	sharedentry.Exit(code)
}

// Fatal is the shared-entry equivalent of log.Fatal.
func Fatal(args ...any) {
	log.Print(args...)
	Exit(1)
}

// Fatalf is the shared-entry equivalent of log.Fatalf.
func Fatalf(format string, args ...any) {
	log.Printf(format, args...)
	Exit(1)
}

// Fatalln is the shared-entry equivalent of log.Fatalln.
func Fatalln(args ...any) {
	log.Println(args...)
	Exit(1)
}

// Invoke runs a command with arguments copied from its C caller and converts
// sharedentry.Exit calls into the C function's return value. Each invocation
// gets a fresh copy of os.Args.
func Invoke(args []string, run func()) (status int) {
	invokeMu.Lock()
	defer invokeMu.Unlock()
	os.Args = append([]string(nil), args...)
	defer func() {
		if value := recover(); value != nil {
			if code, ok := value.(exitStatus); ok {
				status = int(code)
				return
			}
			fmt.Fprintf(os.Stderr, "shared tool panic: %v\n", value)
			debug.PrintStack()
			status = 2
		}
	}()
	run()
	return 0
}
