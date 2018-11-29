// Copyright 2015 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

package ld

import (
	"cmd/link/internal/sym"
	"encoding/binary"
	"fmt"
	"os"
	"time"
)

var startTime time.Time

// TODO(josharian): delete. See issue 19865.
func Cputime() float64 {
	if startTime.IsZero() {
		startTime = time.Now()
	}
	return time.Since(startTime).Seconds()
}

var atExitFuncs []func()

func AtExit(f func()) {
	atExitFuncs = append(atExitFuncs, f)
}

// Exit exits with code after executing all atExitFuncs.
func Exit(code int) {
	for i := len(atExitFuncs) - 1; i >= 0; i-- {
		atExitFuncs[i]()
	}
	os.Exit(code)
}

// Exitf logs an error message then calls Exit(2).
func Exitf(format string, a ...interface{}) {
	fmt.Fprintf(os.Stderr, os.Args[0]+": "+format+"\n", a...)
	nerrors++
	Exit(2)
}

// Errorf logs an error message.
//
// If more than 20 errors have been printed, exit with an error.
//
// Logging an error means that on exit cmd/link will delete any
// output file and return a non-zero error code.
func Errorf(s *sym.Symbol, format string, args ...interface{}) {
	if s != nil {
		format = s.Name + ": " + format
	}
	format += "\n"
	fmt.Fprintf(os.Stderr, format, args...)
	nerrors++
	if *flagH {
		panic("error")
	}
	if nerrors > 20 {
		Exitf("too many errors")
	}
}

func artrim(x []byte) string {
	i := 0
	j := len(x)
	for i < len(x) && x[i] == ' ' {
		i++
	}
	for j > i && x[j-1] == ' ' {
		j--
	}
	return string(x[i:j])
}

func stringtouint32(x []uint32, s string) {
	for i := 0; len(s) > 0; i++ {
		var buf [4]byte
		s = s[copy(buf[:], s):]
		x[i] = binary.LittleEndian.Uint32(buf[:])
	}
}

var start = time.Now()

func elapsed() float64 {
	return time.Since(start).Seconds()
}

// contains reports whether v is in s.
func contains(s []string, v string) bool {
	for _, x := range s {
		if x == v {
			return true
		}
	}
	return false
}
