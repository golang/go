// Copyright 2022 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

package base

import (
	"cmd/internal/notsha256"
	"fmt"
	"io"
	"os"
	"strings"
	"sync"
)

const GOSSAHASH = "GOSSAHASH"

type writeSyncer interface {
	io.Writer
	Sync() error
}

type HashDebug struct {
	mu sync.Mutex
	// what file (if any) receives the yes/no logging?
	// default is os.Stdout
	logfile writeSyncer
}

var hd HashDebug

// DebugHashMatch reports whether environment variable GOSSAHASH
//
//  1. is empty (this is a special more-quickly implemented case of 3)
//  2. is "y" or "Y"
//  3. is a suffix of the sha1 hash of name
//  4. OR
//      if evname(i) is a suffix of the sha1 hash of name
//      where evname(i)=fmt.Sprintf("GOSSAHASH%d", i),
//      for 0<=i<n such that for all i evname(i) != "" and evname(n) == ""
//
//     That is, as long as they're not empty, try GOSSAHASH, GOSSAHASH0, GOSSAHASH1, etc,
//     but quit trying at the first empty environment variable substitution.
//
// Otherwise it returns false.
// Clause 4 is not really intended for human use.
//
// Unless GOSSAHASH is empty, when DebugHashMatch returns true the message
//
//	"%s triggered %s\n", evname, name
//
// is printed on the file named in environment variable GSHS_LOGFILE,
// or standard out if that is empty.
//
// Typical use:
//
//  1. you make a change to the compiler, say, adding a new phase
//  2. it is broken in some mystifying way, for example, make.bash builds a broken
//     compiler that almost works, but crashes compiling a test in run.bash.
//  3. add this guard to the code, which by default leaves it broken, but
//     does not run the broken new code if GOSSAHASH is non-empty and non-matching:
//
//      if !base.DebugHashMatch(ir.PkgFuncName(fn)) {
//      return nil // early exit, do nothing
//      }
//
//  4. rebuild w/o the bad code, GOSSAHASH=n ./all.bash to verify that you
//     put theguard in the right place with the right sense of the test.
//  5. use github.com/dr2chase/gossahash to search for the error:
//
//      go install github.com/dr2chase/gossahash@latest
//
//      gossahash -- <the thing that fails>
//
//      for example: GOMAXPROCS=1 gossahash -- ./all.bash
//  6. gossahash should return a single function whose miscompilation
//     causes the problem, and you can focus on that.
//
func DebugHashMatch(pkgAndName string) bool {
	return hd.DebugHashMatch(pkgAndName)
}

func (d *HashDebug) DebugHashMatch(pkgAndName string) bool {
	evname := GOSSAHASH
	evhash := os.Getenv(evname)
	hstr := ""

	switch evhash {
	case "":
		return true // default behavior with no EV is "on"
	case "n", "N":
		return false
	}

	// Check the hash of the name against a partial input hash.
	// We use this feature to do a binary search to
	// find a function that is incorrectly compiled.
	for _, b := range notsha256.Sum256([]byte(pkgAndName)) {
		hstr += fmt.Sprintf("%08b", b)
	}

	if evhash == "y" || evhash == "Y" || strings.HasSuffix(hstr, evhash) {
		d.logDebugHashMatch(evname, pkgAndName, hstr)
		return true
	}

	// Iteratively try additional hashes to allow tests for multi-point
	// failure.
	for i := 0; true; i++ {
		ev := fmt.Sprintf("%s%d", evname, i)
		evv := os.Getenv(ev)
		if evv == "" {
			break
		}
		if strings.HasSuffix(hstr, evv) {
			d.logDebugHashMatch(ev, pkgAndName, hstr)
			return true
		}
	}
	return false
}

func (d *HashDebug) logDebugHashMatch(evname, name, hstr string) {
	d.mu.Lock()
	defer d.mu.Unlock()
	file := d.logfile
	if file == nil {
		if tmpfile := os.Getenv("GSHS_LOGFILE"); tmpfile != "" {
			var err error
			file, err = os.OpenFile(tmpfile, os.O_RDWR|os.O_CREATE|os.O_APPEND, 0666)
			if err != nil {
				Fatalf("could not open hash-testing logfile %s", tmpfile)
				return
			}
		}
		if file == nil {
			file = os.Stdout
		}
		d.logfile = file
	}
	if len(hstr) > 24 {
		hstr = hstr[len(hstr)-24:]
	}
	// External tools depend on this string
	fmt.Fprintf(file, "%s triggered %s %s\n", evname, name, hstr)
	file.Sync()
}
