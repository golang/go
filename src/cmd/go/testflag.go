// Copyright 2011 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

package main

import (
	"fmt"
	"os"
	"strconv"
	"strings"
)

// The flag handling part of go test is large and distracting.
// We can't use the flag package because some of the flags from
// our command line are for us, and some are for 6.out, and
// some are for both.

var usageMessage = `Usage of go test:
  -c=false: compile but do not run the test binary
  -file=file_test.go: specify file to use for tests;
      use multiple times for multiple files
  -x=false: print command lines as they are executed

  // These flags can be passed with or without a "test." prefix: -v or -test.v.
  -bench="": passes -test.bench to test
  -benchtime=1: passes -test.benchtime to test
  -cpu="": passes -test.cpu to test
  -cpuprofile="": passes -test.cpuprofile to test
  -memprofile="": passes -test.memprofile to test
  -memprofilerate=0: passes -test.memprofilerate to test
  -parallel=0: passes -test.parallel to test
  -run="": passes -test.run to test
  -short=false: passes -test.short to test
  -timeout=0: passes -test.timeout to test
  -v=false: passes -test.v to test
`

// usage prints a usage message and exits.
func testUsage() {
	fmt.Fprint(os.Stderr, usageMessage)
	exitStatus = 2
	exit()
}

// testFlagSpec defines a flag we know about.
type testFlagSpec struct {
	name       string
	isBool     bool
	passToTest bool // pass to Test
	multiOK    bool // OK to have multiple instances
	present    bool // flag has been seen
}

// testFlagDefn is the set of flags we process.
var testFlagDefn = []*testFlagSpec{
	// local.
	{name: "c", isBool: true},
	{name: "file", multiOK: true},
	{name: "x", isBool: true},

	// passed to 6.out, adding a "test." prefix to the name if necessary: -v becomes -test.v.
	{name: "bench", passToTest: true},
	{name: "benchtime", passToTest: true},
	{name: "cpu", passToTest: true},
	{name: "cpuprofile", passToTest: true},
	{name: "memprofile", passToTest: true},
	{name: "memprofilerate", passToTest: true},
	{name: "parallel", passToTest: true},
	{name: "run", passToTest: true},
	{name: "short", isBool: true, passToTest: true},
	{name: "timeout", passToTest: true},
	{name: "v", isBool: true, passToTest: true},
}

// testFlags processes the command line, grabbing -x and -c, rewriting known flags
// to have "test" before them, and reading the command line for the 6.out.
// Unfortunately for us, we need to do our own flag processing because go test
// grabs some flags but otherwise its command line is just a holding place for
// test.out's arguments.
func testFlags(args []string) (passToTest []string) {
	for i := 0; i < len(args); i++ {
		f, value, extraWord := testFlag(args, i)
		if f == nil {
			passToTest = append(passToTest, args[i])
			continue
		}
		switch f.name {
		case "c":
			setBoolFlag(&testC, value)
		case "x":
			setBoolFlag(&testX, value)
		case "file":
			testFiles = append(testFiles, value)
		}
		if extraWord {
			i++
		}
		if f.passToTest {
			passToTest = append(passToTest, "-test."+f.name+"="+value)
		}
	}
	return
}

// testFlag sees if argument i is a known flag and returns its definition, value, and whether it consumed an extra word.
func testFlag(args []string, i int) (f *testFlagSpec, value string, extra bool) {
	arg := args[i]
	if strings.HasPrefix(arg, "--") { // reduce two minuses to one
		arg = arg[1:]
	}
	switch arg {
	case "-?", "-h", "-help":
		usage()
	}
	if arg == "" || arg[0] != '-' {
		return
	}
	name := arg[1:]
	// If there's already "test.", drop it for now.
	if strings.HasPrefix(name, "test.") {
		name = name[5:]
	}
	equals := strings.Index(name, "=")
	if equals >= 0 {
		value = name[equals+1:]
		name = name[:equals]
	}
	for _, f = range testFlagDefn {
		if name == f.name {
			// Booleans are special because they have modes -x, -x=true, -x=false.
			if f.isBool {
				if equals < 0 { // otherwise, it's been set and will be verified in setBoolFlag
					value = "true"
				} else {
					// verify it parses
					setBoolFlag(new(bool), value)
				}
			} else { // Non-booleans must have a value.
				extra = equals < 0
				if extra {
					if i+1 >= len(args) {
						usage()
					}
					value = args[i+1]
				}
			}
			if f.present && !f.multiOK {
				usage()
			}
			f.present = true
			return
		}
	}
	f = nil
	return
}

// setBoolFlag sets the addressed boolean to the value.
func setBoolFlag(flag *bool, value string) {
	x, err := strconv.ParseBool(value)
	if err != nil {
		fmt.Fprintf(os.Stderr, "go test: illegal bool flag value %s\n", value)
		usage()
	}
	*flag = x
}
