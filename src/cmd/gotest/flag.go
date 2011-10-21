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

// The flag handling part of gotest is large and distracting.
// We can't use the flag package because some of the flags from
// our command line are for us, and some are for 6.out, and
// some are for both.

var usageMessage = `Usage of %s:
  -c=false: compile but do not run the test binary
  -file=file:
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
func usage() {
	fmt.Fprintf(os.Stdout, usageMessage, os.Args[0])
	os.Exit(2)
}

// flagSpec defines a flag we know about.
type flagSpec struct {
	name       string
	isBool     bool
	passToTest bool // pass to Test
	multiOK    bool // OK to have multiple instances
	present    bool // flag has been seen
}

// flagDefn is the set of flags we process.
var flagDefn = []*flagSpec{
	// gotest-local.
	&flagSpec{name: "c", isBool: true},
	&flagSpec{name: "file", multiOK: true},
	&flagSpec{name: "x", isBool: true},

	// passed to 6.out, adding a "test." prefix to the name if necessary: -v becomes -test.v.
	&flagSpec{name: "bench", passToTest: true},
	&flagSpec{name: "benchtime", passToTest: true},
	&flagSpec{name: "cpu", passToTest: true},
	&flagSpec{name: "cpuprofile", passToTest: true},
	&flagSpec{name: "memprofile", passToTest: true},
	&flagSpec{name: "memprofilerate", passToTest: true},
	&flagSpec{name: "parallel", passToTest: true},
	&flagSpec{name: "run", passToTest: true},
	&flagSpec{name: "short", isBool: true, passToTest: true},
	&flagSpec{name: "timeout", passToTest: true},
	&flagSpec{name: "v", isBool: true, passToTest: true},
}

// flags processes the command line, grabbing -x and -c, rewriting known flags
// to have "test" before them, and reading the command line for the 6.out.
// Unfortunately for us, we need to do our own flag processing because gotest
// grabs some flags but otherwise its command line is just a holding place for
// 6.out's arguments.
func flags() {
	for i := 1; i < len(os.Args); i++ {
		arg := os.Args[i]
		f, value, extraWord := flag(i)
		if f == nil {
			args = append(args, arg)
			continue
		}
		switch f.name {
		case "c":
			setBoolFlag(&cFlag, value)
		case "x":
			setBoolFlag(&xFlag, value)
		case "file":
			fileNames = append(fileNames, value)
		}
		if extraWord {
			i++
		}
		if f.passToTest {
			args = append(args, "-test."+f.name+"="+value)
		}
	}
}

// flag sees if argument i is a known flag and returns its definition, value, and whether it consumed an extra word.
func flag(i int) (f *flagSpec, value string, extra bool) {
	arg := os.Args[i]
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
	for _, f = range flagDefn {
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
					if i+1 >= len(os.Args) {
						usage()
					}
					value = os.Args[i+1]
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
	x, err := strconv.Atob(value)
	if err != nil {
		fmt.Fprintf(os.Stderr, "gotest: illegal bool flag value %s\n", value)
		usage()
	}
	*flag = x
}
