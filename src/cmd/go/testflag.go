// Copyright 2011 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

package main

import (
	"flag"
	"fmt"
	"os"
	"strconv"
	"strings"
)

// The flag handling part of go test is large and distracting.
// We can't use the flag package because some of the flags from
// our command line are for us, and some are for 6.out, and
// some are for both.

// testFlagSpec defines a flag we know about.
type testFlagSpec struct {
	name       string
	boolVar    *bool
	flagValue  flag.Value
	passToTest bool // pass to Test
	multiOK    bool // OK to have multiple instances
	present    bool // flag has been seen
}

// testFlagDefn is the set of flags we process.
var testFlagDefn = []*testFlagSpec{
	// local.
	{name: "c", boolVar: &testC},
	{name: "i", boolVar: &buildI},
	{name: "o"},
	{name: "cover", boolVar: &testCover},
	{name: "covermode"},
	{name: "coverpkg"},
	{name: "exec"},

	// passed to 6.out, adding a "test." prefix to the name if necessary: -v becomes -test.v.
	{name: "bench", passToTest: true},
	{name: "benchmem", boolVar: new(bool), passToTest: true},
	{name: "benchtime", passToTest: true},
	{name: "count", passToTest: true},
	{name: "coverprofile", passToTest: true},
	{name: "cpu", passToTest: true},
	{name: "cpuprofile", passToTest: true},
	{name: "memprofile", passToTest: true},
	{name: "memprofilerate", passToTest: true},
	{name: "blockprofile", passToTest: true},
	{name: "blockprofilerate", passToTest: true},
	{name: "outputdir", passToTest: true},
	{name: "parallel", passToTest: true},
	{name: "run", passToTest: true},
	{name: "short", boolVar: new(bool), passToTest: true},
	{name: "timeout", passToTest: true},
	{name: "trace", passToTest: true},
	{name: "v", boolVar: &testV, passToTest: true},
}

// add build flags to testFlagDefn
func init() {
	var cmd Command
	addBuildFlags(&cmd)
	cmd.Flag.VisitAll(func(f *flag.Flag) {
		if f.Name == "v" {
			// test overrides the build -v flag
			return
		}
		testFlagDefn = append(testFlagDefn, &testFlagSpec{
			name:      f.Name,
			flagValue: f.Value,
		})
	})
}

// testFlags processes the command line, grabbing -x and -c, rewriting known flags
// to have "test" before them, and reading the command line for the 6.out.
// Unfortunately for us, we need to do our own flag processing because go test
// grabs some flags but otherwise its command line is just a holding place for
// pkg.test's arguments.
// We allow known flags both before and after the package name list,
// to allow both
//	go test fmt -custom-flag-for-fmt-test
//	go test -x math
func testFlags(args []string) (packageNames, passToTest []string) {
	inPkg := false
	outputDir := ""
	var explicitArgs []string
	for i := 0; i < len(args); i++ {
		if !strings.HasPrefix(args[i], "-") {
			if !inPkg && packageNames == nil {
				// First package name we've seen.
				inPkg = true
			}
			if inPkg {
				packageNames = append(packageNames, args[i])
				continue
			}
		}

		if inPkg {
			// Found an argument beginning with "-"; end of package list.
			inPkg = false
		}

		f, value, extraWord := testFlag(args, i)
		if f == nil {
			// This is a flag we do not know; we must assume
			// that any args we see after this might be flag
			// arguments, not package names.
			inPkg = false
			if packageNames == nil {
				// make non-nil: we have seen the empty package list
				packageNames = []string{}
			}
			if args[i] == "-args" || args[i] == "--args" {
				// -args or --args signals that everything that follows
				// should be passed to the test.
				explicitArgs = args[i+1:]
				break
			}
			passToTest = append(passToTest, args[i])
			continue
		}
		if f.flagValue != nil {
			if err := f.flagValue.Set(value); err != nil {
				fatalf("invalid flag argument for -%s: %v", f.name, err)
			}
		} else {
			// Test-only flags.
			// Arguably should be handled by f.flagValue, but aren't.
			var err error
			switch f.name {
			// bool flags.
			case "c", "i", "v", "cover":
				setBoolFlag(f.boolVar, value)
			case "o":
				testO = value
				testNeedBinary = true
			case "exec":
				execCmd, err = splitQuotedFields(value)
				if err != nil {
					fatalf("invalid flag argument for -%s: %v", f.name, err)
				}
			case "bench":
				// record that we saw the flag; don't care about the value
				testBench = true
			case "timeout":
				testTimeout = value
			case "blockprofile", "cpuprofile", "memprofile":
				testProfile = true
				testNeedBinary = true
			case "trace":
				testProfile = true
			case "coverpkg":
				testCover = true
				if value == "" {
					testCoverPaths = nil
				} else {
					testCoverPaths = strings.Split(value, ",")
				}
			case "coverprofile":
				testCover = true
				testProfile = true
			case "covermode":
				switch value {
				case "set", "count", "atomic":
					testCoverMode = value
				default:
					fatalf("invalid flag argument for -covermode: %q", value)
				}
				testCover = true
			case "outputdir":
				outputDir = value
			}
		}
		if extraWord {
			i++
		}
		if f.passToTest {
			passToTest = append(passToTest, "-test."+f.name+"="+value)
		}
	}

	if testCoverMode == "" {
		testCoverMode = "set"
		if buildRace {
			// Default coverage mode is atomic when -race is set.
			testCoverMode = "atomic"
		}
	}

	// Tell the test what directory we're running in, so it can write the profiles there.
	if testProfile && outputDir == "" {
		dir, err := os.Getwd()
		if err != nil {
			fatalf("error from os.Getwd: %s", err)
		}
		passToTest = append(passToTest, "-test.outputdir", dir)
	}

	passToTest = append(passToTest, explicitArgs...)
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
	name = strings.TrimPrefix(name, "test.")
	equals := strings.Index(name, "=")
	if equals >= 0 {
		value = name[equals+1:]
		name = name[:equals]
	}
	for _, f = range testFlagDefn {
		if name == f.name {
			// Booleans are special because they have modes -x, -x=true, -x=false.
			if f.boolVar != nil || isBoolFlag(f.flagValue) {
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
						testSyntaxError("missing argument for flag " + f.name)
					}
					value = args[i+1]
				}
			}
			if f.present && !f.multiOK {
				testSyntaxError(f.name + " flag may be set only once")
			}
			f.present = true
			return
		}
	}
	f = nil
	return
}

// isBoolFlag reports whether v is a bool flag.
func isBoolFlag(v flag.Value) bool {
	vv, ok := v.(interface {
		IsBoolFlag() bool
	})
	if ok {
		return vv.IsBoolFlag()
	}
	return false
}

// setBoolFlag sets the addressed boolean to the value.
func setBoolFlag(flag *bool, value string) {
	x, err := strconv.ParseBool(value)
	if err != nil {
		testSyntaxError("illegal bool flag value " + value)
	}
	*flag = x
}

// setIntFlag sets the addressed integer to the value.
func setIntFlag(flag *int, value string) {
	x, err := strconv.Atoi(value)
	if err != nil {
		testSyntaxError("illegal int flag value " + value)
	}
	*flag = x
}

func testSyntaxError(msg string) {
	fmt.Fprintf(os.Stderr, "go test: %s\n", msg)
	fmt.Fprintf(os.Stderr, `run "go help test" or "go help testflag" for more information`+"\n")
	os.Exit(2)
}
