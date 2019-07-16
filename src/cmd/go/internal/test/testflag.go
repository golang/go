// Copyright 2011 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

package test

import (
	"flag"
	"os"
	"strings"

	"cmd/go/internal/base"
	"cmd/go/internal/cfg"
	"cmd/go/internal/cmdflag"
	"cmd/go/internal/str"
	"cmd/go/internal/work"
)

const cmd = "test"

// The flag handling part of go test is large and distracting.
// We can't use the flag package because some of the flags from
// our command line are for us, and some are for 6.out, and
// some are for both.

// testFlagDefn is the set of flags we process.
var testFlagDefn = []*cmdflag.Defn{
	// local.
	{Name: "c", BoolVar: &testC},
	{Name: "i", BoolVar: &cfg.BuildI},
	{Name: "o"},
	{Name: "cover", BoolVar: &testCover},
	{Name: "covermode"},
	{Name: "coverpkg"},
	{Name: "exec"},
	{Name: "json", BoolVar: &testJSON},
	{Name: "vet"},

	// Passed to 6.out, adding a "test." prefix to the name if necessary: -v becomes -test.v.
	{Name: "bench", PassToTest: true},
	{Name: "benchmem", BoolVar: new(bool), PassToTest: true},
	{Name: "benchtime", PassToTest: true},
	{Name: "blockprofile", PassToTest: true},
	{Name: "blockprofilerate", PassToTest: true},
	{Name: "count", PassToTest: true},
	{Name: "coverprofile", PassToTest: true},
	{Name: "cpu", PassToTest: true},
	{Name: "cpuprofile", PassToTest: true},
	{Name: "failfast", BoolVar: new(bool), PassToTest: true},
	{Name: "list", PassToTest: true},
	{Name: "memprofile", PassToTest: true},
	{Name: "memprofilerate", PassToTest: true},
	{Name: "mutexprofile", PassToTest: true},
	{Name: "mutexprofilefraction", PassToTest: true},
	{Name: "outputdir", PassToTest: true},
	{Name: "parallel", PassToTest: true},
	{Name: "run", PassToTest: true},
	{Name: "short", BoolVar: new(bool), PassToTest: true},
	{Name: "timeout", PassToTest: true},
	{Name: "trace", PassToTest: true},
	{Name: "v", BoolVar: &testV, PassToTest: true},
}

// add build flags to testFlagDefn
func init() {
	cmdflag.AddKnownFlags("test", testFlagDefn)
	var cmd base.Command
	work.AddBuildFlags(&cmd)
	cmd.Flag.VisitAll(func(f *flag.Flag) {
		if f.Name == "v" {
			// test overrides the build -v flag
			return
		}
		testFlagDefn = append(testFlagDefn, &cmdflag.Defn{
			Name:  f.Name,
			Value: f.Value,
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
func testFlags(usage func(), args []string) (packageNames, passToTest []string) {
	args = str.StringList(cmdflag.FindGOFLAGS(testFlagDefn), args)
	inPkg := false
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

		f, value, extraWord := cmdflag.Parse(cmd, usage, testFlagDefn, args, i)
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
		if f.Value != nil {
			if err := f.Value.Set(value); err != nil {
				base.Fatalf("invalid flag argument for -%s: %v", f.Name, err)
			}
		} else {
			// Test-only flags.
			// Arguably should be handled by f.Value, but aren't.
			switch f.Name {
			// bool flags.
			case "c", "i", "v", "cover", "json":
				cmdflag.SetBool(cmd, f.BoolVar, value)
				if f.Name == "json" && testJSON {
					passToTest = append(passToTest, "-test.v=true")
				}
			case "o":
				testO = value
				testNeedBinary = true
			case "exec":
				xcmd, err := str.SplitQuotedFields(value)
				if err != nil {
					base.Fatalf("invalid flag argument for -%s: %v", f.Name, err)
				}
				work.ExecCmd = xcmd
			case "bench":
				// record that we saw the flag; don't care about the value
				testBench = true
			case "list":
				testList = true
			case "timeout":
				testTimeout = value
			case "blockprofile", "cpuprofile", "memprofile", "mutexprofile":
				testProfile = "-" + f.Name
				testNeedBinary = true
			case "trace":
				testProfile = "-trace"
			case "coverpkg":
				testCover = true
				if value == "" {
					testCoverPaths = nil
				} else {
					testCoverPaths = strings.Split(value, ",")
				}
			case "coverprofile":
				testCover = true
				testCoverProfile = value
			case "covermode":
				switch value {
				case "set", "count", "atomic":
					testCoverMode = value
				default:
					base.Fatalf("invalid flag argument for -covermode: %q", value)
				}
				testCover = true
			case "outputdir":
				testOutputDir = value
			case "vet":
				testVetList = value
			}
		}
		if extraWord {
			i++
		}
		if f.PassToTest {
			passToTest = append(passToTest, "-test."+f.Name+"="+value)
		}
	}

	if testCoverMode == "" {
		testCoverMode = "set"
		if cfg.BuildRace {
			// Default coverage mode is atomic when -race is set.
			testCoverMode = "atomic"
		}
	}

	testVetExplicit = testVetList != ""
	if testVetList != "" && testVetList != "off" {
		if strings.Contains(testVetList, "=") {
			base.Fatalf("-vet argument cannot contain equal signs")
		}
		if strings.Contains(testVetList, " ") {
			base.Fatalf("-vet argument is comma-separated list, cannot contain spaces")
		}
		list := strings.Split(testVetList, ",")
		for i, arg := range list {
			list[i] = "-" + arg
		}
		testVetFlags = list
	}

	if cfg.BuildRace && testCoverMode != "atomic" {
		base.Fatalf(`-covermode must be "atomic", not %q, when -race is enabled`, testCoverMode)
	}

	// Tell the test what directory we're running in, so it can write the profiles there.
	if testProfile != "" && testOutputDir == "" {
		dir, err := os.Getwd()
		if err != nil {
			base.Fatalf("error from os.Getwd: %s", err)
		}
		passToTest = append(passToTest, "-test.outputdir", dir)
	}

	passToTest = append(passToTest, explicitArgs...)
	return
}
