// Copyright 2015 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

// run runs the docs tests found in this directory.
package main

import (
	"bytes"
	"flag"
	"fmt"
	"io/ioutil"
	"os"
	"os/exec"
	"path/filepath"
	"regexp"
	"runtime"
	"strings"
)

const usage = `go run run.go [tests]

run.go runs the docs tests in this directory.
If no tests are provided, it runs all tests.
Tests may be specified without their .go suffix.
`

func main() {
	flag.Usage = func() {
		fmt.Fprintf(os.Stderr, usage)
		flag.PrintDefaults()
		os.Exit(2)
	}

	flag.Parse()
	if flag.NArg() == 0 {
		// run all tests
		fixcgo()
	} else {
		// run specified tests
		onlyTest(flag.Args()...)
	}

	tmpdir, err := ioutil.TempDir("", "go-progs")
	if err != nil {
		fmt.Fprintln(os.Stderr, err)
		os.Exit(1)
	}

	// ratec limits the number of tests running concurrently.
	// None of the tests are intensive, so don't bother
	// trying to manually adjust for slow builders.
	ratec := make(chan bool, runtime.NumCPU())
	errc := make(chan error, len(tests))

	for _, tt := range tests {
		tt := tt
		ratec <- true
		go func() {
			errc <- test(tmpdir, tt.file, tt.want)
			<-ratec
		}()
	}

	var rc int
	for range tests {
		if err := <-errc; err != nil {
			fmt.Fprintln(os.Stderr, err)
			rc = 1
		}
	}
	os.Remove(tmpdir)
	os.Exit(rc)
}

// test builds the test in the given file.
// If want is non-empty, test also runs the test
// and checks that the output matches the regexp want.
func test(tmpdir, file, want string) error {
	// Build the program.
	prog := filepath.Join(tmpdir, file)
	cmd := exec.Command("go", "build", "-o", prog, file+".go")
	out, err := cmd.CombinedOutput()
	if err != nil {
		return fmt.Errorf("go build %s.go failed: %v\nOutput:\n%s", file, err, out)
	}
	defer os.Remove(prog)

	// Only run the test if we have output to check.
	if want == "" {
		return nil
	}

	cmd = exec.Command(prog)
	out, err = cmd.CombinedOutput()
	if err != nil {
		return fmt.Errorf("%s failed: %v\nOutput:\n%s", file, err, out)
	}

	// Canonicalize output.
	out = bytes.TrimRight(out, "\n")
	out = bytes.Replace(out, []byte{'\n'}, []byte{' '}, -1)

	// Check the result.
	match, err := regexp.Match(want, out)
	if err != nil {
		return fmt.Errorf("failed to parse regexp %q: %v", want, err)
	}
	if !match {
		return fmt.Errorf("%s.go:\n%q\ndoes not match %s", file, out, want)
	}

	return nil
}

type testcase struct {
	file string
	want string
}

var tests = []testcase{
	// defer_panic_recover
	{"defer", `^0 3210 2$`},
	{"defer2", `^Calling g. Printing in g 0 Printing in g 1 Printing in g 2 Printing in g 3 Panicking! Defer in g 3 Defer in g 2 Defer in g 1 Defer in g 0 Recovered in f 4 Returned normally from f.$`},

	// effective_go
	{"eff_bytesize", `^1.00YB 9.09TB$`},
	{"eff_qr", ""},
	{"eff_sequence", `^\[-1 2 6 16 44\]$`},
	{"eff_unused2", ""},

	// error_handling
	{"error", ""},
	{"error2", ""},
	{"error3", ""},
	{"error4", ""},

	// law_of_reflection
	{"interface", ""},
	{"interface2", `^type: float64$`},

	// c_go_cgo
	{"cgo1", ""},
	{"cgo2", ""},
	{"cgo3", ""},
	{"cgo4", ""},

	// timeout
	{"timeout1", ""},
	{"timeout2", ""},

	// gobs
	{"gobs1", ""},
	{"gobs2", ""},

	// json
	{"json1", `^$`},
	{"json2", `the reciprocal of i is`},
	{"json3", `Age is int 6`},
	{"json4", `^$`},
	{"json5", ""},

	// image_package
	{"image_package1", `^X is 2 Y is 1$`},
	{"image_package2", `^3 4 false$`},
	{"image_package3", `^3 4 true$`},
	{"image_package4", `^image.Point{X:2, Y:1}$`},
	{"image_package5", `^{255 0 0 255}$`},
	{"image_package6", `^8 4 true$`},

	// other
	{"go1", `^Christmas is a holiday: true .*go1.go already exists$`},
	{"slices", ""},
}

func onlyTest(files ...string) {
	var new []testcase
NextFile:
	for _, file := range files {
		file = strings.TrimSuffix(file, ".go")
		for _, tt := range tests {
			if tt.file == file {
				new = append(new, tt)
				continue NextFile
			}
		}
		fmt.Fprintf(os.Stderr, "test %s.go not found\n", file)
		os.Exit(1)
	}
	tests = new
}

func skipTest(file string) {
	for i, tt := range tests {
		if tt.file == file {
			copy(tests[i:], tests[i+1:])
			tests = tests[:len(tests)-1]
			return
		}
	}
	panic("delete(" + file + "): not found")
}

func fixcgo() {
	if os.Getenv("CGO_ENABLED") != "1" {
		skipTest("cgo1")
		skipTest("cgo2")
		skipTest("cgo3")
		skipTest("cgo4")
		return
	}

	switch runtime.GOOS {
	case "freebsd":
		// cgo1 and cgo2 don't run on freebsd, srandom has a different signature
		skipTest("cgo1")
		skipTest("cgo2")
	case "netbsd":
		// cgo1 and cgo2 don't run on netbsd, srandom has a different signature
		skipTest("cgo1")
		skipTest("cgo2")
		// cgo3 and cgo4 don't run on netbsd, since cgo cannot handle stdout correctly, see issue #10715.
		skipTest("cgo3")
		skipTest("cgo4")
	case "openbsd", "solaris":
		// cgo3 and cgo4 don't run on openbsd and solaris, since cgo cannot handle stdout correctly, see issue #10715.
		skipTest("cgo3")
		skipTest("cgo4")
	}
}
