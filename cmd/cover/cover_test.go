// Copyright 2013 The Go Authors.  All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

// No testdata on Android.

// +build !android

package main_test

import (
	"bytes"
	"fmt"
	"io/ioutil"
	"os"
	"os/exec"
	"path/filepath"
	"testing"
)

const (
	// Data directory, also the package directory for the test.
	testdata = "testdata"

	// Binaries we compile.
	testcover = "./testcover.exe"
)

var (
	// Files we use.
	testMain    = filepath.Join(testdata, "main.go")
	testTest    = filepath.Join(testdata, "test.go")
	coverInput  = filepath.Join(testdata, "test_line.go")
	coverOutput = filepath.Join(testdata, "test_cover.go")
)

var debug = false // Keeps the rewritten files around if set.

// Run this shell script, but do it in Go so it can be run by "go test".
//
//	replace the word LINE with the line number < testdata/test.go > testdata/test_line.go
// 	go build -o ./testcover
// 	./testcover -mode=count -var=CoverTest -o ./testdata/test_cover.go testdata/test_line.go
//	go run ./testdata/main.go ./testdata/test.go
//
func TestCover(t *testing.T) {
	// Read in the test file (testTest) and write it, with LINEs specified, to coverInput.
	file, err := ioutil.ReadFile(testTest)
	if err != nil {
		t.Fatal(err)
	}
	lines := bytes.Split(file, []byte("\n"))
	for i, line := range lines {
		lines[i] = bytes.Replace(line, []byte("LINE"), []byte(fmt.Sprint(i+1)), -1)
	}
	err = ioutil.WriteFile(coverInput, bytes.Join(lines, []byte("\n")), 0666)
	if err != nil {
		t.Fatal(err)
	}

	// defer removal of test_line.go
	if !debug {
		defer os.Remove(coverInput)
	}

	// go build -o testcover
	cmd := exec.Command("go", "build", "-o", testcover)
	run(cmd, t)

	// defer removal of testcover
	defer os.Remove(testcover)

	// ./testcover -mode=count -var=coverTest -o ./testdata/test_cover.go testdata/test_line.go
	cmd = exec.Command(testcover, "-mode=count", "-var=coverTest", "-o", coverOutput, coverInput)
	run(cmd, t)

	// defer removal of ./testdata/test_cover.go
	if !debug {
		defer os.Remove(coverOutput)
	}

	// go run ./testdata/main.go ./testdata/test.go
	cmd = exec.Command("go", "run", testMain, coverOutput)
	run(cmd, t)
}

func run(c *exec.Cmd, t *testing.T) {
	c.Stdout = os.Stdout
	c.Stderr = os.Stderr
	err := c.Run()
	if err != nil {
		t.Fatal(err)
	}
}
