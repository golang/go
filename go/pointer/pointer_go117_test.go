// Copyright 2021 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

// No testdata on Android.

//go:build !android && go1.17
// +build !android,go1.17

package pointer_test

import (
	"fmt"
	"io/ioutil"
	"os"
	"testing"
)

func TestSliceToArrayPointer(t *testing.T) {
	// Based on TestInput. Keep this up to date with that.
	filename := "testdata/arrays_go117.go"

	if testing.Short() {
		t.Skip("skipping in short mode; this test requires tons of memory; https://golang.org/issue/14113")
	}

	wd, err := os.Getwd()
	if err != nil {
		t.Fatalf("os.Getwd: %s", err)
	}
	fmt.Fprintf(os.Stderr, "Entering directory `%s'\n", wd)

	content, err := ioutil.ReadFile(filename)
	if err != nil {
		t.Fatalf("couldn't read file '%s': %s", filename, err)
	}

	if !doOneInput(t, string(content), filename) {
		t.Fail()
	}
}
