// Copyright 2011 The Go Authors.  All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.
package main

import (
	"runtime"
	"testing"
)

var (
	thisOS    = runtime.GOOS
	thisArch  = runtime.GOARCH
	otherOS   = "freebsd"
	otherArch = "arm"
)

func init() {
	if thisOS == otherOS {
		otherOS = "linux"
	}
	if thisArch == otherArch {
		otherArch = "amd64"
	}
}

type GoodFileTest struct {
	name   string
	result bool
}

var tests = []GoodFileTest{
	{"file.go", true},
	{"file.c", true},
	{"file_foo.go", true},
	{"file_" + thisArch + ".go", true},
	{"file_" + otherArch + ".go", false},
	{"file_" + thisOS + ".go", true},
	{"file_" + otherOS + ".go", false},
	{"file_" + thisOS + "_" + thisArch + ".go", true},
	{"file_" + otherOS + "_" + thisArch + ".go", false},
	{"file_" + thisOS + "_" + otherArch + ".go", false},
	{"file_" + otherOS + "_" + otherArch + ".go", false},
	{"file_foo_" + thisArch + ".go", true},
	{"file_foo_" + otherArch + ".go", false},
	{"file_" + thisOS + ".c", true},
	{"file_" + otherOS + ".c", false},
}

func TestGoodOSArch(t *testing.T) {
	for _, test := range tests {
		if goodOSArch(test.name) != test.result {
			t.Fatalf("goodOSArch(%q) != %v", test.name, test.result)
		}
	}
}
