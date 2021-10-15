// Copyright 2018 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

package asmdecl_test

import (
	"os"
	"strings"
	"testing"

	"golang.org/x/tools/go/analysis/analysistest"
	"golang.org/x/tools/go/analysis/passes/asmdecl"
)

var goosarches = []string{
	"linux/amd64", // asm1.s, asm4.s
	"linux/386",   // asm2.s
	"linux/arm",   // asm3.s
	// TODO: skip test on loong64 until go toolchain supported loong64.
	// "linux/loong64", // asm10.s
	"linux/mips64", // asm5.s
	"linux/s390x",  // asm6.s
	"linux/ppc64",  // asm7.s
	"linux/mips",   // asm8.s,
	"js/wasm",      // asm9.s
}

func Test(t *testing.T) {
	testdata := analysistest.TestData()
	for _, goosarch := range goosarches {
		t.Run(goosarch, func(t *testing.T) {
			i := strings.Index(goosarch, "/")
			os.Setenv("GOOS", goosarch[:i])
			os.Setenv("GOARCH", goosarch[i+1:])
			analysistest.Run(t, testdata, asmdecl.Analyzer, "a")
		})
	}
}
