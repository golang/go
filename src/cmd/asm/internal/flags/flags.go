// Copyright 2015 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

// Package flags implements top-level flags and the usage message for the assembler.
package flags

import (
	"flag"
	"fmt"
	"os"
	"path/filepath"
	"strings"
)

var (
	OutputFile = flag.String("o", "", "output file; default foo.6 for /a/b/c/foo.s on arm64 (unused TODO)")
	PrintOut   = flag.Bool("S", true, "print assembly and machine code") // TODO: set to false
	TrimPath   = flag.String("trimpath", "", "remove prefix from recorded source file paths (unused TODO)")
)

var (
	D MultiFlag
	I MultiFlag
)

func init() {
	flag.Var(&D, "D", "predefined symbol with optional simple value -D=identifer=value; can be set multiple times")
	flag.Var(&I, "I", "include directory; can be set multiple times")
}

// MultiFlag allows setting a value multiple times to collect a list, as in -I=dir1 -I=dir2.
type MultiFlag []string

func (m *MultiFlag) String() string {
	return fmt.Sprint(*m)
}

func (m *MultiFlag) Set(val string) error {
	(*m) = append(*m, val)
	return nil
}

func Usage() {
	fmt.Fprintf(os.Stderr, "usage: asm [options] file.s\n")
	fmt.Fprintf(os.Stderr, "Flags:\n")
	flag.PrintDefaults()
	os.Exit(2)
}

func Parse(goroot, goos, goarch string, theChar int) { // TODO: see below
	flag.Usage = Usage
	flag.Parse()
	if flag.NArg() != 1 {
		flag.Usage()
	}

	// Flag refinement.
	if *OutputFile == "" {
		input := filepath.Base(flag.Arg(0))
		if strings.HasSuffix(input, ".s") {
			input = input[:len(input)-2]
		}
		*OutputFile = fmt.Sprintf("%s.%c", input, theChar)
	}
	// Initialize to include $GOROOT/pkg/$GOOS_GOARCH/ so we find textflag.h
	// TODO: Delete last line once asm is installed because the go command takes care of this.
	// The arguments to Parse can be simplified then too.
	I = append(I, filepath.Join(goroot, "pkg", fmt.Sprintf("%s_%s", goos, goarch)))
}
