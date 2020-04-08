// Copyright 2015 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

// Package flags implements top-level flags and the usage message for the assembler.
package flags

import (
	"cmd/internal/objabi"
	"flag"
	"fmt"
	"os"
	"path/filepath"
	"strings"
)

var (
	Debug      = flag.Bool("debug", false, "dump instructions as they are parsed")
	OutputFile = flag.String("o", "", "output file; default foo.o for /a/b/c/foo.s as first argument")
	PrintOut   = flag.Bool("S", false, "print assembly and machine code")
	TrimPath   = flag.String("trimpath", "", "remove prefix from recorded source file paths")
	Shared     = flag.Bool("shared", false, "generate code that can be linked into a shared library")
	Dynlink    = flag.Bool("dynlink", false, "support references to Go symbols defined in other shared libraries")
	AllErrors  = flag.Bool("e", false, "no limit on number of errors reported")
	SymABIs    = flag.Bool("gensymabis", false, "write symbol ABI information to output file, don't assemble")
	Newobj     = flag.Bool("newobj", false, "use new object file format")

	Spectre = flag.String("spectre", "", "enable spectre mitigations in `list` (all, ret)")
)

var (
	D MultiFlag
	I MultiFlag
)

func init() {
	flag.Var(&D, "D", "predefined symbol with optional simple value -D=identifier=value; can be set multiple times")
	flag.Var(&I, "I", "include directory; can be set multiple times")
	objabi.AddVersionFlag() // -V
}

// MultiFlag allows setting a value multiple times to collect a list, as in -I=dir1 -I=dir2.
type MultiFlag []string

func (m *MultiFlag) String() string {
	if len(*m) == 0 {
		return ""
	}
	return fmt.Sprint(*m)
}

func (m *MultiFlag) Set(val string) error {
	(*m) = append(*m, val)
	return nil
}

func Usage() {
	fmt.Fprintf(os.Stderr, "usage: asm [options] file.s ...\n")
	fmt.Fprintf(os.Stderr, "Flags:\n")
	flag.PrintDefaults()
	os.Exit(2)
}

func Parse() {
	flag.Usage = Usage
	flag.Parse()
	if flag.NArg() == 0 {
		flag.Usage()
	}

	// Flag refinement.
	if *OutputFile == "" {
		if flag.NArg() != 1 {
			flag.Usage()
		}
		input := filepath.Base(flag.Arg(0))
		if strings.HasSuffix(input, ".s") {
			input = input[:len(input)-2]
		}
		*OutputFile = fmt.Sprintf("%s.o", input)
	}
}
