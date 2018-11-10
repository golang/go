// Copyright 2015 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

// +build ignore

// This is a program that works like the GNU c++filt program.
// It's here for testing purposes and as an example.

package main

import (
	"bufio"
	"flag"
	"fmt"
	"io"
	"os"
	"strings"
	"unicode"

	"github.com/ianlancetaylor/demangle"
)

func flagUsage() {
	usage(os.Stderr, 2)
}

func usage(w io.Writer, status int) {
	fmt.Fprintf(w, "Usage: %s [options] [mangled names]\n", os.Args[0])
	flag.CommandLine.SetOutput(w)
	flag.PrintDefaults()
	fmt.Fprintln(w, `Demangled names are displayed to stdout
If a name cannot be demangled it is just echoed to stdout.
If no names are provided on the command line, stdin is read.`)
	os.Exit(status)
}

var stripUnderscore = flag.Bool("_", false, "Ignore first leading underscore")
var noParams = flag.Bool("p", false, "Do not display function argument types")
var noVerbose = flag.Bool("i", false, "Do not show implementation details (if any)")
var help = flag.Bool("h", false, "Display help information")
var debug = flag.Bool("d", false, "Display debugging information for strings on command line")

// Unimplemented c++filt flags:
// -n (opposite of -_)
// -t (demangle types)
// -s (set demangling style)
// -V (print version information)

// Characters considered to be part of a symbol.
const symbolChars = "_$."

func main() {
	flag.Usage = func() { usage(os.Stderr, 1) }
	flag.Parse()

	if *help {
		usage(os.Stdout, 0)
	}

	out := bufio.NewWriter(os.Stdout)

	if flag.NArg() > 0 {
		for _, f := range flag.Args() {
			if *debug {
				a, err := demangle.ToAST(f, options()...)
				if err != nil {
					fmt.Fprintf(os.Stderr, "%s: %v\n", f, err)
				} else {
					fmt.Fprintf(out, "%#v\n", a)
				}
			} else {
				doDemangle(out, f)
			}
			out.WriteByte('\n')
		}
		if err := out.Flush(); err != nil {
			fmt.Fprintln(os.Stderr, err)
			os.Exit(2)
		}
		return
	}

	scanner := bufio.NewScanner(bufio.NewReader(os.Stdin))
	for scanner.Scan() {
		line := scanner.Text()
		start := -1
		for i, c := range line {
			if unicode.IsLetter(c) || unicode.IsNumber(c) || strings.ContainsRune(symbolChars, c) {
				if start < 0 {
					start = i
				}
			} else {
				if start >= 0 {
					doDemangle(out, line[start:i])
				}
				out.WriteRune(c)
				start = -1
			}
		}
		if start >= 0 {
			doDemangle(out, line[start:])
			start = -1
		}
		out.WriteByte('\n')
		if err := out.Flush(); err != nil {
			fmt.Fprintln(os.Stderr, err)
			os.Exit(2)
		}
	}
}

// Demangle a string just as the GNU c++filt program does.
func doDemangle(out *bufio.Writer, name string) {
	skip := 0
	if name[0] == '.' || name[0] == '$' {
		skip++
	}
	if *stripUnderscore && name[skip] == '_' {
		skip++
	}
	result := demangle.Filter(name[skip:], options()...)
	if result == name[skip:] {
		out.WriteString(name)
	} else {
		if name[0] == '.' {
			out.WriteByte('.')
		}
		out.WriteString(result)
	}
}

// options returns the demangling options to use based on the command
// line flags.
func options() []demangle.Option {
	var options []demangle.Option
	if *noParams {
		options = append(options, demangle.NoParams)
	}
	if !*noVerbose {
		options = append(options, demangle.Verbose)
	}
	return options
}
