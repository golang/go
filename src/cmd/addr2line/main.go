// Copyright 2012 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

// Addr2line is a minimal simulation of the GNU addr2line tool,
// just enough to support pprof.
//
// Usage:
//	go tool addr2line binary
//
// Addr2line reads hexadecimal addresses, one per line and with optional 0x prefix,
// from standard input. For each input address, addr2line prints two output lines,
// first the name of the function containing the address and second the file:line
// of the source code corresponding to that address.
//
// This tool is intended for use only by pprof; its interface may change or
// it may be deleted entirely in future releases.
package main

import (
	"bufio"
	"flag"
	"fmt"
	"log"
	"os"
	"strconv"
	"strings"

	"cmd/internal/objfile"
)

func printUsage(w *os.File) {
	fmt.Fprintf(w, "usage: addr2line binary\n")
	fmt.Fprintf(w, "reads addresses from standard input and writes two lines for each:\n")
	fmt.Fprintf(w, "\tfunction name\n")
	fmt.Fprintf(w, "\tfile:line\n")
}

func usage() {
	printUsage(os.Stderr)
	os.Exit(2)
}

func main() {
	log.SetFlags(0)
	log.SetPrefix("addr2line: ")

	// pprof expects this behavior when checking for addr2line
	if len(os.Args) > 1 && os.Args[1] == "--help" {
		printUsage(os.Stdout)
		os.Exit(0)
	}

	flag.Usage = usage
	flag.Parse()
	if flag.NArg() != 1 {
		usage()
	}

	f, err := objfile.Open(flag.Arg(0))
	if err != nil {
		log.Fatal(err)
	}
	defer f.Close()

	tab, err := f.PCLineTable()
	if err != nil {
		log.Fatalf("reading %s: %v", flag.Arg(0), err)
	}

	stdin := bufio.NewScanner(os.Stdin)
	stdout := bufio.NewWriter(os.Stdout)

	for stdin.Scan() {
		p := stdin.Text()
		if strings.Contains(p, ":") {
			// Reverse translate file:line to pc.
			// This was an extension in the old C version of 'go tool addr2line'
			// and is probably not used by anyone, but recognize the syntax.
			// We don't have an implementation.
			fmt.Fprintf(stdout, "!reverse translation not implemented\n")
			continue
		}
		pc, _ := strconv.ParseUint(strings.TrimPrefix(p, "0x"), 16, 64)
		file, line, fn := tab.PCToLine(pc)
		name := "?"
		if fn != nil {
			name = fn.Name
		} else {
			file = "?"
			line = 0
		}
		fmt.Fprintf(stdout, "%s\n%s:%d\n", name, file, line)
	}
	stdout.Flush()
}
