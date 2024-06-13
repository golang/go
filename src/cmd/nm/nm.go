// Copyright 2013 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

package main

import (
	"bufio"
	"flag"
	"fmt"
	"log"
	"os"
	"sort"

	"cmd/internal/objfile"
	"cmd/internal/telemetry"
)

const helpText = `usage: go tool nm [options] file...
  -n
      an alias for -sort address (numeric),
      for compatibility with other nm commands
  -size
      print symbol size in decimal between address and type
  -sort {address,name,none,size}
      sort output in the given order (default name)
      size orders from largest to smallest
  -type
      print symbol type after name
`

func usage() {
	fmt.Fprint(os.Stderr, helpText)
	os.Exit(2)
}

var (
	sortOrder = flag.String("sort", "name", "")
	printSize = flag.Bool("size", false, "")
	printType = flag.Bool("type", false, "")

	filePrefix = false
)

func init() {
	flag.Var(nflag(0), "n", "") // alias for -sort address
}

type nflag int

func (nflag) IsBoolFlag() bool {
	return true
}

func (nflag) Set(value string) error {
	if value == "true" {
		*sortOrder = "address"
	}
	return nil
}

func (nflag) String() string {
	if *sortOrder == "address" {
		return "true"
	}
	return "false"
}

func main() {
	log.SetFlags(0)
	telemetry.OpenCounters()
	flag.Usage = usage
	flag.Parse()
	telemetry.Inc("nm/invocations")
	telemetry.CountFlags("nm/flag:", *flag.CommandLine)

	switch *sortOrder {
	case "address", "name", "none", "size":
		// ok
	default:
		fmt.Fprintf(os.Stderr, "nm: unknown sort order %q\n", *sortOrder)
		os.Exit(2)
	}

	args := flag.Args()
	filePrefix = len(args) > 1
	if len(args) == 0 {
		flag.Usage()
	}

	for _, file := range args {
		nm(file)
	}

	os.Exit(exitCode)
}

var exitCode = 0

func errorf(format string, args ...any) {
	log.Printf(format, args...)
	exitCode = 1
}

func nm(file string) {
	f, err := objfile.Open(file)
	if err != nil {
		errorf("%v", err)
		return
	}
	defer f.Close()

	w := bufio.NewWriter(os.Stdout)

	entries := f.Entries()

	var found bool

	for _, e := range entries {
		syms, err := e.Symbols()
		if err != nil {
			errorf("reading %s: %v", file, err)
		}
		if len(syms) == 0 {
			continue
		}

		found = true

		switch *sortOrder {
		case "address":
			sort.Slice(syms, func(i, j int) bool { return syms[i].Addr < syms[j].Addr })
		case "name":
			sort.Slice(syms, func(i, j int) bool { return syms[i].Name < syms[j].Name })
		case "size":
			sort.Slice(syms, func(i, j int) bool { return syms[i].Size > syms[j].Size })
		}

		for _, sym := range syms {
			if len(entries) > 1 {
				name := e.Name()
				if name == "" {
					fmt.Fprintf(w, "%s(%s):\t", file, "_go_.o")
				} else {
					fmt.Fprintf(w, "%s(%s):\t", file, name)
				}
			} else if filePrefix {
				fmt.Fprintf(w, "%s:\t", file)
			}
			if sym.Code == 'U' {
				fmt.Fprintf(w, "%8s", "")
			} else {
				fmt.Fprintf(w, "%8x", sym.Addr)
			}
			if *printSize {
				fmt.Fprintf(w, " %10d", sym.Size)
			}
			fmt.Fprintf(w, " %c %s", sym.Code, sym.Name)
			if *printType && sym.Type != "" {
				fmt.Fprintf(w, " %s", sym.Type)
			}
			fmt.Fprintf(w, "\n")
		}
	}

	if !found {
		errorf("reading %s: no symbols", file)
	}

	w.Flush()
}
