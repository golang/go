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
)

func usage() {
	fmt.Fprintf(os.Stderr, "usage: go tool nm [-n] [-size] [-sort order] [-type] file...\n")
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
	flag.Usage = usage
	flag.Parse()

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

func errorf(format string, args ...interface{}) {
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

	syms, err := f.Symbols()
	if err != nil {
		errorf("reading %s: %v", file, err)
	}
	if len(syms) == 0 {
		errorf("reading %s: no symbols", file)
	}

	switch *sortOrder {
	case "address":
		sort.Sort(byAddr(syms))
	case "name":
		sort.Sort(byName(syms))
	case "size":
		sort.Sort(bySize(syms))
	}

	w := bufio.NewWriter(os.Stdout)
	for _, sym := range syms {
		if filePrefix {
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
	w.Flush()
}

type byAddr []objfile.Sym

func (x byAddr) Len() int           { return len(x) }
func (x byAddr) Swap(i, j int)      { x[i], x[j] = x[j], x[i] }
func (x byAddr) Less(i, j int) bool { return x[i].Addr < x[j].Addr }

type byName []objfile.Sym

func (x byName) Len() int           { return len(x) }
func (x byName) Swap(i, j int)      { x[i], x[j] = x[j], x[i] }
func (x byName) Less(i, j int) bool { return x[i].Name < x[j].Name }

type bySize []objfile.Sym

func (x bySize) Len() int           { return len(x) }
func (x bySize) Swap(i, j int)      { x[i], x[j] = x[j], x[i] }
func (x bySize) Less(i, j int) bool { return x[i].Size > x[j].Size }
