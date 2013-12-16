// Copyright 2013 The Go Authors.  All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

package main

import (
	"bufio"
	"bytes"
	"flag"
	"fmt"
	"io"
	"log"
	"os"
	"sort"
)

func usage() {
	fmt.Fprintf(os.Stderr, "usage: go tool nm [-n] [-size] [-sort order] [-type] file...\n")
	os.Exit(2)
}

var (
	sortOrder = flag.String("sort", "name", "")
	printSize = flag.Bool("size", false, "")
	printType = flag.Bool("type", false, "")
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
	case "address", "name", "none":
		// ok
	default:
		fmt.Fprintf(os.Stderr, "nm: unknown sort order %q\n", *sortOrder)
		os.Exit(2)
	}

	args := flag.Args()
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

type Sym struct {
	Addr uint64
	Size int64
	Code rune
	Name string
	Type string
}

var parsers = []struct {
	prefix []byte
	parse  func(*os.File) []Sym
}{
	{[]byte("!<arch>\n"), goobjSymbols},
	{[]byte("go object "), goobjSymbols},
	{[]byte("\x7FELF"), elfSymbols},
	{[]byte("\xFE\xED\xFA\xCE"), machoSymbols},
	{[]byte("\xFE\xED\xFA\xCF"), machoSymbols},
	{[]byte("\xCE\xFA\xED\xFE"), machoSymbols},
	{[]byte("\xCF\xFA\xED\xFE"), machoSymbols},
	{[]byte("MZ"), peSymbols},
}

func nm(file string) {
	f, err := os.Open(file)
	if err != nil {
		errorf("%v", err)
		return
	}
	defer f.Close()

	buf := make([]byte, 16)
	io.ReadFull(f, buf)
	f.Seek(0, 0)

	var syms []Sym
	for _, p := range parsers {
		if bytes.HasPrefix(buf, p.prefix) {
			syms = p.parse(f)
			goto HaveSyms
		}
	}
	errorf("%v: unknown file format", file)
	return

HaveSyms:
	switch *sortOrder {
	case "address":
		sort.Sort(byAddr(syms))
	case "name":
		sort.Sort(byName(syms))
	}

	w := bufio.NewWriter(os.Stdout)
	for _, sym := range syms {
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

func filter(syms []Sym, ok func(Sym) bool) []Sym {
	out := syms[:0]
	for _, sym := range syms {
		if ok(sym) {
			out = append(out, sym)
		}
	}
	return out
}

type byAddr []Sym

func (x byAddr) Len() int           { return len(x) }
func (x byAddr) Swap(i, j int)      { x[i], x[j] = x[j], x[i] }
func (x byAddr) Less(i, j int) bool { return x[i].Addr < x[j].Addr }

type byName []Sym

func (x byName) Len() int           { return len(x) }
func (x byName) Swap(i, j int)      { x[i], x[j] = x[j], x[i] }
func (x byName) Less(i, j int) bool { return x[i].Name < x[j].Name }
