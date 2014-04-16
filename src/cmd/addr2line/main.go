// Copyright 2012 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

// Addr2line is a minimal simulation of the GNU addr2line tool,
// just enough to support pprof.
//
// Usage:
//	go tool addr2line binary
//
// Addr2line reads hexadecimal addresses, one per line and without a 0x prefix,
// from standard input. For each input address, addr2line prints two output lines,
// first the name of the function containing the address and second the file:line
// of the source code corresponding to that address.
//
// This tool is intended for use only by pprof; its interface may change or
// it may be deleted entirely in future releases.
package main

import (
	"bufio"
	"debug/elf"
	"debug/gosym"
	"debug/macho"
	"debug/pe"
	"flag"
	"fmt"
	"log"
	"os"
	"strconv"
	"strings"
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

	f, err := os.Open(flag.Arg(0))
	if err != nil {
		log.Fatal(err)
	}

	textStart, symtab, pclntab, err := loadTables(f)
	if err != nil {
		log.Fatalf("reading %s: %v", flag.Arg(0), err)
	}

	pcln := gosym.NewLineTable(pclntab, textStart)
	tab, err := gosym.NewTable(symtab, pcln)
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
		pc, _ := strconv.ParseUint(p, 16, 64)
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

func loadTables(f *os.File) (textStart uint64, symtab, pclntab []byte, err error) {
	if obj, err := elf.NewFile(f); err == nil {
		if sect := obj.Section(".text"); sect != nil {
			textStart = sect.Addr
		}
		if sect := obj.Section(".gosymtab"); sect != nil {
			if symtab, err = sect.Data(); err != nil {
				return 0, nil, nil, err
			}
		}
		if sect := obj.Section(".gopclntab"); sect != nil {
			if pclntab, err = sect.Data(); err != nil {
				return 0, nil, nil, err
			}
		}
		return textStart, symtab, pclntab, nil
	}

	if obj, err := macho.NewFile(f); err == nil {
		if sect := obj.Section("__text"); sect != nil {
			textStart = sect.Addr
		}
		if sect := obj.Section("__gosymtab"); sect != nil {
			if symtab, err = sect.Data(); err != nil {
				return 0, nil, nil, err
			}
		}
		if sect := obj.Section("__gopclntab"); sect != nil {
			if pclntab, err = sect.Data(); err != nil {
				return 0, nil, nil, err
			}
		}
		return textStart, symtab, pclntab, nil
	}

	if obj, err := pe.NewFile(f); err == nil {
		if sect := obj.Section(".text"); sect != nil {
			textStart = uint64(sect.VirtualAddress)
		}
		if sect := obj.Section(".gosymtab"); sect != nil {
			if symtab, err = sect.Data(); err != nil {
				return 0, nil, nil, err
			}
		}
		if sect := obj.Section(".gopclntab"); sect != nil {
			if pclntab, err = sect.Data(); err != nil {
				return 0, nil, nil, err
			}
		}
		return textStart, symtab, pclntab, nil
	}

	return 0, nil, nil, fmt.Errorf("unrecognized binary format")
}
