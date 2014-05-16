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
	"debug/elf"
	"debug/gosym"
	"debug/macho"
	"debug/pe"
	"debug/plan9obj"
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
		var imageBase uint64
		switch oh := obj.OptionalHeader.(type) {
		case *pe.OptionalHeader32:
			imageBase = uint64(oh.ImageBase)
		case *pe.OptionalHeader64:
			imageBase = oh.ImageBase
		default:
			return 0, nil, nil, fmt.Errorf("pe file format not recognized")
		}
		if sect := obj.Section(".text"); sect != nil {
			textStart = imageBase + uint64(sect.VirtualAddress)
		}
		if pclntab, err = loadPETable(obj, "pclntab", "epclntab"); err != nil {
			return 0, nil, nil, err
		}
		if symtab, err = loadPETable(obj, "symtab", "esymtab"); err != nil {
			return 0, nil, nil, err
		}
		return textStart, symtab, pclntab, nil
	}

	if obj, err := plan9obj.NewFile(f); err == nil {
		sym, err := findPlan9Symbol(obj, "text")
		if err != nil {
			return 0, nil, nil, err
		}
		textStart = sym.Value
		if pclntab, err = loadPlan9Table(obj, "pclntab", "epclntab"); err != nil {
			return 0, nil, nil, err
		}
		if symtab, err = loadPlan9Table(obj, "symtab", "esymtab"); err != nil {
			return 0, nil, nil, err
		}
		return textStart, symtab, pclntab, nil
	}

	return 0, nil, nil, fmt.Errorf("unrecognized binary format")
}

func findPESymbol(f *pe.File, name string) (*pe.Symbol, error) {
	for _, s := range f.Symbols {
		if s.Name != name {
			continue
		}
		if s.SectionNumber <= 0 {
			return nil, fmt.Errorf("symbol %s: invalid section number %d", name, s.SectionNumber)
		}
		if len(f.Sections) < int(s.SectionNumber) {
			return nil, fmt.Errorf("symbol %s: section number %d is larger than max %d", name, s.SectionNumber, len(f.Sections))
		}
		return s, nil
	}
	return nil, fmt.Errorf("no %s symbol found", name)
}

func loadPETable(f *pe.File, sname, ename string) ([]byte, error) {
	ssym, err := findPESymbol(f, sname)
	if err != nil {
		return nil, err
	}
	esym, err := findPESymbol(f, ename)
	if err != nil {
		return nil, err
	}
	if ssym.SectionNumber != esym.SectionNumber {
		return nil, fmt.Errorf("%s and %s symbols must be in the same section", sname, ename)
	}
	sect := f.Sections[ssym.SectionNumber-1]
	data, err := sect.Data()
	if err != nil {
		return nil, err
	}
	return data[ssym.Value:esym.Value], nil
}

func findPlan9Symbol(f *plan9obj.File, name string) (*plan9obj.Sym, error) {
	syms, err := f.Symbols()
	if err != nil {
		return nil, err
	}
	for _, s := range syms {
		if s.Name != name {
			continue
		}
		return &s, nil
	}
	return nil, fmt.Errorf("no %s symbol found", name)
}

func loadPlan9Table(f *plan9obj.File, sname, ename string) ([]byte, error) {
	ssym, err := findPlan9Symbol(f, sname)
	if err != nil {
		return nil, err
	}
	esym, err := findPlan9Symbol(f, ename)
	if err != nil {
		return nil, err
	}
	text, err := findPlan9Symbol(f, "text")
	if err != nil {
		return nil, err
	}
	sect := f.Section("text")
	if sect == nil {
		return nil, err
	}
	data, err := sect.Data()
	if err != nil {
		return nil, err
	}
	return data[ssym.Value-text.Value : esym.Value-text.Value], nil
}
