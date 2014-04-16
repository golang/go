// Copyright 2012 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

// Objdump is a minimal simulation of the GNU objdump tool,
// just enough to support pprof.
//
// Usage:
//	go tool objdump binary start end
//
// Objdump disassembles the binary starting at the start address and
// stopping at the end address. The start and end addresses are program
// counters written in hexadecimal without a leading 0x prefix.
//
// It prints a sequence of stanzas of the form:
//
//	file:line
//	 address: assembly
//	 address: assembly
//	 ...
//
// Each stanza gives the disassembly for a contiguous range of addresses
// all mapped to the same original source file and line number.
//
// The disassembler is missing (golang.org/issue/7452) but will be added
// before the Go 1.3 release.
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
)

func printUsage(w *os.File) {
	fmt.Fprintf(w, "usage: objdump binary start end\n")
	fmt.Fprintf(w, "disassembles binary from start PC to end PC.\n")
	fmt.Fprintf(w, "start and end are hexadecimal numbers with no 0x prefix.\n")
}

func usage() {
	printUsage(os.Stderr)
	os.Exit(2)
}

func main() {
	log.SetFlags(0)
	log.SetPrefix("objdump: ")

	flag.Usage = usage
	flag.Parse()
	if flag.NArg() != 3 {
		usage()
	}

	f, err := os.Open(flag.Arg(0))
	if err != nil {
		log.Fatal(err)
	}

	textStart, textData, symtab, pclntab, err := loadTables(f)
	if err != nil {
		log.Fatalf("reading %s: %v", flag.Arg(0), err)
	}

	pcln := gosym.NewLineTable(pclntab, textStart)
	tab, err := gosym.NewTable(symtab, pcln)
	if err != nil {
		log.Fatalf("reading %s: %v", flag.Arg(0), err)
	}

	start, err := strconv.ParseUint(flag.Arg(1), 0, 64)
	if err != nil {
		log.Fatalf("invalid start PC: %v", err)
	}
	end, err := strconv.ParseUint(flag.Arg(2), 0, 64)
	if err != nil {
		log.Fatalf("invalid end PC: %v", err)
	}

	stdout := bufio.NewWriter(os.Stdout)

	// For now, find spans of same PC/line/fn and
	// emit them as having dummy instructions.
	var (
		spanPC   uint64
		spanFile string
		spanLine int
		spanFn   *gosym.Func
	)

	flush := func(endPC uint64) {
		if spanPC == 0 {
			return
		}
		fmt.Fprintf(stdout, "%s:%d\n", spanFile, spanLine)
		for pc := spanPC; pc < endPC; pc++ {
			// TODO(rsc): Disassemble instructions here.
			if textStart <= pc && pc-textStart < uint64(len(textData)) {
				fmt.Fprintf(stdout, " %x: byte %#x\n", pc, textData[pc-textStart])
			} else {
				fmt.Fprintf(stdout, " %x: ?\n", pc)
			}
		}
		spanPC = 0
	}

	for pc := start; pc < end; pc++ {
		file, line, fn := tab.PCToLine(pc)
		if file != spanFile || line != spanLine || fn != spanFn {
			flush(pc)
			spanPC, spanFile, spanLine, spanFn = pc, file, line, fn
		}
	}
	flush(end)

	stdout.Flush()
}

func loadTables(f *os.File) (textStart uint64, textData, symtab, pclntab []byte, err error) {
	if obj, err := elf.NewFile(f); err == nil {
		if sect := obj.Section(".text"); sect != nil {
			textStart = sect.Addr
			textData, _ = sect.Data()
		}
		if sect := obj.Section(".gosymtab"); sect != nil {
			if symtab, err = sect.Data(); err != nil {
				return 0, nil, nil, nil, err
			}
		}
		if sect := obj.Section(".gopclntab"); sect != nil {
			if pclntab, err = sect.Data(); err != nil {
				return 0, nil, nil, nil, err
			}
		}
		return textStart, textData, symtab, pclntab, nil
	}

	if obj, err := macho.NewFile(f); err == nil {
		if sect := obj.Section("__text"); sect != nil {
			textStart = sect.Addr
			textData, _ = sect.Data()
		}
		if sect := obj.Section("__gosymtab"); sect != nil {
			if symtab, err = sect.Data(); err != nil {
				return 0, nil, nil, nil, err
			}
		}
		if sect := obj.Section("__gopclntab"); sect != nil {
			if pclntab, err = sect.Data(); err != nil {
				return 0, nil, nil, nil, err
			}
		}
		return textStart, textData, symtab, pclntab, nil
	}

	if obj, err := pe.NewFile(f); err == nil {
		if sect := obj.Section(".text"); sect != nil {
			textStart = uint64(sect.VirtualAddress)
			textData, _ = sect.Data()
		}
		if sect := obj.Section(".gosymtab"); sect != nil {
			if symtab, err = sect.Data(); err != nil {
				return 0, nil, nil, nil, err
			}
		}
		if sect := obj.Section(".gopclntab"); sect != nil {
			if pclntab, err = sect.Data(); err != nil {
				return 0, nil, nil, nil, err
			}
		}
		return textStart, textData, symtab, pclntab, nil
	}

	return 0, nil, nil, nil, fmt.Errorf("unrecognized binary format")
}
