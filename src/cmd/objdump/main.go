// Copyright 2012 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

// Objdump disassembles executable files.
//
// Usage:
//
//	go tool objdump [-s symregexp] binary
//
// Objdump prints a disassembly of all text symbols (code) in the binary.
// If the -s option is present, objdump only disassembles
// symbols with names matching the regular expression.
//
// Alternate usage:
//
//	go tool objdump binary start end
//
// In this mode, objdump disassembles the binary starting at the start address and
// stopping at the end address. The start and end addresses are program
// counters written in hexadecimal with optional leading 0x prefix.
// In this mode, objdump prints a sequence of stanzas of the form:
//
//	file:line
//	 address: assembly
//	 address: assembly
//	 ...
//
// Each stanza gives the disassembly for a contiguous range of addresses
// all mapped to the same original source file and line number.
// This mode is intended for use by pprof.
package main

import (
	"bufio"
	"debug/gosym"
	"encoding/binary"
	"flag"
	"fmt"
	"io"
	"log"
	"os"
	"regexp"
	"sort"
	"strconv"
	"strings"
	"text/tabwriter"

	"cmd/internal/objfile"

	"cmd/internal/rsc.io/arm/armasm"
	"cmd/internal/rsc.io/x86/x86asm"
)

var symregexp = flag.String("s", "", "only dump symbols matching this regexp")
var symRE *regexp.Regexp

func usage() {
	fmt.Fprintf(os.Stderr, "usage: go tool objdump [-s symregexp] binary [start end]\n\n")
	flag.PrintDefaults()
	os.Exit(2)
}

type lookupFunc func(addr uint64) (sym string, base uint64)
type disasmFunc func(code []byte, pc uint64, lookup lookupFunc) (text string, size int)

func main() {
	log.SetFlags(0)
	log.SetPrefix("objdump: ")

	flag.Usage = usage
	flag.Parse()
	if flag.NArg() != 1 && flag.NArg() != 3 {
		usage()
	}

	if *symregexp != "" {
		re, err := regexp.Compile(*symregexp)
		if err != nil {
			log.Fatalf("invalid -s regexp: %v", err)
		}
		symRE = re
	}

	f, err := objfile.Open(flag.Arg(0))
	if err != nil {
		log.Fatal(err)
	}

	syms, err := f.Symbols()
	if err != nil {
		log.Fatalf("reading %s: %v", flag.Arg(0), err)
	}

	tab, err := f.PCLineTable()
	if err != nil {
		log.Fatalf("reading %s: %v", flag.Arg(0), err)
	}

	textStart, textBytes, err := f.Text()
	if err != nil {
		log.Fatalf("reading %s: %v", flag.Arg(0), err)
	}

	goarch := f.GOARCH()

	disasm := disasms[goarch]
	if disasm == nil {
		log.Fatalf("reading %s: unknown architecture", flag.Arg(0))
	}

	// Filter out section symbols, overwriting syms in place.
	keep := syms[:0]
	for _, sym := range syms {
		switch sym.Name {
		case "runtime.text", "text", "_text", "runtime.etext", "etext", "_etext":
			// drop
		default:
			keep = append(keep, sym)
		}
	}
	syms = keep

	sort.Sort(ByAddr(syms))
	lookup := func(addr uint64) (string, uint64) {
		i := sort.Search(len(syms), func(i int) bool { return addr < syms[i].Addr })
		if i > 0 {
			s := syms[i-1]
			if s.Addr != 0 && s.Addr <= addr && addr < s.Addr+uint64(s.Size) {
				return s.Name, s.Addr
			}
		}
		return "", 0
	}

	if flag.NArg() == 1 {
		// disassembly of entire object - our format
		dump(tab, lookup, disasm, goarch, syms, textBytes, textStart)
		os.Exit(0)
	}

	// disassembly of specific piece of object - gnu objdump format for pprof
	gnuDump(tab, lookup, disasm, textBytes, textStart)
	os.Exit(0)
}

// base returns the final element in the path.
// It works on both Windows and Unix paths.
func base(path string) string {
	path = path[strings.LastIndex(path, "/")+1:]
	path = path[strings.LastIndex(path, `\`)+1:]
	return path
}

func dump(tab *gosym.Table, lookup lookupFunc, disasm disasmFunc, goarch string, syms []objfile.Sym, textData []byte, textStart uint64) {
	stdout := bufio.NewWriter(os.Stdout)
	defer stdout.Flush()

	printed := false
	for _, sym := range syms {
		if (sym.Code != 'T' && sym.Code != 't') || sym.Size == 0 || sym.Addr < textStart || symRE != nil && !symRE.MatchString(sym.Name) {
			continue
		}
		if sym.Addr >= textStart+uint64(len(textData)) || sym.Addr+uint64(sym.Size) > textStart+uint64(len(textData)) {
			break
		}
		if printed {
			fmt.Fprintf(stdout, "\n")
		} else {
			printed = true
		}
		file, _, _ := tab.PCToLine(sym.Addr)
		fmt.Fprintf(stdout, "TEXT %s(SB) %s\n", sym.Name, file)
		tw := tabwriter.NewWriter(stdout, 1, 8, 1, '\t', 0)
		start := sym.Addr
		end := sym.Addr + uint64(sym.Size)
		for pc := start; pc < end; {
			i := pc - textStart
			text, size := disasm(textData[i:end-textStart], pc, lookup)
			file, line, _ := tab.PCToLine(pc)

			// ARM is word-based, so show actual word hex, not byte hex.
			// Since ARM is little endian, they're different.
			if goarch == "arm" && size == 4 {
				fmt.Fprintf(tw, "\t%s:%d\t%#x\t%08x\t%s\n", base(file), line, pc, binary.LittleEndian.Uint32(textData[i:i+uint64(size)]), text)
			} else {
				fmt.Fprintf(tw, "\t%s:%d\t%#x\t%x\t%s\n", base(file), line, pc, textData[i:i+uint64(size)], text)
			}
			pc += uint64(size)
		}
		tw.Flush()
	}
}

func disasm_386(code []byte, pc uint64, lookup lookupFunc) (string, int) {
	return disasm_x86(code, pc, lookup, 32)
}

func disasm_amd64(code []byte, pc uint64, lookup lookupFunc) (string, int) {
	return disasm_x86(code, pc, lookup, 64)
}

func disasm_x86(code []byte, pc uint64, lookup lookupFunc, arch int) (string, int) {
	inst, err := x86asm.Decode(code, 64)
	var text string
	size := inst.Len
	if err != nil || size == 0 || inst.Op == 0 {
		size = 1
		text = "?"
	} else {
		text = x86asm.Plan9Syntax(inst, pc, lookup)
	}
	return text, size
}

type textReader struct {
	code []byte
	pc   uint64
}

func (r textReader) ReadAt(data []byte, off int64) (n int, err error) {
	if off < 0 || uint64(off) < r.pc {
		return 0, io.EOF
	}
	d := uint64(off) - r.pc
	if d >= uint64(len(r.code)) {
		return 0, io.EOF
	}
	n = copy(data, r.code[d:])
	if n < len(data) {
		err = io.ErrUnexpectedEOF
	}
	return
}

func disasm_arm(code []byte, pc uint64, lookup lookupFunc) (string, int) {
	inst, err := armasm.Decode(code, armasm.ModeARM)
	var text string
	size := inst.Len
	if err != nil || size == 0 || inst.Op == 0 {
		size = 4
		text = "?"
	} else {
		text = armasm.Plan9Syntax(inst, pc, lookup, textReader{code, pc})
	}
	return text, size
}

var disasms = map[string]disasmFunc{
	"386":   disasm_386,
	"amd64": disasm_amd64,
	"arm":   disasm_arm,
}

func gnuDump(tab *gosym.Table, lookup lookupFunc, disasm disasmFunc, textData []byte, textStart uint64) {
	start, err := strconv.ParseUint(strings.TrimPrefix(flag.Arg(1), "0x"), 16, 64)
	if err != nil {
		log.Fatalf("invalid start PC: %v", err)
	}
	end, err := strconv.ParseUint(strings.TrimPrefix(flag.Arg(2), "0x"), 16, 64)
	if err != nil {
		log.Fatalf("invalid end PC: %v", err)
	}
	if start < textStart {
		start = textStart
	}
	if end < start {
		end = start
	}
	if end > textStart+uint64(len(textData)) {
		end = textStart + uint64(len(textData))
	}

	stdout := bufio.NewWriter(os.Stdout)
	defer stdout.Flush()

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
		for pc := spanPC; pc < endPC; {
			text, size := disasm(textData[pc-textStart:], pc, lookup)
			fmt.Fprintf(stdout, " %x: %s\n", pc, text)
			pc += uint64(size)
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
}

type ByAddr []objfile.Sym

func (x ByAddr) Less(i, j int) bool { return x[i].Addr < x[j].Addr }
func (x ByAddr) Len() int           { return len(x) }
func (x ByAddr) Swap(i, j int)      { x[i], x[j] = x[j], x[i] }
