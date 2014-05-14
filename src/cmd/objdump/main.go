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
//
// The ARM disassembler is missing (golang.org/issue/7452) but will be added
// before the Go 1.3 release.
package main

import (
	"bufio"
	"bytes"
	"debug/elf"
	"debug/gosym"
	"debug/macho"
	"debug/pe"
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

	f, err := os.Open(flag.Arg(0))
	if err != nil {
		log.Fatal(err)
	}

	textStart, textData, symtab, pclntab, err := loadTables(f)
	if err != nil {
		log.Fatalf("reading %s: %v", flag.Arg(0), err)
	}

	syms, goarch, err := loadSymbols(f)
	if err != nil {
		log.Fatalf("reading %s: %v", flag.Arg(0), err)
	}

	// Filter out section symbols, overwriting syms in place.
	keep := syms[:0]
	for _, sym := range syms {
		switch sym.Name {
		case "text", "_text", "etext", "_etext":
			// drop
		default:
			keep = append(keep, sym)
		}
	}
	syms = keep

	disasm := disasms[goarch]
	if disasm == nil {
		log.Fatalf("reading %s: unknown architecture", flag.Arg(0))
	}

	lookup := func(addr uint64) (string, uint64) {
		i := sort.Search(len(syms), func(i int) bool { return syms[i].Addr > addr })
		if i > 0 {
			s := syms[i-1]
			if s.Addr <= addr && addr < s.Addr+uint64(s.Size) && s.Name != "etext" && s.Name != "_etext" {
				return s.Name, s.Addr
			}
		}
		return "", 0
	}

	pcln := gosym.NewLineTable(pclntab, textStart)
	tab, err := gosym.NewTable(symtab, pcln)
	if err != nil {
		log.Fatalf("reading %s: %v", flag.Arg(0), err)
	}

	if flag.NArg() == 1 {
		// disassembly of entire object - our format
		dump(tab, lookup, disasm, syms, textData, textStart)
		os.Exit(exitCode)
	}

	// disassembly of specific piece of object - gnu objdump format for pprof
	gnuDump(tab, lookup, disasm, textData, textStart)
	os.Exit(exitCode)
}

// base returns the final element in the path.
// It works on both Windows and Unix paths.
func base(path string) string {
	path = path[strings.LastIndex(path, "/")+1:]
	path = path[strings.LastIndex(path, `\`)+1:]
	return path
}

func dump(tab *gosym.Table, lookup lookupFunc, disasm disasmFunc, syms []Sym, textData []byte, textStart uint64) {
	stdout := bufio.NewWriter(os.Stdout)
	defer stdout.Flush()

	printed := false
	for _, sym := range syms {
		if sym.Code != 'T' || sym.Size == 0 || sym.Name == "_text" || sym.Name == "text" || sym.Addr < textStart || symRE != nil && !symRE.MatchString(sym.Name) {
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
			fmt.Fprintf(tw, "\t%s:%d\t%#x\t%x\t%s\n", base(file), line, pc, textData[i:i+uint64(size)], text)
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
	inst, err := x86_Decode(code, 64)
	var text string
	size := inst.Len
	if err != nil || size == 0 || inst.Op == 0 {
		size = 1
		text = "?"
	} else {
		text = x86_plan9Syntax(inst, pc, lookup)
	}
	return text, size
}

func disasm_arm(code []byte, pc uint64, lookup lookupFunc) (string, int) {
	/*
		inst, size, err := arm_Decode(code, 64)
		var text string
		if err != nil || size == 0 || inst.Op == 0 {
			size = 1
			text = "?"
		} else {
			text = arm_plan9Syntax(inst, pc, lookup)
		}
		return text, size
	*/
	return "?", 4
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
		if pclntab, err = loadPETable(obj, "pclntab", "epclntab"); err != nil {
			return 0, nil, nil, nil, err
		}
		if symtab, err = loadPETable(obj, "symtab", "esymtab"); err != nil {
			return 0, nil, nil, nil, err
		}
		return textStart, textData, symtab, pclntab, nil
	}

	return 0, nil, nil, nil, fmt.Errorf("unrecognized binary format")
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

// TODO(rsc): This code is taken from cmd/nm. Arrange some way to share the code.

var exitCode = 0

func errorf(format string, args ...interface{}) {
	log.Printf(format, args...)
	exitCode = 1
}

func loadSymbols(f *os.File) (syms []Sym, goarch string, err error) {
	f.Seek(0, 0)
	buf := make([]byte, 16)
	io.ReadFull(f, buf)
	f.Seek(0, 0)

	for _, p := range parsers {
		if bytes.HasPrefix(buf, p.prefix) {
			syms, goarch = p.parse(f)
			sort.Sort(byAddr(syms))
			return
		}
	}
	err = fmt.Errorf("unknown file format")
	return
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
	parse  func(*os.File) ([]Sym, string)
}{
	{[]byte("\x7FELF"), elfSymbols},
	{[]byte("\xFE\xED\xFA\xCE"), machoSymbols},
	{[]byte("\xFE\xED\xFA\xCF"), machoSymbols},
	{[]byte("\xCE\xFA\xED\xFE"), machoSymbols},
	{[]byte("\xCF\xFA\xED\xFE"), machoSymbols},
	{[]byte("MZ"), peSymbols},
	{[]byte("\x00\x00\x01\xEB"), plan9Symbols}, // 386
	{[]byte("\x00\x00\x04\x07"), plan9Symbols}, // mips
	{[]byte("\x00\x00\x06\x47"), plan9Symbols}, // arm
	{[]byte("\x00\x00\x8A\x97"), plan9Symbols}, // amd64
}

type byAddr []Sym

func (x byAddr) Len() int           { return len(x) }
func (x byAddr) Swap(i, j int)      { x[i], x[j] = x[j], x[i] }
func (x byAddr) Less(i, j int) bool { return x[i].Addr < x[j].Addr }
