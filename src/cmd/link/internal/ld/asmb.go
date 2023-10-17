// Copyright 2020 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

package ld

import (
	"cmd/internal/objabi"
	"cmd/link/internal/loader"
	"cmd/link/internal/sym"
	"fmt"
	"runtime"
	"sync"
)

// Assembling the binary is broken into two steps:
//   - writing out the code/data/dwarf Segments, applying relocations on the fly
//   - writing out the architecture specific pieces.
//
// This function handles the first part.
func asmb(ctxt *Link) {
	// TODO(jfaller): delete me.
	if thearch.Asmb != nil {
		thearch.Asmb(ctxt, ctxt.loader)
		return
	}

	if ctxt.IsELF {
		Asmbelfsetup()
	}

	var wg sync.WaitGroup
	f := func(ctxt *Link, out *OutBuf, start, length int64) {
		pad := thearch.CodePad
		if pad == nil {
			pad = zeros[:]
		}
		CodeblkPad(ctxt, out, start, length, pad)
	}

	for _, sect := range Segtext.Sections {
		offset := sect.Vaddr - Segtext.Vaddr + Segtext.Fileoff
		// Handle text sections with Codeblk
		if sect.Name == ".text" {
			writeParallel(&wg, f, ctxt, offset, sect.Vaddr, sect.Length)
		} else {
			writeParallel(&wg, datblk, ctxt, offset, sect.Vaddr, sect.Length)
		}
	}

	if Segrodata.Filelen > 0 {
		writeParallel(&wg, datblk, ctxt, Segrodata.Fileoff, Segrodata.Vaddr, Segrodata.Filelen)
	}

	if Segrelrodata.Filelen > 0 {
		writeParallel(&wg, datblk, ctxt, Segrelrodata.Fileoff, Segrelrodata.Vaddr, Segrelrodata.Filelen)
	}

	writeParallel(&wg, datblk, ctxt, Segdata.Fileoff, Segdata.Vaddr, Segdata.Filelen)

	writeParallel(&wg, dwarfblk, ctxt, Segdwarf.Fileoff, Segdwarf.Vaddr, Segdwarf.Filelen)

	if Segpdata.Filelen > 0 {
		writeParallel(&wg, pdatablk, ctxt, Segpdata.Fileoff, Segpdata.Vaddr, Segpdata.Filelen)
	}
	if Segxdata.Filelen > 0 {
		writeParallel(&wg, xdatablk, ctxt, Segxdata.Fileoff, Segxdata.Vaddr, Segxdata.Filelen)
	}

	wg.Wait()
}

// Assembling the binary is broken into two steps:
//   - writing out the code/data/dwarf Segments
//   - writing out the architecture specific pieces.
//
// This function handles the second part.
func asmb2(ctxt *Link) {
	if thearch.Asmb2 != nil {
		thearch.Asmb2(ctxt, ctxt.loader)
		return
	}

	symSize = 0
	spSize = 0
	lcSize = 0

	switch ctxt.HeadType {
	default:
		panic("unknown platform")

	// Macho
	case objabi.Hdarwin:
		asmbMacho(ctxt)

	// Plan9
	case objabi.Hplan9:
		asmbPlan9(ctxt)

	// PE
	case objabi.Hwindows:
		asmbPe(ctxt)

	// Xcoff
	case objabi.Haix:
		asmbXcoff(ctxt)

	// Elf
	case objabi.Hdragonfly,
		objabi.Hfreebsd,
		objabi.Hlinux,
		objabi.Hnetbsd,
		objabi.Hopenbsd,
		objabi.Hsolaris:
		asmbElf(ctxt)
	}

	if *FlagC {
		fmt.Printf("textsize=%d\n", Segtext.Filelen)
		fmt.Printf("datsize=%d\n", Segdata.Filelen)
		fmt.Printf("bsssize=%d\n", Segdata.Length-Segdata.Filelen)
		fmt.Printf("symsize=%d\n", symSize)
		fmt.Printf("lcsize=%d\n", lcSize)
		fmt.Printf("total=%d\n", Segtext.Filelen+Segdata.Length+uint64(symSize)+uint64(lcSize))
	}
}

// writePlan9Header writes out the plan9 header at the present position in the OutBuf.
func writePlan9Header(buf *OutBuf, magic uint32, entry int64, is64Bit bool) {
	if is64Bit {
		magic |= 0x00008000
	}
	buf.Write32b(magic)
	buf.Write32b(uint32(Segtext.Filelen))
	buf.Write32b(uint32(Segdata.Filelen))
	buf.Write32b(uint32(Segdata.Length - Segdata.Filelen))
	buf.Write32b(uint32(symSize))
	if is64Bit {
		buf.Write32b(uint32(entry &^ 0x80000000))
	} else {
		buf.Write32b(uint32(entry))
	}
	buf.Write32b(uint32(spSize))
	buf.Write32b(uint32(lcSize))
	// amd64 includes the entry at the beginning of the symbol table.
	if is64Bit {
		buf.Write64b(uint64(entry))
	}
}

// asmbPlan9 assembles a plan 9 binary.
func asmbPlan9(ctxt *Link) {
	if !*FlagS {
		*FlagS = true
		symo := int64(Segdata.Fileoff + Segdata.Filelen)
		ctxt.Out.SeekSet(symo)
		asmbPlan9Sym(ctxt)
	}
	ctxt.Out.SeekSet(0)
	writePlan9Header(ctxt.Out, thearch.Plan9Magic, Entryvalue(ctxt), thearch.Plan9_64Bit)
}

// sizeExtRelocs precomputes the size needed for the reloc records,
// sets the size and offset for relocation records in each section,
// and mmap the output buffer with the proper size.
func sizeExtRelocs(ctxt *Link, relsize uint32) {
	if relsize == 0 {
		panic("sizeExtRelocs: relocation size not set")
	}
	var sz int64
	for _, seg := range Segments {
		for _, sect := range seg.Sections {
			sect.Reloff = uint64(ctxt.Out.Offset() + sz)
			sect.Rellen = uint64(relsize * sect.Relcount)
			sz += int64(sect.Rellen)
		}
	}
	filesz := ctxt.Out.Offset() + sz
	err := ctxt.Out.Mmap(uint64(filesz))
	if err != nil {
		Exitf("mapping output file failed: %v", err)
	}
}

// relocSectFn wraps the function writing relocations of a section
// for parallel execution. Returns the wrapped function and a wait
// group for which the caller should wait.
func relocSectFn(ctxt *Link, relocSect func(*Link, *OutBuf, *sym.Section, []loader.Sym)) (func(*Link, *sym.Section, []loader.Sym), *sync.WaitGroup) {
	var fn func(ctxt *Link, sect *sym.Section, syms []loader.Sym)
	var wg sync.WaitGroup
	var sem chan int
	if ctxt.Out.isMmapped() {
		// Write sections in parallel.
		sem = make(chan int, 2*runtime.GOMAXPROCS(0))
		fn = func(ctxt *Link, sect *sym.Section, syms []loader.Sym) {
			wg.Add(1)
			sem <- 1
			out, err := ctxt.Out.View(sect.Reloff)
			if err != nil {
				panic(err)
			}
			go func() {
				relocSect(ctxt, out, sect, syms)
				wg.Done()
				<-sem
			}()
		}
	} else {
		// We cannot Mmap. Write sequentially.
		fn = func(ctxt *Link, sect *sym.Section, syms []loader.Sym) {
			relocSect(ctxt, ctxt.Out, sect, syms)
		}
	}
	return fn, &wg
}
