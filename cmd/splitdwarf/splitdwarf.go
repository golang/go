// Copyright 2018 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

//go:build aix || darwin || dragonfly || freebsd || linux || netbsd || openbsd
// +build aix darwin dragonfly freebsd linux netbsd openbsd

/*
Splitdwarf uncompresses and copies the DWARF segment of a Mach-O
executable into the "dSYM" file expected by lldb and ports of gdb
on OSX.

Usage: splitdwarf osxMachoFile [ osxDsymFile ]

Unless a dSYM file name is provided on the command line,
splitdwarf will place it where the OSX tools expect it, in
"<osxMachoFile>.dSYM/Contents/Resources/DWARF/<osxMachoFile>",
creating directories as necessary.
*/
package main // import "golang.org/x/tools/cmd/splitdwarf"

import (
	"crypto/sha256"
	"fmt"
	"io"
	"os"
	"path/filepath"
	"strings"
	"syscall"

	"golang.org/x/tools/cmd/splitdwarf/internal/macho"
)

const (
	pageAlign = 12 // 4096 = 1 << 12
)

func note(format string, why ...interface{}) {
	fmt.Fprintf(os.Stderr, format+"\n", why...)
}

func fail(format string, why ...interface{}) {
	note(format, why...)
	os.Exit(1)
}

// splitdwarf inputexe [ outputdwarf ]
func main() {
	if len(os.Args) < 2 || len(os.Args) > 3 {
		fmt.Printf(`
Usage: %s input_exe [ output_dsym ]
Reads the executable input_exe, uncompresses and copies debugging
information into output_dsym. If output_dsym is not specified,
the path
      input_exe.dSYM/Contents/Resources/DWARF/input_exe
is used instead.  That is the path that gdb and lldb expect
on OSX.  Input_exe needs a UUID segment; if that is missing,
then one is created and added.  In that case, the permissions
for input_exe need to allow writing.
`, os.Args[0])
		return
	}

	// Read input, find DWARF, be sure it looks right
	inputExe := os.Args[1]
	exeFile, err := os.Open(inputExe)
	if err != nil {
		fail("%v", err)
	}
	exeMacho, err := macho.NewFile(exeFile)
	if err != nil {
		fail("(internal) Couldn't create macho, %v", err)
	}
	// Postpone dealing with output till input is known-good

	// describe(&exeMacho.FileTOC)

	// Offsets into __LINKEDIT:
	//
	// Command LC_SYMTAB =
	//  (1) number of symbols at file offset (within link edit section) of 16-byte symbol table entries
	// struct {
	//  StringTableIndex uint32
	//  Type, SectionIndex uint8
	//  Description uint16
	//  Value uint64
	// }
	//
	// (2) string table offset and size.  Strings are zero-byte terminated.  First must be " ".
	//
	// Command LC_DYSYMTAB = indices within symtab (above), except for IndSym
	//   IndSym Offset = file offset (within link edit section) of 4-byte indices within symtab.
	//
	// Section __TEXT.__symbol_stub1.
	//   Offset and size (Reserved2) locate and describe a table for this section.
	//   Symbols beginning at IndirectSymIndex (Reserved1) (see LC_DYSYMTAB.IndSymOffset) refer to this table.
	//   (These table entries are apparently PLTs [Procedure Linkage Table/Trampoline])
	//
	// Section __DATA.__nl_symbol_ptr.
	//   Reserved1 seems to be an index within the Indirect symbols (see LC_DYSYMTAB.IndSymOffset)
	//   Some of these symbols appear to be duplicates of other indirect symbols appearing early
	//
	// Section __DATA.__la_symbol_ptr.
	//   Reserved1 seems to be an index within the Indirect symbols (see LC_DYSYMTAB.IndSymOffset)
	//   Some of these symbols appear to be duplicates of other indirect symbols appearing early
	//

	// Create a File for the output dwarf.
	// Copy header, file type is MH_DSYM
	// Copy the relevant load commands

	// LoadCmdUuid
	// Symtab -- very abbreviated (Use DYSYMTAB Iextdefsym, Nextdefsym to identify these).
	// Segment __PAGEZERO
	// Segment __TEXT (zero the size, zero the offset of each section)
	// Segment __DATA (zero the size, zero the offset of each section)
	// Segment __LINKEDIT (contains the symbols and strings from Symtab)
	// Segment __DWARF (uncompressed)

	var uuid *macho.Uuid
	for _, l := range exeMacho.Loads {
		switch l.Command() {
		case macho.LcUuid:
			uuid = l.(*macho.Uuid)
		}
	}

	// Ensure a given load is not nil
	nonnilC := func(l macho.Load, s string) {
		if l == nil {
			fail("input file %s lacks load command %s", inputExe, s)
		}
	}

	// Find a segment by name and ensure it is not nil
	nonnilS := func(s string) *macho.Segment {
		l := exeMacho.Segment(s)
		if l == nil {
			fail("input file %s lacks segment %s", inputExe, s)
		}
		return l
	}

	newtoc := exeMacho.FileTOC.DerivedCopy(macho.MhDsym, 0)

	symtab := exeMacho.Symtab
	dysymtab := exeMacho.Dysymtab // Not appearing in output, but necessary to construct output
	nonnilC(symtab, "symtab")
	nonnilC(dysymtab, "dysymtab")
	text := nonnilS("__TEXT")
	data := nonnilS("__DATA")
	linkedit := nonnilS("__LINKEDIT")
	pagezero := nonnilS("__PAGEZERO")

	newtext := text.CopyZeroed()
	newdata := data.CopyZeroed()
	newsymtab := symtab.Copy()

	// Linkedit segment contain symbols and strings;
	// Symtab refers to offsets into linkedit.
	// This next bit initializes newsymtab and sets up data structures for the linkedit segment
	linkeditsyms := []macho.Nlist64{}
	linkeditstrings := []string{}

	// Linkedit will begin at the second page, i.e., offset is one page from beginning
	// Symbols come first
	linkeditsymbase := uint32(1) << pageAlign

	// Strings come second, offset by the number of symbols times their size.
	// Only those symbols from dysymtab.defsym are written into the debugging information.
	linkeditstringbase := linkeditsymbase + exeMacho.FileTOC.SymbolSize()*dysymtab.Nextdefsym

	// The first two bytes of the strings are reserved for space, null (' ', \000)
	linkeditstringcur := uint32(2)

	newsymtab.Syms = newsymtab.Syms[:0]
	newsymtab.Symoff = linkeditsymbase
	newsymtab.Stroff = linkeditstringbase
	newsymtab.Nsyms = dysymtab.Nextdefsym
	for i := uint32(0); i < dysymtab.Nextdefsym; i++ {
		ii := i + dysymtab.Iextdefsym
		oldsym := symtab.Syms[ii]
		newsymtab.Syms = append(newsymtab.Syms, oldsym)

		linkeditsyms = append(linkeditsyms, macho.Nlist64{Name: linkeditstringcur,
			Type: oldsym.Type, Sect: oldsym.Sect, Desc: oldsym.Desc, Value: oldsym.Value})
		linkeditstringcur += uint32(len(oldsym.Name)) + 1
		linkeditstrings = append(linkeditstrings, oldsym.Name)
	}
	newsymtab.Strsize = linkeditstringcur

	exeNeedsUuid := uuid == nil
	if exeNeedsUuid {
		uuid = &macho.Uuid{macho.UuidCmd{LoadCmd: macho.LcUuid}}
		uuid.Len = uuid.LoadSize(newtoc)
		copy(uuid.Id[0:], contentuuid(&exeMacho.FileTOC)[0:16])
		uuid.Id[6] = uuid.Id[6]&^0xf0 | 0x40 // version 4 (pseudo-random); see section 4.1.3
		uuid.Id[8] = uuid.Id[8]&^0xc0 | 0x80 // variant bits; see section 4.1.1
	}
	newtoc.AddLoad(uuid)

	// For the specified segment (assumed to be in exeMacho) make a copy of its
	// sections with appropriate fields zeroed out, and append them to the
	// currently-last segment in newtoc.
	copyZOdSections := func(g *macho.Segment) {
		for i := g.Firstsect; i < g.Firstsect+g.Nsect; i++ {
			s := exeMacho.Sections[i].Copy()
			s.Offset = 0
			s.Reloff = 0
			s.Nreloc = 0
			newtoc.AddSection(s)
		}
	}

	newtoc.AddLoad(newsymtab)
	newtoc.AddSegment(pagezero)
	newtoc.AddSegment(newtext)
	copyZOdSections(text)
	newtoc.AddSegment(newdata)
	copyZOdSections(data)

	newlinkedit := linkedit.Copy()
	newlinkedit.Offset = uint64(linkeditsymbase)
	newlinkedit.Filesz = uint64(linkeditstringcur)
	newlinkedit.Addr = macho.RoundUp(newdata.Addr+newdata.Memsz, 1<<pageAlign) // Follows data sections in file
	newlinkedit.Memsz = macho.RoundUp(newlinkedit.Filesz, 1<<pageAlign)
	// The rest should copy over fine.
	newtoc.AddSegment(newlinkedit)

	dwarf := nonnilS("__DWARF")
	newdwarf := dwarf.CopyZeroed()
	newdwarf.Offset = macho.RoundUp(newlinkedit.Offset+newlinkedit.Filesz, 1<<pageAlign)
	newdwarf.Filesz = dwarf.UncompressedSize(&exeMacho.FileTOC, 1)
	newdwarf.Addr = newlinkedit.Addr + newlinkedit.Memsz // Follows linkedit sections in file.
	newdwarf.Memsz = macho.RoundUp(newdwarf.Filesz, 1<<pageAlign)
	newtoc.AddSegment(newdwarf)

	// Map out Dwarf sections (that is, this is section descriptors, not their contents).
	offset := uint32(newdwarf.Offset)
	for i := dwarf.Firstsect; i < dwarf.Firstsect+dwarf.Nsect; i++ {
		o := exeMacho.Sections[i]
		s := o.Copy()
		s.Offset = offset
		us := o.UncompressedSize()
		if s.Size < us {
			s.Size = uint64(us)
			s.Align = 0 // This is apparently true for debugging sections; not sure if it generalizes.
		}
		offset += uint32(us)
		if strings.HasPrefix(s.Name, "__z") {
			s.Name = "__" + s.Name[3:] // remove "z"
		}
		s.Reloff = 0
		s.Nreloc = 0
		newtoc.AddSection(s)
	}

	// Write segments/sections.
	// Only dwarf and linkedit contain anything interesting.

	// Memory map the output file to get the buffer directly.
	outDwarf := inputExe + ".dSYM/Contents/Resources/DWARF"
	if len(os.Args) > 2 {
		outDwarf = os.Args[2]
	} else {
		err := os.MkdirAll(outDwarf, 0755)
		if err != nil {
			fail("%v", err)
		}
		outDwarf = filepath.Join(outDwarf, filepath.Base(inputExe))
	}
	dwarfFile, buffer := CreateMmapFile(outDwarf, int64(newtoc.FileSize()))

	// (1) Linkedit segment
	// Symbol table
	offset = uint32(newlinkedit.Offset)
	for i := range linkeditsyms {
		if exeMacho.Magic == macho.Magic64 {
			offset += linkeditsyms[i].Put64(buffer[offset:], newtoc.ByteOrder)
		} else {
			offset += linkeditsyms[i].Put32(buffer[offset:], newtoc.ByteOrder)
		}
	}

	// Initial two bytes of string table, followed by actual zero-terminated strings.
	buffer[linkeditstringbase] = ' '
	buffer[linkeditstringbase+1] = 0
	offset = linkeditstringbase + 2
	for _, str := range linkeditstrings {
		for i := 0; i < len(str); i++ {
			buffer[offset] = str[i]
			offset++
		}
		buffer[offset] = 0
		offset++
	}

	// (2) DWARF segment
	ioff := newdwarf.Firstsect - dwarf.Firstsect
	for i := dwarf.Firstsect; i < dwarf.Firstsect+dwarf.Nsect; i++ {
		s := exeMacho.Sections[i]
		j := i + ioff
		s.PutUncompressedData(buffer[newtoc.Sections[j].Offset:])
	}

	// Because "text" overlaps the header and the loads, write them afterwards, just in case.
	// Write header.
	newtoc.Put(buffer)

	err = syscall.Munmap(buffer)
	if err != nil {
		fail("Munmap %s for dwarf output failed, %v", outDwarf, err)
	}
	err = dwarfFile.Close()
	if err != nil {
		fail("Close %s for dwarf output after mmap/munmap failed, %v", outDwarf, err)
	}

	if exeNeedsUuid { // Map the original exe, modify the header, and write the UUID command
		hdr := exeMacho.FileTOC.FileHeader
		oldCommandEnd := hdr.SizeCommands + newtoc.HdrSize()
		hdr.NCommands += 1
		hdr.SizeCommands += uuid.LoadSize(newtoc)

		mapf, err := os.OpenFile(inputExe, os.O_RDWR, 0)
		if err != nil {
			fail("Updating UUID in binary failed, %v", err)
		}
		exebuf, err := syscall.Mmap(int(mapf.Fd()), 0, int(macho.RoundUp(uint64(hdr.SizeCommands), 1<<pageAlign)),
			syscall.PROT_READ|syscall.PROT_WRITE, syscall.MAP_FILE|syscall.MAP_SHARED)
		if err != nil {
			fail("Mmap of %s for UUID update failed, %v", inputExe, err)
		}
		_ = hdr.Put(exebuf, newtoc.ByteOrder)
		_ = uuid.Put(exebuf[oldCommandEnd:], newtoc.ByteOrder)
		err = syscall.Munmap(exebuf)
		if err != nil {
			fail("Munmap of %s for UUID update failed, %v", inputExe, err)
		}
	}
}

// CreateMmapFile creates the file 'outDwarf' of the specified size, mmaps that file,
// and returns the file descriptor and mapped buffer.
func CreateMmapFile(outDwarf string, size int64) (*os.File, []byte) {
	dwarfFile, err := os.OpenFile(outDwarf, os.O_RDWR|os.O_CREATE|os.O_TRUNC, 0666)
	if err != nil {
		fail("Open for mmap failed, %v", err)
	}
	err = os.Truncate(outDwarf, size)
	if err != nil {
		fail("Truncate/extend of %s to %d bytes failed, %v", dwarfFile, size, err)
	}
	buffer, err := syscall.Mmap(int(dwarfFile.Fd()), 0, int(size), syscall.PROT_READ|syscall.PROT_WRITE, syscall.MAP_FILE|syscall.MAP_SHARED)
	if err != nil {
		fail("Mmap %s for dwarf output update failed, %v", outDwarf, err)
	}
	return dwarfFile, buffer
}

func describe(exem *macho.FileTOC) {
	note("Type = %s, Flags=0x%x", exem.Type, uint32(exem.Flags))
	for i, l := range exem.Loads {
		if s, ok := l.(*macho.Segment); ok {
			fmt.Printf("Load %d is Segment %s, offset=0x%x, filesz=%d, addr=0x%x, memsz=%d, nsect=%d\n", i, s.Name,
				s.Offset, s.Filesz, s.Addr, s.Memsz, s.Nsect)
			for j := uint32(0); j < s.Nsect; j++ {
				c := exem.Sections[j+s.Firstsect]
				fmt.Printf("   Section %s, offset=0x%x, size=%d, addr=0x%x, flags=0x%x, nreloc=%d, res1=%d, res2=%d, res3=%d\n", c.Name, c.Offset, c.Size, c.Addr, c.Flags, c.Nreloc, c.Reserved1, c.Reserved2, c.Reserved3)
			}
		} else {
			fmt.Printf("Load %d is %v\n", i, l)
		}
	}
	if exem.SizeCommands != exem.LoadSize() {
		fail("recorded command size %d does not equal computed command size %d", exem.SizeCommands, exem.LoadSize())
	} else {
		note("recorded command size %d, computed command size %d", exem.SizeCommands, exem.LoadSize())
	}
	note("File size is %d", exem.FileSize())
}

// contentuuid returns a UUID derived from (some of) the content of an executable.
// specifically included are the non-DWARF sections, specifically excluded are things
// that surely depend on the presence or absence of DWARF sections (e.g., section
// numbers, positions with file, number of load commands).
// (It was considered desirable if this was insensitive to the presence of the
// __DWARF segment, however because it is not last, it moves other segments,
// whose contents appear to contain file offset references.)
func contentuuid(exem *macho.FileTOC) []byte {
	h := sha256.New()
	for _, l := range exem.Loads {
		if l.Command() == macho.LcUuid {
			continue
		}
		if s, ok := l.(*macho.Segment); ok {
			if s.Name == "__DWARF" || s.Name == "__PAGEZERO" {
				continue
			}
			for j := uint32(0); j < s.Nsect; j++ {
				c := exem.Sections[j+s.Firstsect]
				io.Copy(h, c.Open())
			}
		} // Getting dependence on other load commands right is fiddly.
	}
	return h.Sum(nil)
}
