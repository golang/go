// Copyright 2014 Google Inc. All Rights Reserved.
//
// Licensed under the Apache License, Version 2.0 (the "License");
// you may not use this file except in compliance with the License.
// You may obtain a copy of the License at
//
//     http://www.apache.org/licenses/LICENSE-2.0
//
// Unless required by applicable law or agreed to in writing, software
// distributed under the License is distributed on an "AS IS" BASIS,
// WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
// See the License for the specific language governing permissions and
// limitations under the License.

// Package elfexec provides utility routines to examine ELF binaries.
package elfexec

import (
	"bufio"
	"debug/elf"
	"encoding/binary"
	"fmt"
	"io"
)

const (
	maxNoteSize        = 1 << 20 // in bytes
	noteTypeGNUBuildID = 3
)

// elfNote is the payload of a Note Section in an ELF file.
type elfNote struct {
	Name string // Contents of the "name" field, omitting the trailing zero byte.
	Desc []byte // Contents of the "desc" field.
	Type uint32 // Contents of the "type" field.
}

// parseNotes returns the notes from a SHT_NOTE section or PT_NOTE segment.
func parseNotes(reader io.Reader, alignment int, order binary.ByteOrder) ([]elfNote, error) {
	r := bufio.NewReader(reader)

	// padding returns the number of bytes required to pad the given size to an
	// alignment boundary.
	padding := func(size int) int {
		return ((size + (alignment - 1)) &^ (alignment - 1)) - size
	}

	var notes []elfNote
	for {
		noteHeader := make([]byte, 12) // 3 4-byte words
		if _, err := io.ReadFull(r, noteHeader); err == io.EOF {
			break
		} else if err != nil {
			return nil, err
		}
		namesz := order.Uint32(noteHeader[0:4])
		descsz := order.Uint32(noteHeader[4:8])
		typ := order.Uint32(noteHeader[8:12])

		if uint64(namesz) > uint64(maxNoteSize) {
			return nil, fmt.Errorf("note name too long (%d bytes)", namesz)
		}
		var name string
		if namesz > 0 {
			// Documentation differs as to whether namesz is meant to include the
			// trailing zero, but everyone agrees that name is null-terminated.
			// So we'll just determine the actual length after the fact.
			var err error
			name, err = r.ReadString('\x00')
			if err == io.EOF {
				return nil, fmt.Errorf("missing note name (want %d bytes)", namesz)
			} else if err != nil {
				return nil, err
			}
			namesz = uint32(len(name))
			name = name[:len(name)-1]
		}

		// Drop padding bytes until the desc field.
		for n := padding(len(noteHeader) + int(namesz)); n > 0; n-- {
			if _, err := r.ReadByte(); err == io.EOF {
				return nil, fmt.Errorf(
					"missing %d bytes of padding after note name", n)
			} else if err != nil {
				return nil, err
			}
		}

		if uint64(descsz) > uint64(maxNoteSize) {
			return nil, fmt.Errorf("note desc too long (%d bytes)", descsz)
		}
		desc := make([]byte, int(descsz))
		if _, err := io.ReadFull(r, desc); err == io.EOF {
			return nil, fmt.Errorf("missing desc (want %d bytes)", len(desc))
		} else if err != nil {
			return nil, err
		}

		notes = append(notes, elfNote{Name: name, Desc: desc, Type: typ})

		// Drop padding bytes until the next note or the end of the section,
		// whichever comes first.
		for n := padding(len(desc)); n > 0; n-- {
			if _, err := r.ReadByte(); err == io.EOF {
				// We hit the end of the section before an alignment boundary.
				// This can happen if this section is at the end of the file or the next
				// section has a smaller alignment requirement.
				break
			} else if err != nil {
				return nil, err
			}
		}
	}
	return notes, nil
}

// GetBuildID returns the GNU build-ID for an ELF binary.
//
// If no build-ID was found but the binary was read without error, it returns
// (nil, nil).
func GetBuildID(f *elf.File) ([]byte, error) {
	findBuildID := func(notes []elfNote) ([]byte, error) {
		var buildID []byte
		for _, note := range notes {
			if note.Name == "GNU" && note.Type == noteTypeGNUBuildID {
				if buildID == nil {
					buildID = note.Desc
				} else {
					return nil, fmt.Errorf("multiple build ids found, don't know which to use")
				}
			}
		}
		return buildID, nil
	}

	for _, p := range f.Progs {
		if p.Type != elf.PT_NOTE {
			continue
		}
		notes, err := parseNotes(p.Open(), int(p.Align), f.ByteOrder)
		if err != nil {
			return nil, err
		}
		if b, err := findBuildID(notes); b != nil || err != nil {
			return b, err
		}
	}
	for _, s := range f.Sections {
		if s.Type != elf.SHT_NOTE {
			continue
		}
		notes, err := parseNotes(s.Open(), int(s.Addralign), f.ByteOrder)
		if err != nil {
			return nil, err
		}
		if b, err := findBuildID(notes); b != nil || err != nil {
			return b, err
		}
	}
	return nil, nil
}

// kernelBase calculates the base for kernel mappings, which usually require
// special handling. For kernel mappings, tools (like perf) use the address of
// the kernel relocation symbol (_text or _stext) as the mmap start. Additionally,
// for obfuscation, ChromeOS profiles have the kernel image remapped to the 0-th page.
func kernelBase(loadSegment *elf.ProgHeader, stextOffset *uint64, start, limit, offset uint64) (uint64, bool) {
	const (
		// PAGE_OFFSET for PowerPC64, see arch/powerpc/Kconfig in the kernel sources.
		pageOffsetPpc64 = 0xc000000000000000
		pageSize        = 4096
	)

	if loadSegment.Vaddr == start-offset {
		return offset, true
	}
	if start == 0 && limit != 0 && stextOffset != nil {
		// ChromeOS remaps its kernel to 0. Nothing else should come
		// down this path. Empirical values:
		//       VADDR=0xffffffff80200000
		// stextOffset=0xffffffff80200198
		return start - *stextOffset, true
	}
	if start >= loadSegment.Vaddr && limit > start && (offset == 0 || offset == pageOffsetPpc64 || offset == start) {
		// Some kernels look like:
		//       VADDR=0xffffffff80200000
		// stextOffset=0xffffffff80200198
		//       Start=0xffffffff83200000
		//       Limit=0xffffffff84200000
		//      Offset=0 (0xc000000000000000 for PowerPC64) (== Start for ASLR kernel)
		// So the base should be:
		if stextOffset != nil && (start%pageSize) == (*stextOffset%pageSize) {
			// perf uses the address of _stext as start. Some tools may
			// adjust for this before calling GetBase, in which case the page
			// alignment should be different from that of stextOffset.
			return start - *stextOffset, true
		}

		return start - loadSegment.Vaddr, true
	}
	if start%pageSize != 0 && stextOffset != nil && *stextOffset%pageSize == start%pageSize {
		// ChromeOS remaps its kernel to 0 + start%pageSize. Nothing
		// else should come down this path. Empirical values:
		//       start=0x198 limit=0x2f9fffff offset=0
		//       VADDR=0xffffffff81000000
		// stextOffset=0xffffffff81000198
		return start - *stextOffset, true
	}
	return 0, false
}

// GetBase determines the base address to subtract from virtual
// address to get symbol table address. For an executable, the base
// is 0. Otherwise, it's a shared library, and the base is the
// address where the mapping starts. The kernel needs special handling.
func GetBase(fh *elf.FileHeader, loadSegment *elf.ProgHeader, stextOffset *uint64, start, limit, offset uint64) (uint64, error) {

	if start == 0 && offset == 0 && (limit == ^uint64(0) || limit == 0) {
		// Some tools may introduce a fake mapping that spans the entire
		// address space. Assume that the address has already been
		// adjusted, so no additional base adjustment is necessary.
		return 0, nil
	}

	switch fh.Type {
	case elf.ET_EXEC:
		if loadSegment == nil {
			// Assume fixed-address executable and so no adjustment.
			return 0, nil
		}
		if stextOffset == nil && start > 0 && start < 0x8000000000000000 {
			// A regular user-mode executable. Compute the base offset using same
			// arithmetics as in ET_DYN case below, see the explanation there.
			// Ideally, the condition would just be "stextOffset == nil" as that
			// represents the address of _stext symbol in the vmlinux image. Alas,
			// the caller may skip reading it from the binary (it's expensive to scan
			// all the symbols) and so it may be nil even for the kernel executable.
			// So additionally check that the start is within the user-mode half of
			// the 64-bit address space.
			return start - offset + loadSegment.Off - loadSegment.Vaddr, nil
		}
		// Various kernel heuristics and cases are handled separately.
		if base, match := kernelBase(loadSegment, stextOffset, start, limit, offset); match {
			return base, nil
		}
		// ChromeOS can remap its kernel to 0, and the caller might have not found
		// the _stext symbol. Split this case from kernelBase() above, since we don't
		// want to apply it to an ET_DYN user-mode executable.
		if start == 0 && limit != 0 && stextOffset == nil {
			return start - loadSegment.Vaddr, nil
		}

		return 0, fmt.Errorf("don't know how to handle EXEC segment: %v start=0x%x limit=0x%x offset=0x%x", *loadSegment, start, limit, offset)
	case elf.ET_REL:
		if offset != 0 {
			return 0, fmt.Errorf("don't know how to handle mapping.Offset")
		}
		return start, nil
	case elf.ET_DYN:
		// The process mapping information, start = start of virtual address range,
		// and offset = offset in the executable file of the start address, tells us
		// that a runtime virtual address x maps to a file offset
		// fx = x - start + offset.
		if loadSegment == nil {
			return start - offset, nil
		}
		// Kernels compiled as PIE can be ET_DYN as well. Use heuristic, similar to
		// the ET_EXEC case above.
		if base, match := kernelBase(loadSegment, stextOffset, start, limit, offset); match {
			return base, nil
		}
		// The program header, if not nil, indicates the offset in the file where
		// the executable segment is located (loadSegment.Off), and the base virtual
		// address where the first byte of the segment is loaded
		// (loadSegment.Vaddr). A file offset fx maps to a virtual (symbol) address
		// sx = fx - loadSegment.Off + loadSegment.Vaddr.
		//
		// Thus, a runtime virtual address x maps to a symbol address
		// sx = x - start + offset - loadSegment.Off + loadSegment.Vaddr.
		return start - offset + loadSegment.Off - loadSegment.Vaddr, nil
	}
	return 0, fmt.Errorf("don't know how to handle FileHeader.Type %v", fh.Type)
}

// FindTextProgHeader finds the program segment header containing the .text
// section or nil if the segment cannot be found.
func FindTextProgHeader(f *elf.File) *elf.ProgHeader {
	for _, s := range f.Sections {
		if s.Name == ".text" {
			// Find the LOAD segment containing the .text section.
			for _, p := range f.Progs {
				if p.Type == elf.PT_LOAD && p.Flags&elf.PF_X != 0 && s.Addr >= p.Vaddr && s.Addr < p.Vaddr+p.Memsz {
					return &p.ProgHeader
				}
			}
		}
	}
	return nil
}

// ProgramHeadersForMapping returns the program segment headers that overlap
// the runtime mapping with file offset mapOff and memory size mapSz. We skip
// over segments zero file size because their file offset values are unreliable.
// Even if overlapping, a segment is not selected if its aligned file offset is
// greater than the mapping file offset, or if the mapping includes the last
// page of the segment, but not the full segment and the mapping includes
// additional pages after the segment end.
// The function returns a slice of pointers to the headers in the input
// slice, which are valid only while phdrs is not modified or discarded.
func ProgramHeadersForMapping(phdrs []elf.ProgHeader, mapOff, mapSz uint64) []*elf.ProgHeader {
	const (
		// pageSize defines the virtual memory page size used by the loader. This
		// value is dependent on the memory management unit of the CPU. The page
		// size is 4KB virtually on all the architectures that we care about, so we
		// define this metric as a constant. If we encounter architectures where
		// page sie is not 4KB, we must try to guess the page size on the system
		// where the profile was collected, possibly using the architecture
		// specified in the ELF file header.
		pageSize       = 4096
		pageOffsetMask = pageSize - 1
	)
	mapLimit := mapOff + mapSz
	var headers []*elf.ProgHeader
	for i := range phdrs {
		p := &phdrs[i]
		// Skip over segments with zero file size. Their file offsets can have
		// arbitrary values, see b/195427553.
		if p.Filesz == 0 {
			continue
		}
		segLimit := p.Off + p.Memsz
		// The segment must overlap the mapping.
		if p.Type == elf.PT_LOAD && mapOff < segLimit && p.Off < mapLimit {
			// If the mapping offset is strictly less than the page aligned segment
			// offset, then this mapping comes from a different segment, fixes
			// b/179920361.
			alignedSegOffset := uint64(0)
			if p.Off > (p.Vaddr & pageOffsetMask) {
				alignedSegOffset = p.Off - (p.Vaddr & pageOffsetMask)
			}
			if mapOff < alignedSegOffset {
				continue
			}
			// If the mapping starts in the middle of the segment, it covers less than
			// one page of the segment, and it extends at least one page past the
			// segment, then this mapping comes from a different segment.
			if mapOff > p.Off && (segLimit < mapOff+pageSize) && (mapLimit >= segLimit+pageSize) {
				continue
			}
			headers = append(headers, p)
		}
	}
	return headers
}

// HeaderForFileOffset attempts to identify a unique program header that
// includes the given file offset. It returns an error if it cannot identify a
// unique header.
func HeaderForFileOffset(headers []*elf.ProgHeader, fileOffset uint64) (*elf.ProgHeader, error) {
	var ph *elf.ProgHeader
	for _, h := range headers {
		if fileOffset >= h.Off && fileOffset < h.Off+h.Memsz {
			if ph != nil {
				// Assuming no other bugs, this can only happen if we have two or
				// more small program segments that fit on the same page, and a
				// segment other than the last one includes uninitialized data, or
				// if the debug binary used for symbolization is stripped of some
				// sections, so segment file sizes are smaller than memory sizes.
				return nil, fmt.Errorf("found second program header (%#v) that matches file offset %x, first program header is %#v. Is this a stripped binary, or does the first program segment contain uninitialized data?", *h, fileOffset, *ph)
			}
			ph = h
		}
	}
	if ph == nil {
		return nil, fmt.Errorf("no program header matches file offset %x", fileOffset)
	}
	return ph, nil
}
