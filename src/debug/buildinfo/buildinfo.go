// Copyright 2021 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

// Package buildinfo provides access to information embedded in a Go binary
// about how it was built. This includes the Go toolchain version, and the
// set of modules used (for binaries built in module mode).
//
// Build information is available for the currently running binary in
// runtime/debug.ReadBuildInfo.
package buildinfo

import (
	"bytes"
	"debug/elf"
	"debug/macho"
	"debug/pe"
	"debug/plan9obj"
	"encoding/binary"
	"errors"
	"fmt"
	"internal/saferio"
	"internal/xcoff"
	"io"
	"io/fs"
	"os"
	"runtime/debug"
	_ "unsafe" // for linkname
)

// Type alias for build info. We cannot move the types here, since
// runtime/debug would need to import this package, which would make it
// a much larger dependency.
type BuildInfo = debug.BuildInfo

// errUnrecognizedFormat is returned when a given executable file doesn't
// appear to be in a known format, or it breaks the rules of that format,
// or when there are I/O errors reading the file.
var errUnrecognizedFormat = errors.New("unrecognized file format")

// errNotGoExe is returned when a given executable file is valid but does
// not contain Go build information.
//
// errNotGoExe should be an internal detail,
// but widely used packages access it using linkname.
// Notable members of the hall of shame include:
//   - github.com/quay/claircore
//
// Do not remove or change the type signature.
// See go.dev/issue/67401.
//
//go:linkname errNotGoExe
var errNotGoExe = errors.New("not a Go executable")

// The build info blob left by the linker is identified by a 32-byte header,
// consisting of buildInfoMagic (14 bytes), followed by version-dependent
// fields.
var buildInfoMagic = []byte("\xff Go buildinf:")

const (
	buildInfoAlign      = 16
	buildInfoHeaderSize = 32
)

// ReadFile returns build information embedded in a Go binary
// file at the given path. Most information is only available for binaries built
// with module support.
func ReadFile(name string) (info *BuildInfo, err error) {
	defer func() {
		if pathErr := (*fs.PathError)(nil); errors.As(err, &pathErr) {
			err = fmt.Errorf("could not read Go build info: %w", err)
		} else if err != nil {
			err = fmt.Errorf("could not read Go build info from %s: %w", name, err)
		}
	}()

	f, err := os.Open(name)
	if err != nil {
		return nil, err
	}
	defer f.Close()
	return Read(f)
}

// Read returns build information embedded in a Go binary file
// accessed through the given ReaderAt. Most information is only available for
// binaries built with module support.
func Read(r io.ReaderAt) (*BuildInfo, error) {
	vers, mod, err := readRawBuildInfo(r)
	if err != nil {
		return nil, err
	}
	bi, err := debug.ParseBuildInfo(mod)
	if err != nil {
		return nil, err
	}
	bi.GoVersion = vers
	return bi, nil
}

type exe interface {
	// ReadData reads and returns up to size bytes starting at virtual address addr.
	ReadData(addr, size uint64) ([]byte, error)

	// DataStart returns the virtual address and size of the segment or section that
	// should contain build information. This is either a specially named section
	// or the first writable non-zero data segment.
	DataStart() (uint64, uint64)
}

// readRawBuildInfo extracts the Go toolchain version and module information
// strings from a Go binary. On success, vers should be non-empty. mod
// is empty if the binary was not built with modules enabled.
func readRawBuildInfo(r io.ReaderAt) (vers, mod string, err error) {
	// Read the first bytes of the file to identify the format, then delegate to
	// a format-specific function to load segment and section headers.
	ident := make([]byte, 16)
	if n, err := r.ReadAt(ident, 0); n < len(ident) || err != nil {
		return "", "", errUnrecognizedFormat
	}

	var x exe
	switch {
	case bytes.HasPrefix(ident, []byte("\x7FELF")):
		f, err := elf.NewFile(r)
		if err != nil {
			return "", "", errUnrecognizedFormat
		}
		x = &elfExe{f}
	case bytes.HasPrefix(ident, []byte("MZ")):
		f, err := pe.NewFile(r)
		if err != nil {
			return "", "", errUnrecognizedFormat
		}
		x = &peExe{f}
	case bytes.HasPrefix(ident, []byte("\xFE\xED\xFA")) || bytes.HasPrefix(ident[1:], []byte("\xFA\xED\xFE")):
		f, err := macho.NewFile(r)
		if err != nil {
			return "", "", errUnrecognizedFormat
		}
		x = &machoExe{f}
	case bytes.HasPrefix(ident, []byte("\xCA\xFE\xBA\xBE")) || bytes.HasPrefix(ident, []byte("\xCA\xFE\xBA\xBF")):
		f, err := macho.NewFatFile(r)
		if err != nil || len(f.Arches) == 0 {
			return "", "", errUnrecognizedFormat
		}
		x = &machoExe{f.Arches[0].File}
	case bytes.HasPrefix(ident, []byte{0x01, 0xDF}) || bytes.HasPrefix(ident, []byte{0x01, 0xF7}):
		f, err := xcoff.NewFile(r)
		if err != nil {
			return "", "", errUnrecognizedFormat
		}
		x = &xcoffExe{f}
	case hasPlan9Magic(ident):
		f, err := plan9obj.NewFile(r)
		if err != nil {
			return "", "", errUnrecognizedFormat
		}
		x = &plan9objExe{f}
	default:
		return "", "", errUnrecognizedFormat
	}

	// Read segment or section to find the build info blob.
	// On some platforms, the blob will be in its own section, and DataStart
	// returns the address of that section. On others, it's somewhere in the
	// data segment; the linker puts it near the beginning.
	// See cmd/link/internal/ld.Link.buildinfo.
	dataAddr, dataSize := x.DataStart()
	if dataSize == 0 {
		return "", "", errNotGoExe
	}

	addr, err := searchMagic(x, dataAddr, dataSize)
	if err != nil {
		return "", "", err
	}

	// Read in the full header first.
	header, err := x.ReadData(addr, buildInfoHeaderSize)
	if err != nil {
		return "", "", err
	}

	const (
		ptrSizeOffset = 14
		flagsOffset   = 15
		versPtrOffset = 16

		flagsEndianMask   = 0x1
		flagsEndianLittle = 0x0
		flagsEndianBig    = 0x1

		flagsVersionMask = 0x2
		flagsVersionPtr  = 0x0
		flagsVersionInl  = 0x2
	)

	// Decode the blob. The blob is a 32-byte header, optionally followed
	// by 2 varint-prefixed string contents.
	//
	// type buildInfoHeader struct {
	// 	magic       [14]byte
	// 	ptrSize     uint8 // used if flagsVersionPtr
	// 	flags       uint8
	// 	versPtr     targetUintptr // used if flagsVersionPtr
	// 	modPtr      targetUintptr // used if flagsVersionPtr
	// }
	//
	// The version bit of the flags field determines the details of the format.
	//
	// Prior to 1.18, the flags version bit is flagsVersionPtr. In this
	// case, the header includes pointers to the version and modinfo Go
	// strings in the header. The ptrSize field indicates the size of the
	// pointers and the endian bit of the flag indicates the pointer
	// endianness.
	//
	// Since 1.18, the flags version bit is flagsVersionInl. In this case,
	// the header is followed by the string contents inline as
	// length-prefixed (as varint) string contents. First is the version
	// string, followed immediately by the modinfo string.
	flags := header[flagsOffset]
	if flags&flagsVersionMask == flagsVersionInl {
		vers, addr, err = decodeString(x, addr+buildInfoHeaderSize)
		if err != nil {
			return "", "", err
		}
		mod, _, err = decodeString(x, addr)
		if err != nil {
			return "", "", err
		}
	} else {
		// flagsVersionPtr (<1.18)
		ptrSize := int(header[ptrSizeOffset])
		bigEndian := flags&flagsEndianMask == flagsEndianBig
		var bo binary.ByteOrder
		if bigEndian {
			bo = binary.BigEndian
		} else {
			bo = binary.LittleEndian
		}
		var readPtr func([]byte) uint64
		if ptrSize == 4 {
			readPtr = func(b []byte) uint64 { return uint64(bo.Uint32(b)) }
		} else if ptrSize == 8 {
			readPtr = bo.Uint64
		} else {
			return "", "", errNotGoExe
		}
		vers = readString(x, ptrSize, readPtr, readPtr(header[versPtrOffset:]))
		mod = readString(x, ptrSize, readPtr, readPtr(header[versPtrOffset+ptrSize:]))
	}
	if vers == "" {
		return "", "", errNotGoExe
	}
	if len(mod) >= 33 && mod[len(mod)-17] == '\n' {
		// Strip module framing: sentinel strings delimiting the module info.
		// These are cmd/go/internal/modload.infoStart and infoEnd.
		mod = mod[16 : len(mod)-16]
	} else {
		mod = ""
	}

	return vers, mod, nil
}

func hasPlan9Magic(magic []byte) bool {
	if len(magic) >= 4 {
		m := binary.BigEndian.Uint32(magic)
		switch m {
		case plan9obj.Magic386, plan9obj.MagicAMD64, plan9obj.MagicARM:
			return true
		}
	}
	return false
}

func decodeString(x exe, addr uint64) (string, uint64, error) {
	// varint length followed by length bytes of data.

	// N.B. ReadData reads _up to_ size bytes from the section containing
	// addr. So we don't need to check that size doesn't overflow the
	// section.
	b, err := x.ReadData(addr, binary.MaxVarintLen64)
	if err != nil {
		return "", 0, err
	}

	length, n := binary.Uvarint(b)
	if n <= 0 {
		return "", 0, errNotGoExe
	}
	addr += uint64(n)

	b, err = x.ReadData(addr, length)
	if err != nil {
		return "", 0, err
	}
	if uint64(len(b)) < length {
		// Section ended before we could read the full string.
		return "", 0, errNotGoExe
	}

	return string(b), addr + length, nil
}

// readString returns the string at address addr in the executable x.
func readString(x exe, ptrSize int, readPtr func([]byte) uint64, addr uint64) string {
	hdr, err := x.ReadData(addr, uint64(2*ptrSize))
	if err != nil || len(hdr) < 2*ptrSize {
		return ""
	}
	dataAddr := readPtr(hdr)
	dataLen := readPtr(hdr[ptrSize:])
	data, err := x.ReadData(dataAddr, dataLen)
	if err != nil || uint64(len(data)) < dataLen {
		return ""
	}
	return string(data)
}

const searchChunkSize = 1 << 20 // 1 MB

// searchMagic returns the aligned first instance of buildInfoMagic in the data
// range [addr, addr+size). Returns false if not found.
func searchMagic(x exe, start, size uint64) (uint64, error) {
	end := start + size
	if end < start {
		// Overflow.
		return 0, errUnrecognizedFormat
	}

	// Round up start; magic can't occur in the initial unaligned portion.
	start = (start + buildInfoAlign - 1) &^ (buildInfoAlign - 1)
	if start >= end {
		return 0, errNotGoExe
	}

	for start < end {
		// Read in chunks to avoid consuming too much memory if data is large.
		//
		// Normally it would be somewhat painful to handle the magic crossing a
		// chunk boundary, but since it must be 16-byte aligned we know it will
		// fall within a single chunk.
		remaining := end - start
		chunkSize := uint64(searchChunkSize)
		if chunkSize > remaining {
			chunkSize = remaining
		}

		data, err := x.ReadData(start, chunkSize)
		if err != nil {
			return 0, err
		}

		for len(data) > 0 {
			i := bytes.Index(data, buildInfoMagic)
			if i < 0 {
				break
			}
			if remaining-uint64(i) < buildInfoHeaderSize {
				// Found magic, but not enough space left for the full header.
				return 0, errNotGoExe
			}
			if i%buildInfoAlign != 0 {
				// Found magic, but misaligned. Keep searching.
				data = data[(i+buildInfoAlign-1)&^(buildInfoAlign-1):]
				continue
			}
			// Good match!
			return start + uint64(i), nil
		}

		start += chunkSize
	}

	return 0, errNotGoExe
}

// elfExe is the ELF implementation of the exe interface.
type elfExe struct {
	f *elf.File
}

func (x *elfExe) ReadData(addr, size uint64) ([]byte, error) {
	for _, prog := range x.f.Progs {
		if prog.Vaddr <= addr && addr <= prog.Vaddr+prog.Filesz-1 {
			n := prog.Vaddr + prog.Filesz - addr
			if n > size {
				n = size
			}
			return saferio.ReadDataAt(prog, n, int64(addr-prog.Vaddr))
		}
	}
	return nil, errUnrecognizedFormat
}

func (x *elfExe) DataStart() (uint64, uint64) {
	for _, s := range x.f.Sections {
		if s.Name == ".go.buildinfo" {
			return s.Addr, s.Size
		}
	}
	for _, p := range x.f.Progs {
		if p.Type == elf.PT_LOAD && p.Flags&(elf.PF_X|elf.PF_W) == elf.PF_W {
			return p.Vaddr, p.Memsz
		}
	}
	return 0, 0
}

// peExe is the PE (Windows Portable Executable) implementation of the exe interface.
type peExe struct {
	f *pe.File
}

func (x *peExe) imageBase() uint64 {
	switch oh := x.f.OptionalHeader.(type) {
	case *pe.OptionalHeader32:
		return uint64(oh.ImageBase)
	case *pe.OptionalHeader64:
		return oh.ImageBase
	}
	return 0
}

func (x *peExe) ReadData(addr, size uint64) ([]byte, error) {
	addr -= x.imageBase()
	for _, sect := range x.f.Sections {
		if uint64(sect.VirtualAddress) <= addr && addr <= uint64(sect.VirtualAddress+sect.Size-1) {
			n := uint64(sect.VirtualAddress+sect.Size) - addr
			if n > size {
				n = size
			}
			return saferio.ReadDataAt(sect, n, int64(addr-uint64(sect.VirtualAddress)))
		}
	}
	return nil, errUnrecognizedFormat
}

func (x *peExe) DataStart() (uint64, uint64) {
	// Assume data is first writable section.
	const (
		IMAGE_SCN_CNT_CODE               = 0x00000020
		IMAGE_SCN_CNT_INITIALIZED_DATA   = 0x00000040
		IMAGE_SCN_CNT_UNINITIALIZED_DATA = 0x00000080
		IMAGE_SCN_MEM_EXECUTE            = 0x20000000
		IMAGE_SCN_MEM_READ               = 0x40000000
		IMAGE_SCN_MEM_WRITE              = 0x80000000
		IMAGE_SCN_MEM_DISCARDABLE        = 0x2000000
		IMAGE_SCN_LNK_NRELOC_OVFL        = 0x1000000
		IMAGE_SCN_ALIGN_32BYTES          = 0x600000
	)
	for _, sect := range x.f.Sections {
		if sect.VirtualAddress != 0 && sect.Size != 0 &&
			sect.Characteristics&^IMAGE_SCN_ALIGN_32BYTES == IMAGE_SCN_CNT_INITIALIZED_DATA|IMAGE_SCN_MEM_READ|IMAGE_SCN_MEM_WRITE {
			return uint64(sect.VirtualAddress) + x.imageBase(), uint64(sect.VirtualSize)
		}
	}
	return 0, 0
}

// machoExe is the Mach-O (Apple macOS/iOS) implementation of the exe interface.
type machoExe struct {
	f *macho.File
}

func (x *machoExe) ReadData(addr, size uint64) ([]byte, error) {
	for _, load := range x.f.Loads {
		seg, ok := load.(*macho.Segment)
		if !ok {
			continue
		}
		if seg.Addr <= addr && addr <= seg.Addr+seg.Filesz-1 {
			if seg.Name == "__PAGEZERO" {
				continue
			}
			n := seg.Addr + seg.Filesz - addr
			if n > size {
				n = size
			}
			return saferio.ReadDataAt(seg, n, int64(addr-seg.Addr))
		}
	}
	return nil, errUnrecognizedFormat
}

func (x *machoExe) DataStart() (uint64, uint64) {
	// Look for section named "__go_buildinfo".
	for _, sec := range x.f.Sections {
		if sec.Name == "__go_buildinfo" {
			return sec.Addr, sec.Size
		}
	}
	// Try the first non-empty writable segment.
	const RW = 3
	for _, load := range x.f.Loads {
		seg, ok := load.(*macho.Segment)
		if ok && seg.Addr != 0 && seg.Filesz != 0 && seg.Prot == RW && seg.Maxprot == RW {
			return seg.Addr, seg.Memsz
		}
	}
	return 0, 0
}

// xcoffExe is the XCOFF (AIX eXtended COFF) implementation of the exe interface.
type xcoffExe struct {
	f *xcoff.File
}

func (x *xcoffExe) ReadData(addr, size uint64) ([]byte, error) {
	for _, sect := range x.f.Sections {
		if sect.VirtualAddress <= addr && addr <= sect.VirtualAddress+sect.Size-1 {
			n := sect.VirtualAddress + sect.Size - addr
			if n > size {
				n = size
			}
			return saferio.ReadDataAt(sect, n, int64(addr-sect.VirtualAddress))
		}
	}
	return nil, errors.New("address not mapped")
}

func (x *xcoffExe) DataStart() (uint64, uint64) {
	if s := x.f.SectionByType(xcoff.STYP_DATA); s != nil {
		return s.VirtualAddress, s.Size
	}
	return 0, 0
}

// plan9objExe is the Plan 9 a.out implementation of the exe interface.
type plan9objExe struct {
	f *plan9obj.File
}

func (x *plan9objExe) DataStart() (uint64, uint64) {
	if s := x.f.Section("data"); s != nil {
		return uint64(s.Offset), uint64(s.Size)
	}
	return 0, 0
}

func (x *plan9objExe) ReadData(addr, size uint64) ([]byte, error) {
	for _, sect := range x.f.Sections {
		if uint64(sect.Offset) <= addr && addr <= uint64(sect.Offset+sect.Size-1) {
			n := uint64(sect.Offset+sect.Size) - addr
			if n > size {
				n = size
			}
			return saferio.ReadDataAt(sect, n, int64(addr-uint64(sect.Offset)))
		}
	}
	return nil, errors.New("address not mapped")
}
