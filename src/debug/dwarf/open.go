// Copyright 2009 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

// Package dwarf provides access to DWARF debugging information loaded from
// executable files, as defined in the DWARF 2.0 Standard at
// http://dwarfstd.org/doc/dwarf-2.0.0.pdf
package dwarf

import (
	"encoding/binary"
	"errors"
)

// Data represents the DWARF debugging information
// loaded from an executable file (for example, an ELF or Mach-O executable).
type Data struct {
	// raw data
	abbrev   []byte
	aranges  []byte
	frame    []byte
	info     []byte
	line     []byte
	pubnames []byte
	ranges   []byte
	str      []byte

	// New sections added in DWARF 5.
	addr       *debugAddr
	lineStr    []byte
	strOffsets []byte
	rngLists   *rngLists

	// parsed data
	abbrevCache map[uint64]abbrevTable
	bigEndian   bool
	order       binary.ByteOrder
	typeCache   map[Offset]Type
	typeSigs    map[uint64]*typeUnit
	unit        []unit
}

// rngLists represents the contents of a debug_rnglists section (DWARFv5).
type rngLists struct {
	is64  bool
	asize uint8
	data  []byte
	ver   uint16
}

// debugAddr represents the contents of a debug_addr section (DWARFv5).
type debugAddr struct {
	is64  bool
	asize uint8
	data  []byte
}

var errSegmentSelector = errors.New("non-zero segment_selector size not supported")

// New returns a new Data object initialized from the given parameters.
// Rather than calling this function directly, clients should typically use
// the DWARF method of the File type of the appropriate package debug/elf,
// debug/macho, or debug/pe.
//
// The []byte arguments are the data from the corresponding debug section
// in the object file; for example, for an ELF object, abbrev is the contents of
// the ".debug_abbrev" section.
func New(abbrev, aranges, frame, info, line, pubnames, ranges, str []byte) (*Data, error) {
	d := &Data{
		abbrev:      abbrev,
		aranges:     aranges,
		frame:       frame,
		info:        info,
		line:        line,
		pubnames:    pubnames,
		ranges:      ranges,
		str:         str,
		abbrevCache: make(map[uint64]abbrevTable),
		typeCache:   make(map[Offset]Type),
		typeSigs:    make(map[uint64]*typeUnit),
	}

	// Sniff .debug_info to figure out byte order.
	// 32-bit DWARF: 4 byte length, 2 byte version.
	// 64-bit DWARf: 4 bytes of 0xff, 8 byte length, 2 byte version.
	if len(d.info) < 6 {
		return nil, DecodeError{"info", Offset(len(d.info)), "too short"}
	}
	offset := 4
	if d.info[0] == 0xff && d.info[1] == 0xff && d.info[2] == 0xff && d.info[3] == 0xff {
		if len(d.info) < 14 {
			return nil, DecodeError{"info", Offset(len(d.info)), "too short"}
		}
		offset = 12
	}
	// Fetch the version, a tiny 16-bit number (1, 2, 3, 4, 5).
	x, y := d.info[offset], d.info[offset+1]
	switch {
	case x == 0 && y == 0:
		return nil, DecodeError{"info", 4, "unsupported version 0"}
	case x == 0:
		d.bigEndian = true
		d.order = binary.BigEndian
	case y == 0:
		d.bigEndian = false
		d.order = binary.LittleEndian
	default:
		return nil, DecodeError{"info", 4, "cannot determine byte order"}
	}

	u, err := d.parseUnits()
	if err != nil {
		return nil, err
	}
	d.unit = u
	return d, nil
}

// AddTypes will add one .debug_types section to the DWARF data. A
// typical object with DWARF version 4 debug info will have multiple
// .debug_types sections. The name is used for error reporting only,
// and serves to distinguish one .debug_types section from another.
func (d *Data) AddTypes(name string, types []byte) error {
	return d.parseTypes(name, types)
}

// AddSection adds another DWARF section by name. The name should be a
// DWARF section name such as ".debug_addr", ".debug_str_offsets", and
// so forth. This approach is used for new DWARF sections added in
// DWARF 5 and later.
func (d *Data) AddSection(name string, contents []byte) error {
	var err error
	switch name {
	case ".debug_addr":
		d.addr, err = d.parseAddrHeader(contents)
	case ".debug_line_str":
		d.lineStr = contents
	case ".debug_str_offsets":
		d.strOffsets = contents
	case ".debug_rnglists":
		d.rngLists, err = d.parseRngListsHeader(contents)
	}
	// Just ignore names that we don't yet support.
	return err
}

// parseRngListsHeader reads the header of a debug_rnglists section, see
// DWARFv5 section 7.28 (page 242).
func (d *Data) parseRngListsHeader(bytes []byte) (*rngLists, error) {
	rngLists := &rngLists{data: bytes}

	buf := makeBuf(d, unknownFormat{}, "rnglists", 0, bytes)
	_, rngLists.is64 = buf.unitLength()

	rngLists.ver = buf.uint16() // version

	rngLists.asize = buf.uint8()
	segsize := buf.uint8()
	if segsize != 0 {
		return nil, errSegmentSelector
	}

	// Header fields not read: offset_entry_count, offset table

	return rngLists, nil
}

func (rngLists *rngLists) version() int {
	return int(rngLists.ver)
}

func (rngLists *rngLists) dwarf64() (bool, bool) {
	return rngLists.is64, true
}

func (rngLists *rngLists) addrsize() int {
	return int(rngLists.asize)
}

// parseAddrHeader reads the header of a debug_addr section, see DWARFv5
// section 7.27 (page 241).
func (d *Data) parseAddrHeader(bytes []byte) (*debugAddr, error) {
	addr := &debugAddr{data: bytes}

	buf := makeBuf(d, unknownFormat{}, "addr", 0, bytes)
	_, addr.is64 = buf.unitLength()

	addr.asize = buf.uint8()
	segsize := buf.uint8()
	if segsize != 0 {
		return nil, errSegmentSelector
	}

	return addr, nil
}

func (addr *debugAddr) version() int {
	return 5
}

func (addr *debugAddr) dwarf64() (bool, bool) {
	return addr.is64, true
}

func (addr *debugAddr) addrsize() int {
	return int(addr.asize)
}
