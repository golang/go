// Copyright 2009 The Go Authors.  All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

// This package provides access to DWARF debugging information
// loaded from executable files, as defined in the DWARF 2.0 Standard
// at http://dwarfstd.org/doc/dwarf-2.0.0.pdf
package dwarf

import (
	"encoding/binary"
	"os"
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

	// parsed data
	abbrevCache map[uint32]abbrevTable
	addrsize    int
	order       binary.ByteOrder
	typeCache   map[Offset]Type
	unit        []unit
}

// New returns a new Data object initialized from the given parameters.
// Clients should typically use [TODO(rsc): method to be named later] instead of calling
// New directly.
//
// The []byte arguments are the data from the corresponding debug section
// in the object file; for example, for an ELF object, abbrev is the contents of
// the ".debug_abbrev" section.
func New(abbrev, aranges, frame, info, line, pubnames, ranges, str []byte) (*Data, os.Error) {
	d := &Data{
		abbrev:      abbrev,
		aranges:     aranges,
		frame:       frame,
		info:        info,
		line:        line,
		pubnames:    pubnames,
		ranges:      ranges,
		str:         str,
		abbrevCache: make(map[uint32]abbrevTable),
		typeCache:   make(map[Offset]Type),
	}

	// Sniff .debug_info to figure out byte order.
	// bytes 4:6 are the version, a tiny 16-bit number (1, 2, 3).
	if len(d.info) < 6 {
		return nil, DecodeError{"info", Offset(len(d.info)), "too short"}
	}
	x, y := d.info[4], d.info[5]
	switch {
	case x == 0 && y == 0:
		return nil, DecodeError{"info", 4, "unsupported version 0"}
	case x == 0:
		d.order = binary.BigEndian
	case y == 0:
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
