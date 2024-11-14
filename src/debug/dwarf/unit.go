// Copyright 2009 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

package dwarf

import (
	"sort"
	"strconv"
)

// DWARF debug info is split into a sequence of compilation units.
// Each unit has its own abbreviation table and address size.

type unit struct {
	base   Offset // byte offset of header within the aggregate info
	off    Offset // byte offset of data within the aggregate info
	data   []byte
	atable abbrevTable
	asize  int
	vers   int
	utype  uint8 // DWARF 5 unit type
	is64   bool  // True for 64-bit DWARF format
}

// Implement the dataFormat interface.

func (u *unit) version() int {
	return u.vers
}

func (u *unit) dwarf64() (bool, bool) {
	return u.is64, true
}

func (u *unit) addrsize() int {
	return u.asize
}

func (d *Data) parseUnits() ([]unit, error) {
	// Count units.
	nunit := 0
	b := makeBuf(d, unknownFormat{}, "info", 0, d.info)
	for len(b.data) > 0 {
		len, _ := b.unitLength()
		if len != Offset(uint32(len)) {
			b.error("unit length overflow")
			break
		}
		b.skip(int(len))
		if len > 0 {
			nunit++
		}
	}
	if b.err != nil {
		return nil, b.err
	}

	// Again, this time writing them down.
	b = makeBuf(d, unknownFormat{}, "info", 0, d.info)
	units := make([]unit, nunit)
	for i := range units {
		u := &units[i]
		u.base = b.off
		var n Offset
		if b.err != nil {
			return nil, b.err
		}
		for n == 0 {
			n, u.is64 = b.unitLength()
		}
		dataOff := b.off
		vers := b.uint16()
		if vers < 2 || vers > 5 {
			b.error("unsupported DWARF version " + strconv.Itoa(int(vers)))
			break
		}
		u.vers = int(vers)
		if vers >= 5 {
			u.utype = b.uint8()
			u.asize = int(b.uint8())
		}
		var abbrevOff uint64
		if u.is64 {
			abbrevOff = b.uint64()
		} else {
			abbrevOff = uint64(b.uint32())
		}
		atable, err := d.parseAbbrev(abbrevOff, u.vers)
		if err != nil {
			if b.err == nil {
				b.err = err
			}
			break
		}
		u.atable = atable
		if vers < 5 {
			u.asize = int(b.uint8())
		}

		switch u.utype {
		case utSkeleton, utSplitCompile:
			b.uint64() // unit ID
		case utType, utSplitType:
			b.uint64()  // type signature
			if u.is64 { // type offset
				b.uint64()
			} else {
				b.uint32()
			}
		}

		u.off = b.off
		u.data = b.bytes(int(n - (b.off - dataOff)))
	}
	if b.err != nil {
		return nil, b.err
	}
	return units, nil
}

// offsetToUnit returns the index of the unit containing offset off.
// It returns -1 if no unit contains this offset.
func (d *Data) offsetToUnit(off Offset) int {
	// Find the unit after off
	next := sort.Search(len(d.unit), func { i -> d.unit[i].off > off })
	if next == 0 {
		return -1
	}
	u := &d.unit[next-1]
	if u.off <= off && off < u.off+Offset(len(u.data)) {
		return next - 1
	}
	return -1
}
