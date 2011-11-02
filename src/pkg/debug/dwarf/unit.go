// Copyright 2009 The Go Authors.  All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

package dwarf

import "strconv"

// DWARF debug info is split into a sequence of compilation units.
// Each unit has its own abbreviation table and address size.

type unit struct {
	base     Offset // byte offset of header within the aggregate info
	off      Offset // byte offset of data within the aggregate info
	data     []byte
	atable   abbrevTable
	addrsize int
}

func (d *Data) parseUnits() ([]unit, error) {
	// Count units.
	nunit := 0
	b := makeBuf(d, "info", 0, d.info, 0)
	for len(b.data) > 0 {
		b.skip(int(b.uint32()))
		nunit++
	}
	if b.err != nil {
		return nil, b.err
	}

	// Again, this time writing them down.
	b = makeBuf(d, "info", 0, d.info, 0)
	units := make([]unit, nunit)
	for i := range units {
		u := &units[i]
		u.base = b.off
		n := b.uint32()
		if vers := b.uint16(); vers != 2 {
			b.error("unsupported DWARF version " + strconv.Itoa(int(vers)))
			break
		}
		atable, err := d.parseAbbrev(b.uint32())
		if err != nil {
			if b.err == nil {
				b.err = err
			}
			break
		}
		u.atable = atable
		u.addrsize = int(b.uint8())
		u.off = b.off
		u.data = b.bytes(int(n - (2 + 4 + 1)))
	}
	if b.err != nil {
		return nil, b.err
	}
	return units, nil
}
