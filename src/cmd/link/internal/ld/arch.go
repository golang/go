// Copyright 2015 The Go Authors.  All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

package ld

import "encoding/binary"

var Linkarm = LinkArch{
	ByteOrder: binary.LittleEndian,
	Name:      "arm",
	Thechar:   '5',
	Minlc:     4,
	Ptrsize:   4,
	Regsize:   4,
}

var Linkarm64 = LinkArch{
	ByteOrder: binary.LittleEndian,
	Name:      "arm64",
	Thechar:   '7',
	Minlc:     4,
	Ptrsize:   8,
	Regsize:   8,
}

var Linkamd64 = LinkArch{
	ByteOrder: binary.LittleEndian,
	Name:      "amd64",
	Thechar:   '6',
	Minlc:     1,
	Ptrsize:   8,
	Regsize:   8,
}

var Linkamd64p32 = LinkArch{
	ByteOrder: binary.LittleEndian,
	Name:      "amd64p32",
	Thechar:   '6',
	Minlc:     1,
	Ptrsize:   4,
	Regsize:   8,
}

var Link386 = LinkArch{
	ByteOrder: binary.LittleEndian,
	Name:      "386",
	Thechar:   '8',
	Minlc:     1,
	Ptrsize:   4,
	Regsize:   4,
}

var Linkppc64 = LinkArch{
	ByteOrder: binary.BigEndian,
	Name:      "ppc64",
	Thechar:   '9',
	Minlc:     4,
	Ptrsize:   8,
	Regsize:   8,
}

var Linkppc64le = LinkArch{
	ByteOrder: binary.LittleEndian,
	Name:      "ppc64le",
	Thechar:   '9',
	Minlc:     4,
	Ptrsize:   8,
	Regsize:   8,
}

var Linkmips64 = LinkArch{
	ByteOrder: binary.BigEndian,
	Name:      "mips64",
	Thechar:   '0',
	Minlc:     4,
	Ptrsize:   8,
	Regsize:   8,
}

var Linkmips64le = LinkArch{
	ByteOrder: binary.LittleEndian,
	Name:      "mips64le",
	Thechar:   '0',
	Minlc:     4,
	Ptrsize:   8,
	Regsize:   8,
}
