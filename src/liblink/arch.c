// Copyright 2015 The Go Authors.  All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

#include <u.h>
#include <libc.h>
#include <bio.h>
#include <link.h>

LinkArch linkarm = {
	.name = "arm",
	.thechar = '5',
	.endian = LittleEndian,

	.minlc = 4,
	.ptrsize = 4,
	.regsize = 4,
};

LinkArch linkamd64 = {
	.name = "amd64",
	.thechar = '6',
	.endian = LittleEndian,

	.minlc = 1,
	.ptrsize = 8,
	.regsize = 8,
};

LinkArch linkamd64p32 = {
	.name = "amd64p32",
	.thechar = '6',
	.endian = LittleEndian,

	.minlc = 1,
	.ptrsize = 4,
	.regsize = 8,
};

LinkArch link386 = {
	.name = "386",
	.thechar = '8',
	.endian = LittleEndian,

	.minlc = 1,
	.ptrsize = 4,
	.regsize = 4,
};

LinkArch linkppc64 = {
	.name = "ppc64",
	.thechar = '9',
	.endian = BigEndian,

	.minlc = 4,
	.ptrsize = 8,
	.regsize = 8,
};

LinkArch linkppc64le = {
	.name = "ppc64le",
	.thechar = '9',
	.endian = LittleEndian,

	.minlc = 4,
	.ptrsize = 8,
	.regsize = 8,
};
