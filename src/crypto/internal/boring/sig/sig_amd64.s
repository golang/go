// Copyright 2017 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

#include "textflag.h"

// These functions are no-ops, but you can search for their implementations
// to find out whether they are linked into a particular binary.
//
// Each function consists of a two-byte jump over the next 29-bytes,
// then a 5-byte indicator sequence unlikely to occur in real x86 instructions,
// then a randomly-chosen 24-byte sequence, and finally a return instruction
// (the target of the jump).
//
// These sequences are known to rsc.io/goversion.

#define START \
	BYTE $0xEB; BYTE $0x1D; BYTE $0xF4; BYTE $0x48; BYTE $0xF4; BYTE $0x4B; BYTE $0xF4

#define END \
	BYTE $0xC3

// BoringCrypto indicates that BoringCrypto (in particular, its func init) is present.
TEXT ·BoringCrypto(SB),NOSPLIT,$0
	START
	BYTE $0xB3; BYTE $0x32; BYTE $0xF5; BYTE $0x28;
	BYTE $0x13; BYTE $0xA3; BYTE $0xB4; BYTE $0x50;
	BYTE $0xD4; BYTE $0x41; BYTE $0xCC; BYTE $0x24;
	BYTE $0x85; BYTE $0xF0; BYTE $0x01; BYTE $0x45;
	BYTE $0x4E; BYTE $0x92; BYTE $0x10; BYTE $0x1B;
	BYTE $0x1D; BYTE $0x2F; BYTE $0x19; BYTE $0x50;
	END

// StandardCrypto indicates that standard Go crypto is present.
TEXT ·StandardCrypto(SB),NOSPLIT,$0
	START
	BYTE $0xba; BYTE $0xee; BYTE $0x4d; BYTE $0xfa;
	BYTE $0x98; BYTE $0x51; BYTE $0xca; BYTE $0x56;
	BYTE $0xa9; BYTE $0x11; BYTE $0x45; BYTE $0xe8;
	BYTE $0x3e; BYTE $0x99; BYTE $0xc5; BYTE $0x9c;
	BYTE $0xf9; BYTE $0x11; BYTE $0xcb; BYTE $0x8e;
	BYTE $0x80; BYTE $0xda;  BYTE $0xf1; BYTE $0x2f;
	END

// FIPSOnly indicates that crypto/tls/fipsonly is present.
TEXT ·FIPSOnly(SB),NOSPLIT,$0
	START
	BYTE $0x36; BYTE $0x3C; BYTE $0xB9; BYTE $0xCE;
	BYTE $0x9D; BYTE $0x68; BYTE $0x04; BYTE $0x7D;
	BYTE $0x31; BYTE $0xF2; BYTE $0x8D; BYTE $0x32;
	BYTE $0x5D; BYTE $0x5C; BYTE $0xA5; BYTE $0x87;
	BYTE $0x3F; BYTE $0x5D; BYTE $0x80; BYTE $0xCA;
	BYTE $0xF6; BYTE $0xD6; BYTE $0x15; BYTE $0x1B;
	END
