// Copyright 2013 The Go Authors.  All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

// This file defines flags attached to various functions
// and data objects.  The compilers, assemblers, and linker must
// all agree on these values.

// Don't profile the marked routine.  This flag is deprecated.
#define NOPROF	1
// It is ok for the linker to get multiple of these symbols.  It will
// pick one of the duplicates to use.
#define DUPOK	2
// Don't insert stack check preamble.
#define NOSPLIT	4
// Put this data in a read-only section.
#define RODATA	8
// This data contains no pointers.
#define NOPTR	16
// This is a wrapper function and should not count as disabling 'recover'.
#define WRAPPER 32
// This function uses its incoming context register.
#define NEEDCTXT 64

/*c2go
enum
{
	NOPROF = 1,
	DUPOK = 2,
	NOSPLIT = 4,
	RODATA = 8,
	NOPTR = 16,
	WRAPPER = 32,
	NEEDCTXT = 64,
};
*/
