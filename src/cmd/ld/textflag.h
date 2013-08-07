// Copyright 2013 The Go Authors.  All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

// This file defines flags attached to various functions
// and data objects.  The compilers, assemblers, and linker must
// all agree on these values.

// Don't profile the marked routine.  This flag is deprecated.
#define NOPROF	(1<<0)
// It is ok for the linker to get multiple of these symbols.  It will
// pick one of the duplicates to use.
#define DUPOK	(1<<1)
// Don't insert stack check preamble.
#define NOSPLIT	(1<<2)
// Put this data in a read-only section.
#define RODATA	(1<<3)
// This data contains no pointers.
#define NOPTR	(1<<4)
