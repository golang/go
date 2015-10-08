// Copyright 2015 The Go Authors.  All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

// FIXED_FRAME defines the size of the fixed part of a stack frame. A stack
// frame looks like this:
//
// +---------------------+
// | local variable area |
// +---------------------+
// | argument area       |
// +---------------------+ <- R1+FIXED_FRAME
// | fixed area          |
// +---------------------+ <- R1
//
// So a function that sets up a stack frame at all uses as least FIXED_FRAME
// bytes of stack.  This mostly affects assembly that calls other functions
// with arguments (the arguments should be stored at FIXED_FRAME+0(R1),
// FIXED_FRAME+8(R1) etc) and some other low-level places.
//
// The reason for using a constant is when code is compiled as PIC on ppc64le
// the fixed part of the stack is 32 bytes large (although PIC is not actually
// supported yet).

#ifdef GOARCH_ppc64
#define FIXED_FRAME 8
#endif

#ifdef GOARCH_ppc64le
#define FIXED_FRAME 8
#endif
