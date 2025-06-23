// Copyright 2013 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

// This file defines flags attached to various functions
// and data objects. The compilers, assemblers, and linker must
// all agree on these values.
//
// Keep in sync with src/cmd/internal/obj/textflag.go.

// Don't profile the marked routine. This flag is deprecated.
#define NOPROF	1
// It is ok for the linker to get multiple of these symbols. It will
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
// Allocate a word of thread local storage and store the offset from the
// thread local base to the thread local storage in this variable.
#define TLSBSS	256
// Do not insert instructions to allocate a stack frame for this function.
// Only valid on functions that declare a frame size of 0.
#define NOFRAME 512
// Function can call reflect.Type.Method or reflect.Type.MethodByName.
#define REFLECTMETHOD 1024
// Function is the outermost frame of the call stack. Call stack unwinders
// should stop at this function.
#define TOPFRAME 2048
// Function is an ABI wrapper.
#define ABIWRAPPER 4096
