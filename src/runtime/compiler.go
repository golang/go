// Copyright 2012 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

package runtime

// Compiler is the name of the compiler toolchain that built the
// running binary. Known toolchains are:
//
//	gc      Also known as cmd/compile.
//	gccgo   The gccgo front end, part of the GCC compiler suite.
//
const Compiler = "gc"
