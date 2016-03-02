// Copyright 2015 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

/*
Asm, typically invoked as ``go tool asm'', assembles the source file into an object
file named for the basename of the argument source file with a .o suffix. The
object file can then be combined with other objects into a package archive.

Command Line

Usage:

	go tool asm [flags] file

The specified file must be a Go assembly file.
The same assembler is used for all target operating systems and architectures.
The GOOS and GOARCH environment variables set the desired target.

Flags:

	-D value
		predefined symbol with optional simple value -D=identifier=value;
		can be set multiple times
	-I value
		include directory; can be set multiple times
	-S	print assembly and machine code
	-debug
		dump instructions as they are parsed
	-dynlink
		support references to Go symbols defined in other shared libraries
	-o string
		output file; default foo.o for /a/b/c/foo.s
	-shared
		generate code that can be linked into a shared library
	-trimpath string
		remove prefix from recorded source file paths

Input language:

The assembler uses mostly the same syntax for all architectures,
the main variation having to do with addressing modes. Input is
run through a simplified C preprocessor that implements #include,
#define, #ifdef/endif, but not #if or ##.

For more information, see https://golang.org/doc/asm.
*/
package main
