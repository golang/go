// Copyright 2015 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

/*
Asm, typically invoked as “go tool asm”, assembles the source file into an object
file named for the basename of the argument source file with a .o suffix. The
object file can then be combined with other objects into a package archive.

# Command Line

Usage:

	go tool asm [flags] file

The specified file must be a Go assembly file.
The same assembler is used for all target operating systems and architectures.
The GOOS and GOARCH environment variables set the desired target.

Flags:

	-D name[=value]
		Predefine symbol name with an optional simple value.
		Can be repeated to define multiple symbols.
	-I dir1 -I dir2
		Search for #include files in dir1, dir2, etc,
		after consulting $GOROOT/pkg/$GOOS_$GOARCH.
	-S
		Print assembly and machine code.
	-V
		Print assembler version and exit.
	-debug
		Dump instructions as they are parsed.
	-dynlink
		Support references to Go symbols defined in other shared libraries.
	-gensymabis
		Write symbol ABI information to output file. Don't assemble.
	-o file
		Write output to file. The default is foo.o for /a/b/c/foo.s.
	-shared
		Generate code that can be linked into a shared library.
	-spectre list
		Enable spectre mitigations in list (all, ret).
	-trimpath prefix
		Remove prefix from recorded source file paths.

Input language:

The assembler uses mostly the same syntax for all architectures,
the main variation having to do with addressing modes. Input is
run through a simplified C preprocessor that implements #include,
#define, #ifdef/endif, but not #if or ##.

For more information, see https://golang.org/doc/asm.
*/
package main
