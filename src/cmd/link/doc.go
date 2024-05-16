// Copyright 2009 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

/*
Link, typically invoked as “go tool link”, reads the Go archive or object
for a package main, along with its dependencies, and combines them
into an executable binary.

# Command Line

Usage:

	go tool link [flags] main.a

Flags:

	-B note
		Add an ELF_NT_GNU_BUILD_ID note when using ELF.
		The value should start with 0x and be an even number of hex digits.
		Alternatively, you can pass "gobuildid" in order to derive the
		GNU build ID from the Go build ID.
	-E entry
		Set entry symbol name.
	-H type
		Set executable format type.
		The default format is inferred from GOOS and GOARCH.
		On Windows, -H windowsgui writes a "GUI binary" instead of a "console binary."
	-I interpreter
		Set the ELF dynamic linker to use.
	-L dir1 -L dir2
		Search for imported packages in dir1, dir2, etc,
		after consulting $GOROOT/pkg/$GOOS_$GOARCH.
	-R quantum
		Set address rounding quantum.
	-T address
		Set the start address of text symbols.
	-V
		Print linker version and exit.
	-X importpath.name=value
		Set the value of the string variable in importpath named name to value.
		This is only effective if the variable is declared in the source code either uninitialized
		or initialized to a constant string expression. -X will not work if the initializer makes
		a function call or refers to other variables.
		Note that before Go 1.5 this option took two separate arguments.
	-asan
		Link with C/C++ address sanitizer support.
	-aslr
		Enable ASLR for buildmode=c-shared on windows (default true).
	-bindnow
		Mark a dynamically linked ELF object for immediate function binding (default false).
	-buildid id
		Record id as Go toolchain build id.
	-buildmode mode
		Set build mode (default exe).
	-c
		Dump call graphs.
	-compressdwarf
		Compress DWARF if possible (default true).
	-cpuprofile file
		Write CPU profile to file.
	-d
		Disable generation of dynamic executables.
		The emitted code is the same in either case; the option
		controls only whether a dynamic header is included.
		The dynamic header is on by default, even without any
		references to dynamic libraries, because many common
		system tools now assume the presence of the header.
	-dumpdep
		Dump symbol dependency graph.
	-extar ar
		Set the external archive program (default "ar").
		Used only for -buildmode=c-archive.
	-extld linker
		Set the external linker (default "clang" or "gcc").
	-extldflags flags
		Set space-separated flags to pass to the external linker.
	-f
		Ignore version mismatch in the linked archives.
	-g
		Disable Go package data checks.
	-importcfg file
		Read import configuration from file.
		In the file, set packagefile, packageshlib to specify import resolution.
	-installsuffix suffix
		Look for packages in $GOROOT/pkg/$GOOS_$GOARCH_suffix
		instead of $GOROOT/pkg/$GOOS_$GOARCH.
	-k symbol
		Set field tracking symbol. Use this flag when GOEXPERIMENT=fieldtrack is set.
	-libgcc file
		Set name of compiler support library.
		This is only used in internal link mode.
		If not set, default value comes from running the compiler,
		which may be set by the -extld option.
		Set to "none" to use no support library.
	-linkmode mode
		Set link mode (internal, external, auto).
		This sets the linking mode as described in cmd/cgo/doc.go.
	-linkshared
		Link against installed Go shared libraries (experimental).
	-memprofile file
		Write memory profile to file.
	-memprofilerate rate
		Set runtime.MemProfileRate to rate.
	-msan
		Link with C/C++ memory sanitizer support.
	-o file
		Write output to file (default a.out, or a.out.exe on Windows).
	-pluginpath path
		The path name used to prefix exported plugin symbols.
	-r dir1:dir2:...
		Set the ELF dynamic linker search path.
	-race
		Link with race detection libraries.
	-s
		Omit the symbol table and debug information.
	-tmpdir dir
		Write temporary files to dir.
		Temporary files are only used in external linking mode.
	-v
		Print trace of linker operations.
	-w
		Omit the DWARF symbol table.
*/
package main
