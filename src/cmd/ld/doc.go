// Copyright 2009 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

// +build ignore

/*

Ld is the portable code for a modified version of the Plan 9 linker.  The original is documented at

	http://plan9.bell-labs.com/magic/man2html/1/8l

It reads object files (.5, .6, or .8 files) and writes a binary named for the
architecture (5.out, 6.out, 8.out) by default (if $GOOS is windows, a .exe suffix
will be appended).

Major changes include:
	- support for ELF, Mach-O and PE binary files
	- support for segmented stacks (this feature is implemented here, not in the compilers).

Original options are listed on the manual page linked above.

Usage:
	go tool 6l [flags] mainObj
Substitute 6l with 8l or 5l as appropriate.

Options new in this version:

	-d
		Elide the dynamic linking header.  With this option, the binary
		is statically linked and does not refer to a dynamic linker.  Without this option
		(the default), the binary's contents are identical but it is loaded with a dynamic
		linker. This flag cannot be used when $GOOS is windows.
	-H darwin     (only in 6l/8l)
		Write Apple Mach-O binaries (default when $GOOS is darwin)
	-H dragonfly  (only in 6l/8l)
		Write DragonFly ELF binaries (default when $GOOS is dragonfly)
	-H linux
		Write Linux ELF binaries (default when $GOOS is linux)
	-H freebsd
		Write FreeBSD ELF binaries (default when $GOOS is freebsd)
	-H netbsd
		Write NetBSD ELF binaries (default when $GOOS is netbsd)
	-H openbsd    (only in 6l/8l)
		Write OpenBSD ELF binaries (default when $GOOS is openbsd)
	-H solaris    (only in 6l)
		Write Solaris ELF binaries (default when $GOOS is solaris)
	-H windows    (only in 6l/8l)
		Write Windows PE32+ Console binaries (default when $GOOS is windows)
	-H windowsgui (only in 6l/8l)
		Write Windows PE32+ GUI binaries
	-I interpreter
		Set the ELF dynamic linker to use.
	-L dir1 -L dir2
		Search for libraries (package files) in dir1, dir2, etc.
		The default is the single location $GOROOT/pkg/$GOOS_$GOARCH.
	-r dir1:dir2:...
		Set the dynamic linker search path when using ELF.
	-s
		Omit the symbol table and debug information.
	-V
		Print the linker version.
	-w
		Omit the DWARF symbol table.
	-X symbol value
		Set the value of a string variable. The symbol name
		should be of the form importpath.name, as displayed
		in the symbol table printed by "go tool nm".
	-race
		Link with race detection libraries.
	-B value
		Add a NT_GNU_BUILD_ID note when using ELF.  The value
		should start with 0x and be an even number of hex digits.
	-Z
		Zero stack on function entry. This is expensive but it might
		be useful in cases where you are suffering from false positives
		during garbage collection and are willing to trade the CPU time
		for getting rid of the false positives.
		NOTE: it only eliminates false positives caused by other function
		calls, not false positives caused by dead temporaries stored in
		the current function call.
	-linkmode argument
		Set the linkmode.  The argument must be one of
		internal, external, or auto.  The default is auto.
		This sets the linking mode as described in
		../cgo/doc.go.
	-tmpdir dir
		Set the location to use for any temporary files.  The
		default is a newly created directory that is removed
		after the linker completes.  Temporary files are only
		used in external linking mode.
	-extld name
		Set the name of the external linker to use in external
		linking mode.  The default is "gcc".
	-extldflags flags
		Set space-separated trailing flags to pass to the
		external linker in external linking mode.  The default
		is to not pass any additional trailing flags.
*/
package main
