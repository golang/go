// Copyright 2009 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

/*

Ld is the portable code for a modified version of the Plan 9 linker.  The original is documented at

	http://plan9.bell-labs.com/magic/man2html/1/2l

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
	-Hdarwin     (only in 6l/8l)
		Write Apple Mach-O binaries (default when $GOOS is darwin)
	-Hlinux
		Write Linux ELF binaries (default when $GOOS is linux)
	-Hfreebsd    (only in 6l/8l)
		Write FreeBSD ELF binaries (default when $GOOS is freebsd)
	-Hnetbsd     (only in 6l/8l)
		Write NetBSD ELF binaries (default when $GOOS is netbsd)
	-Hopenbsd    (only in 6l/8l)
		Write OpenBSD ELF binaries (default when $GOOS is openbsd)
	-Hwindows    (only in 6l/8l)
		Write Windows PE32+ Console binaries (default when $GOOS is windows)
	-Hwindowsgui (only in 6l/8l)
		Write Windows PE32+ GUI binaries
	-I interpreter
		Set the ELF dynamic linker to use.
	-L dir1 -L dir2
		Search for libraries (package files) in dir1, dir2, etc.
		The default is the single location $GOROOT/pkg/$GOOS_$GOARCH.
	-r dir1:dir2:...
		Set the dynamic linker search path when using ELF.
	-V
		Print the linker version.
	-X symbol value
		Set the value of an otherwise uninitialized string variable.
		The symbol name should be of the form importpath.name,
		as displayed in the symbol table printed by "go tool nm".
*/
package documentation
