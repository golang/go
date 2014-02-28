// Copyright 2009 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

// +build ignore

/*

Gc is the generic label for the family of Go compilers
that function as part of the (modified) Plan 9 tool chain.  The C compiler
documentation at

	http://plan9.bell-labs.com/sys/doc/comp.pdf     (Tools overview)
	http://plan9.bell-labs.com/sys/doc/compiler.pdf (C compiler architecture)

gives the overall design of the tool chain.  Aside from a few adapted pieces,
such as the optimizer, the Go compilers are wholly new programs.

The compiler reads in a set of Go files, typically suffixed ".go".  They
must all be part of one package.  The output is a single intermediate file
representing the "binary assembly" of the compiled package, ready as input
for the linker (6l, etc.).

The generated files contain type information about the symbols exported by
the package and about types used by symbols imported by the package from
other packages. It is therefore not necessary when compiling client C of
package P to read the files of P's dependencies, only the compiled output
of P.

Command Line

Usage:
	go tool 6g [flags] file...
The specified files must be Go source files and all part of the same package.
Substitute 6g with 8g or 5g where appropriate.

Flags:
	-o file
		output file, default file.6 for 6g, etc.
	-pack
		write an archive file rather than an object file
	-e
		normally the compiler quits after 10 errors; -e prints all errors
	-p path
		assume that path is the eventual import path for this code,
		and diagnose any attempt to import a package that depends on it.
	-D path
		treat a relative import as relative to path
	-L
		show entire file path when printing line numbers in errors
	-I dir1 -I dir2
		add dir1 and dir2 to the list of paths to check for imported packages
	-N
		disable optimizations
	-nolocalimports
		disallow local (relative) imports
	-S
		write assembly language text to standard output (code only)
	-S -S
		write assembly language text to standard output (code and data)
	-u
		disallow importing packages not marked as safe; implies -nolocalimports
	-V
		print the compiler version
	-race
		compile with race detection enabled

There are also a number of debugging flags; run the command with no arguments
to get a usage message.

Compiler Directives

The compiler accepts two compiler directives in the form of // comments at the
beginning of a line. To distinguish them from non-directive comments, the directives
require no space between the slashes and the name of the directive. However, since
they are comments, tools unaware of the directive convention or of a particular
directive can skip over a directive like any other comment.

    //line path/to/file:linenumber

The //line directive specifies that the source line that follows should be recorded
as having come from the given file path and line number. Successive lines are
recorded using increasing line numbers, until the next directive. This directive
typically appears in machine-generated code, so that compilers and debuggers
will show lines in the original input to the generator.

    //go:noescape

The //go:noescape directive specifies that the next declaration in the file, which
must be a func without a body (meaning that it has an implementation not written
in Go) does not allow any of the pointers passed as arguments to escape into the
heap or into the values returned from the function. This information can be used as
during the compiler's escape analysis of Go code calling the function.
*/
package main
