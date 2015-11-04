// Copyright 2009 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

/*
Compile, typically invoked as ``go tool compile,'' compiles a single Go package
comprising the files named on the command line. It then writes a single
object file named for the basename of the first source file with a .o suffix.
The object file can then be combined with other objects into a package archive
or passed directly to the linker (``go tool link''). If invoked with -pack, the compiler
writes an archive directly, bypassing the intermediate object file.

The generated files contain type information about the symbols exported by
the package and about types used by symbols imported by the package from
other packages. It is therefore not necessary when compiling client C of
package P to read the files of P's dependencies, only the compiled output of P.

Command Line

Usage:

	go tool compile [flags] file...

The specified files must be Go source files and all part of the same package.
The same compiler is used for all target operating systems and architectures.
The GOOS and GOARCH environment variables set the desired target.

Flags:

	-D path
		Set relative path for local imports.
	-I dir1 -I dir2
		Search for imported packages in dir1, dir2, etc,
		after consulting $GOROOT/pkg/$GOOS_$GOARCH.
	-L
		Show complete file path in error messages.
	-N
		Disable optimizations.
	-S
		Print assembly listing to standard output (code only).
	-S -S
		Print assembly listing to standard output (code and data).
	-V
		Print compiler version and exit.
	-asmhdr file
		Write assembly header to file.
	-complete
		Assume package has no non-Go components.
	-cpuprofile file
		Write a CPU profile for the compilation to file.
	-dynlink
		Allow references to Go symbols in shared libraries (experimental).
	-e
		Remove the limit on the number of errors reported (default limit is 10).
	-h
		Halt with a stack trace at the first error detected.
	-importmap old=new
		Interpret import "old" as import "new" during compilation.
		The option may be repeated to add multiple mappings.
	-installsuffix suffix
		Look for packages in $GOROOT/pkg/$GOOS_$GOARCH_suffix
		instead of $GOROOT/pkg/$GOOS_$GOARCH.
	-largemodel
		Generated code that assumes a large memory model.
	-memprofile file
		Write memory profile for the compilation to file.
	-memprofilerate rate
		Set runtime.MemProfileRate for the compilation to rate.
	-msan
		Insert calls to C/C++ memory sanitizer.
	-nolocalimports
		Disallow local (relative) imports.
	-o file
		Write object to file (default file.o or, with -pack, file.a).
	-p path
		Set expected package import path for the code being compiled,
		and diagnose imports that would cause a circular dependency.
	-pack
		Write a package (archive) file rather than an object file
	-race
		Compile with race detector enabled.
	-trimpath prefix
		Remove prefix from recorded source file paths.
	-u
		Disallow importing packages not marked as safe; implies -nolocalimports.

There are also a number of debugging flags; run the command with no arguments
for a usage message.

Compiler Directives

The compiler accepts compiler directives in the form of // comments at the
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

The //line directive is an historical special case; all other directives are of the form
//go:name, indicating that the directive is defined by the Go toolchain.

	//go:noescape

The //go:noescape directive specifies that the next declaration in the file, which
must be a func without a body (meaning that it has an implementation not written
in Go) does not allow any of the pointers passed as arguments to escape into the
heap or into the values returned from the function. This information can be used
during the compiler's escape analysis of Go code calling the function.

	//go:nosplit

The //go:nosplit directive specifies that the next function declared in the file must
not include a stack overflow check. This is most commonly used by low-level
runtime sources invoked at times when it is unsafe for the calling goroutine to be
preempted.

	//go:linkname localname importpath.name

The //go:linkname directive instructs the compiler to use ``importpath.name'' as the
object file symbol name for the variable or function declared as ``localname'' in the
source code. Because this directive can subvert the type system and package
modularity, it is only enabled in files that have imported "unsafe".
*/
package main
