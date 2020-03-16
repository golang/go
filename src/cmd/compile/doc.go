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
	-buildid id
		Record id as the build id in the export metadata.
	-blockprofile file
		Write block profile for the compilation to file.
	-c int
		Concurrency during compilation. Set 1 for no concurrency (default is 1).
	-complete
		Assume package has no non-Go components.
	-cpuprofile file
		Write a CPU profile for the compilation to file.
	-dynlink
		Allow references to Go symbols in shared libraries (experimental).
	-e
		Remove the limit on the number of errors reported (default limit is 10).
	-goversion string
		Specify required go tool version of the runtime.
		Exits when the runtime go version does not match goversion.
	-h
		Halt with a stack trace at the first error detected.
	-importcfg file
		Read import configuration from file.
		In the file, set importmap, packagefile to specify import resolution.
	-importmap old=new
		Interpret import "old" as import "new" during compilation.
		The option may be repeated to add multiple mappings.
	-installsuffix suffix
		Look for packages in $GOROOT/pkg/$GOOS_$GOARCH_suffix
		instead of $GOROOT/pkg/$GOOS_$GOARCH.
	-l
		Disable inlining.
	-lang version
		Set language version to compile, as in -lang=go1.12.
		Default is current version.
	-linkobj file
		Write linker-specific object to file and compiler-specific
		object to usual output file (as specified by -o).
		Without this flag, the -o output is a combination of both
		linker and compiler input.
	-m
		Print optimization decisions.
	-memprofile file
		Write memory profile for the compilation to file.
	-memprofilerate rate
		Set runtime.MemProfileRate for the compilation to rate.
	-msan
		Insert calls to C/C++ memory sanitizer.
	-mutexprofile file
		Write mutex profile for the compilation to file.
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
	-s
		Warn about composite literals that can be simplified.
	-shared
		Generate code that can be linked into a shared library.
	-traceprofile file
		Write an execution trace to file.
	-trimpath prefix
		Remove prefix from recorded source file paths.

Flags related to debugging information:

	-dwarf
		Generate DWARF symbols.
	-dwarflocationlists
		Add location lists to DWARF in optimized mode.
	-gendwarfinl int
		Generate DWARF inline info records (default 2).

Flags to debug the compiler itself:

	-E
		Debug symbol export.
	-K
		Debug missing line numbers.
	-d list
		Print debug information about items in list. Try -d help for further information.
	-live
		Debug liveness analysis.
	-v
		Increase debug verbosity.
	-%
		Debug non-static initializers.
	-W
		Debug parse tree after type checking.
	-f
		Debug stack frames.
	-i
		Debug line number stack.
	-j
		Debug runtime-initialized variables.
	-r
		Debug generated wrappers.
	-w
		Debug type checking.

Compiler Directives

The compiler accepts directives in the form of comments.
To distinguish them from non-directive comments, directives
require no space between the comment opening and the name of the directive. However, since
they are comments, tools unaware of the directive convention or of a particular
directive can skip over a directive like any other comment.
*/
// Line directives come in several forms:
//
// 	//line :line
// 	//line :line:col
// 	//line filename:line
// 	//line filename:line:col
// 	/*line :line*/
// 	/*line :line:col*/
// 	/*line filename:line*/
// 	/*line filename:line:col*/
//
// In order to be recognized as a line directive, the comment must start with
// //line or /*line followed by a space, and must contain at least one colon.
// The //line form must start at the beginning of a line.
// A line directive specifies the source position for the character immediately following
// the comment as having come from the specified file, line and column:
// For a //line comment, this is the first character of the next line, and
// for a /*line comment this is the character position immediately following the closing */.
// If no filename is given, the recorded filename is empty if there is also no column number;
// otherwise it is the most recently recorded filename (actual filename or filename specified
// by previous line directive).
// If a line directive doesn't specify a column number, the column is "unknown" until
// the next directive and the compiler does not report column numbers for that range.
// The line directive text is interpreted from the back: First the trailing :ddd is peeled
// off from the directive text if ddd is a valid number > 0. Then the second :ddd
// is peeled off the same way if it is valid. Anything before that is considered the filename
// (possibly including blanks and colons). Invalid line or column values are reported as errors.
//
// Examples:
//
//	//line foo.go:10      the filename is foo.go, and the line number is 10 for the next line
//	//line C:foo.go:10    colons are permitted in filenames, here the filename is C:foo.go, and the line is 10
//	//line  a:100 :10     blanks are permitted in filenames, here the filename is " a:100 " (excluding quotes)
//	/*line :10:20*/x      the position of x is in the current file with line number 10 and column number 20
//	/*line foo: 10 */     this comment is recognized as invalid line directive (extra blanks around line number)
//
// Line directives typically appear in machine-generated code, so that compilers and debuggers
// will report positions in the original input to the generator.
/*
The line directive is an historical special case; all other directives are of the form
//go:name and must start at the beginning of a line, indicating that the directive is defined
by the Go toolchain.

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

	//go:linkname localname [importpath.name]

The //go:linkname directive instructs the compiler to use ``importpath.name'' as the
object file symbol name for the variable or function declared as ``localname'' in the
source code.
If the ``importpath.name'' argument is omitted, the directive uses the
symbol's default object file symbol name and only has the effect of making
the symbol accessible to other packages.
Because this directive can subvert the type system and package
modularity, it is only enabled in files that have imported "unsafe".
*/
package main
