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
	-asan
		Insert calls to C/C++ address sanitizer.
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
		Print optimization decisions. Higher values or repetition
		produce more detail.
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
	-spectre list
		Enable spectre mitigations in list (all, index, ret).
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
The line directive is a historical special case; all other directives are of the form
//go:name, indicating that they are defined by the Go toolchain.
Each directive must be placed its own line, with only leading spaces and tabs
allowed before the comment.
Each directive applies to the Go code that immediately follows it,
which typically must be a declaration.

	//go:noescape

The //go:noescape directive must be followed by a function declaration without
a body (meaning that the function has an implementation not written in Go).
It specifies that the function does not allow any of the pointers passed as
arguments to escape into the heap or into the values returned from the function.
This information can be used during the compiler's escape analysis of Go code
calling the function.

	//go:uintptrescapes

The //go:uintptrescapes directive must be followed by a function declaration.
It specifies that the function's uintptr arguments may be pointer values that
have been converted to uintptr and must be on the heap and kept alive for the
duration of the call, even though from the types alone it would appear that the
object is no longer needed during the call. The conversion from pointer to
uintptr must appear in the argument list of any call to this function. This
directive is necessary for some low-level system call implementations and
should be avoided otherwise.

	//go:noinline

The //go:noinline directive must be followed by a function declaration.
It specifies that calls to the function should not be inlined, overriding
the compiler's usual optimization rules. This is typically only needed
for special runtime functions or when debugging the compiler.

	//go:norace

The //go:norace directive must be followed by a function declaration.
It specifies that the function's memory accesses must be ignored by the
race detector. This is most commonly used in low-level code invoked
at times when it is unsafe to call into the race detector runtime.

	//go:nosplit

The //go:nosplit directive must be followed by a function declaration.
It specifies that the function must omit its usual stack overflow check.
This is most commonly used by low-level runtime code invoked
at times when it is unsafe for the calling goroutine to be preempted.

	//go:linkname localname [importpath.name]

The //go:linkname directive conventionally precedes the var or func
declaration named by ``localname``, though its position does not
change its effect.
This directive determines the object-file symbol used for a Go var or
func declaration, allowing two Go symbols to alias the same
object-file symbol, thereby enabling one package to access a symbol in
another package even when this would violate the usual encapsulation
of unexported declarations, or even type safety.
For that reason, it is only enabled in files that have imported "unsafe".

It may be used in two scenarios. Let's assume that package upper
imports package lower, perhaps indirectly. In the first scenario,
package lower defines a symbol whose object file name belongs to
package upper. Both packages contain a linkname directive: package
lower uses the two-argument form and package upper uses the
one-argument form. In the example below, lower.f is an alias for the
function upper.g:

    package upper
    import _ "unsafe"
    //go:linkname g
    func g()

    package lower
    import _ "unsafe"
    //go:linkname f upper.g
    func f() { ... }

The linkname directive in package upper suppresses the usual error for
a function that lacks a body. (That check may alternatively be
suppressed by including a .s file, even an empty one, in the package.)

In the second scenario, package upper unilaterally creates an alias
for a symbol in package lower. In the example below, upper.g is an alias
for the function lower.f.

    package upper
    import _ "unsafe"
    //go:linkname g lower.f
    func g()

    package lower
    func f() { ... }

The declaration of lower.f may also have a linkname directive with a
single argument, f. This is optional, but helps alert the reader that
the function is accessed from outside the package.

    //go:wasmimport importmodule importname

The //go:wasmimport is a wasm-only directive that declares a dependency
on an external function provided by a wasm module. It replaces any Go
function calls to the annotated function with a `CALL` instruction to
the wasm function identified by ``importmodule`` and ``importname`` during
compilation.

    //go:wasmimport a_module f
    func g()

The types of parameters and return values to the Go function are translated to
Wasm according to the following table:

    Go types        Wasm types
    int32, uint32   i32
    int64, uint64   i64
    float32         f32
    float64         f64
    unsafe.Pointer  i32

Any other parameter types are disallowed by the compiler.

*/
package main
