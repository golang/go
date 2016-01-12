// Copyright 2009 The Go Authors.  All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

/*

Cgo enables the creation of Go packages that call C code.

Using cgo with the go command

To use cgo write normal Go code that imports a pseudo-package "C".
The Go code can then refer to types such as C.size_t, variables such
as C.stdout, or functions such as C.putchar.

If the import of "C" is immediately preceded by a comment, that
comment, called the preamble, is used as a header when compiling
the C parts of the package.  For example:

	// #include <stdio.h>
	// #include <errno.h>
	import "C"

The preamble may contain any C code, including function and variable
declarations and definitions.  These may then be referred to from Go
code as though they were defined in the package "C".  All names
declared in the preamble may be used, even if they start with a
lower-case letter.  Exception: static variables in the preamble may
not be referenced from Go code; static functions are permitted.

See $GOROOT/misc/cgo/stdio and $GOROOT/misc/cgo/gmp for examples.  See
"C? Go? Cgo!" for an introduction to using cgo:
https://golang.org/doc/articles/c_go_cgo.html.

CFLAGS, CPPFLAGS, CXXFLAGS and LDFLAGS may be defined with pseudo #cgo
directives within these comments to tweak the behavior of the C or C++
compiler.  Values defined in multiple directives are concatenated
together.  The directive can include a list of build constraints limiting its
effect to systems satisfying one of the constraints
(see https://golang.org/pkg/go/build/#hdr-Build_Constraints for details about the constraint syntax).
For example:

	// #cgo CFLAGS: -DPNG_DEBUG=1
	// #cgo amd64 386 CFLAGS: -DX86=1
	// #cgo LDFLAGS: -lpng
	// #include <png.h>
	import "C"

Alternatively, CPPFLAGS and LDFLAGS may be obtained via the pkg-config
tool using a '#cgo pkg-config:' directive followed by the package names.
For example:

	// #cgo pkg-config: png cairo
	// #include <png.h>
	import "C"

When building, the CGO_CFLAGS, CGO_CPPFLAGS, CGO_CXXFLAGS and
CGO_LDFLAGS environment variables are added to the flags derived from
these directives.  Package-specific flags should be set using the
directives, not the environment variables, so that builds work in
unmodified environments.

All the cgo CPPFLAGS and CFLAGS directives in a package are concatenated and
used to compile C files in that package.  All the CPPFLAGS and CXXFLAGS
directives in a package are concatenated and used to compile C++ files in that
package.  All the LDFLAGS directives in any package in the program are
concatenated and used at link time.  All the pkg-config directives are
concatenated and sent to pkg-config simultaneously to add to each appropriate
set of command-line flags.

When the cgo directives are parsed, any occurrence of the string ${SRCDIR}
will be replaced by the absolute path to the directory containing the source
file. This allows pre-compiled static libraries to be included in the package
directory and linked properly.
For example if package foo is in the directory /go/src/foo:

       // #cgo LDFLAGS: -L${SRCDIR}/libs -lfoo

Will be expanded to:

       // #cgo LDFLAGS: -L/go/src/foo/libs -lfoo

When the Go tool sees that one or more Go files use the special import
"C", it will look for other non-Go files in the directory and compile
them as part of the Go package.  Any .c, .s, or .S files will be
compiled with the C compiler.  Any .cc, .cpp, or .cxx files will be
compiled with the C++ compiler.  Any .h, .hh, .hpp, or .hxx files will
not be compiled separately, but, if these header files are changed,
the C and C++ files will be recompiled.  The default C and C++
compilers may be changed by the CC and CXX environment variables,
respectively; those environment variables may include command line
options.

The cgo tool is enabled by default for native builds on systems where
it is expected to work.  It is disabled by default when
cross-compiling.  You can control this by setting the CGO_ENABLED
environment variable when running the go tool: set it to 1 to enable
the use of cgo, and to 0 to disable it.  The go tool will set the
build constraint "cgo" if cgo is enabled.

When cross-compiling, you must specify a C cross-compiler for cgo to
use.  You can do this by setting the CC_FOR_TARGET environment
variable when building the toolchain using make.bash, or by setting
the CC environment variable any time you run the go tool.  The
CXX_FOR_TARGET and CXX environment variables work in a similar way for
C++ code.

Go references to C

Within the Go file, C's struct field names that are keywords in Go
can be accessed by prefixing them with an underscore: if x points at a C
struct with a field named "type", x._type accesses the field.
C struct fields that cannot be expressed in Go, such as bit fields
or misaligned data, are omitted in the Go struct, replaced by
appropriate padding to reach the next field or the end of the struct.

The standard C numeric types are available under the names
C.char, C.schar (signed char), C.uchar (unsigned char),
C.short, C.ushort (unsigned short), C.int, C.uint (unsigned int),
C.long, C.ulong (unsigned long), C.longlong (long long),
C.ulonglong (unsigned long long), C.float, C.double,
C.complexfloat (complex float), and C.complexdouble (complex double).
The C type void* is represented by Go's unsafe.Pointer.
The C types __int128_t and __uint128_t are represented by [16]byte.

To access a struct, union, or enum type directly, prefix it with
struct_, union_, or enum_, as in C.struct_stat.

The size of any C type T is available as C.sizeof_T, as in
C.sizeof_struct_stat.

As Go doesn't have support for C's union type in the general case,
C's union types are represented as a Go byte array with the same length.

Go structs cannot embed fields with C types.

Go code can not refer to zero-sized fields that occur at the end of
non-empty C structs.  To get the address of such a field (which is the
only operation you can do with a zero-sized field) you must take the
address of the struct and add the size of the struct.

Cgo translates C types into equivalent unexported Go types.
Because the translations are unexported, a Go package should not
expose C types in its exported API: a C type used in one Go package
is different from the same C type used in another.

Any C function (even void functions) may be called in a multiple
assignment context to retrieve both the return value (if any) and the
C errno variable as an error (use _ to skip the result value if the
function returns void).  For example:

	n, err := C.sqrt(-1)
	_, err := C.voidFunc()

Calling C function pointers is currently not supported, however you can
declare Go variables which hold C function pointers and pass them
back and forth between Go and C. C code may call function pointers
received from Go. For example:

	package main

	// typedef int (*intFunc) ();
	//
	// int
	// bridge_int_func(intFunc f)
	// {
	//		return f();
	// }
	//
	// int fortytwo()
	// {
	//	    return 42;
	// }
	import "C"
	import "fmt"

	func main() {
		f := C.intFunc(C.fortytwo)
		fmt.Println(int(C.bridge_int_func(f)))
		// Output: 42
	}

In C, a function argument written as a fixed size array
actually requires a pointer to the first element of the array.
C compilers are aware of this calling convention and adjust
the call accordingly, but Go cannot.  In Go, you must pass
the pointer to the first element explicitly: C.f(&C.x[0]).

A few special functions convert between Go and C types
by making copies of the data.  In pseudo-Go definitions:

	// Go string to C string
	// The C string is allocated in the C heap using malloc.
	// It is the caller's responsibility to arrange for it to be
	// freed, such as by calling C.free (be sure to include stdlib.h
	// if C.free is needed).
	func C.CString(string) *C.char

	// C string to Go string
	func C.GoString(*C.char) string

	// C data with explicit length to Go string
	func C.GoStringN(*C.char, C.int) string

	// C data with explicit length to Go []byte
	func C.GoBytes(unsafe.Pointer, C.int) []byte

C references to Go

Go functions can be exported for use by C code in the following way:

	//export MyFunction
	func MyFunction(arg1, arg2 int, arg3 string) int64 {...}

	//export MyFunction2
	func MyFunction2(arg1, arg2 int, arg3 string) (int64, *C.char) {...}

They will be available in the C code as:

	extern int64 MyFunction(int arg1, int arg2, GoString arg3);
	extern struct MyFunction2_return MyFunction2(int arg1, int arg2, GoString arg3);

found in the _cgo_export.h generated header, after any preambles
copied from the cgo input files. Functions with multiple
return values are mapped to functions returning a struct.
Not all Go types can be mapped to C types in a useful way.

Using //export in a file places a restriction on the preamble:
since it is copied into two different C output files, it must not
contain any definitions, only declarations. If a file contains both
definitions and declarations, then the two output files will produce
duplicate symbols and the linker will fail. To avoid this, definitions
must be placed in preambles in other files, or in C source files.

Passing pointers

Go is a garbage collected language, and the garbage collector needs to
know the location of every pointer to Go memory.  Because of this,
there are restrictions on passing pointers between Go and C.

In this section the term Go pointer means a pointer to memory
allocated by Go (such as by using the & operator or calling the
predefined new function) and the term C pointer means a pointer to
memory allocated by C (such as by a call to C.malloc).  Whether a
pointer is a Go pointer or a C pointer is a dynamic property
determined by how the memory was allocated; it has nothing to do with
the type of the pointer.

Go code may pass a Go pointer to C provided the Go memory to which it
points does not contain any Go pointers.  The C code must preserve
this property: it must not store any Go pointers in Go memory, even
temporarily.  When passing a pointer to a field in a struct, the Go
memory in question is the memory occupied by the field, not the entire
struct.  When passing a pointer to an element in an array or slice,
the Go memory in question is the entire array or the entire backing
array of the slice.

C code may not keep a copy of a Go pointer after the call returns.

A Go function called by C code may not return a Go pointer.  A Go
function called by C code may take C pointers as arguments, and it may
store non-pointer or C pointer data through those pointers, but it may
not store a Go pointer in memory pointed to by a C pointer.  A Go
function called by C code may take a Go pointer as an argument, but it
must preserve the property that the Go memory to which it points does
not contain any Go pointers.

Go code may not store a Go pointer in C memory.  C code may store Go
pointers in C memory, subject to the rule above: it must stop storing
the Go pointer when the C function returns.

These rules are checked dynamically at runtime.  The checking is
controlled by the cgocheck setting of the GODEBUG environment
variable.  The default setting is GODEBUG=cgocheck=1, which implements
reasonably cheap dynamic checks.  These checks may be disabled
entirely using GODEBUG=cgocheck=0.  Complete checking of pointer
handling, at some cost in run time, is available via GODEBUG=cgocheck=2.

It is possible to defeat this enforcement by using the unsafe package,
and of course there is nothing stopping the C code from doing anything
it likes.  However, programs that break these rules are likely to fail
in unexpected and unpredictable ways.

Using cgo directly

Usage:
	go tool cgo [cgo options] [-- compiler options] gofiles...

Cgo transforms the specified input Go source files into several output
Go and C source files.

The compiler options are passed through uninterpreted when
invoking the C compiler to compile the C parts of the package.

The following options are available when running cgo directly:

	-dynimport file
		Write list of symbols imported by file. Write to
		-dynout argument or to standard output. Used by go
		build when building a cgo package.
	-dynout file
		Write -dynimport output to file.
	-dynpackage package
		Set Go package for -dynimport output.
	-dynlinker
		Write dynamic linker as part of -dynimport output.
	-godefs
		Write out input file in Go syntax replacing C package
		names with real values. Used to generate files in the
		syscall package when bootstrapping a new target.
	-objdir directory
		Put all generated files in directory.
	-importpath string
		The import path for the Go package. Optional; used for
		nicer comments in the generated files.
	-exportheader file
		If there are any exported functions, write the
		generated export declarations to file.
		C code can #include this to see the declarations.
	-gccgo
		Generate output for the gccgo compiler rather than the
		gc compiler.
	-gccgoprefix prefix
		The -fgo-prefix option to be used with gccgo.
	-gccgopkgpath path
		The -fgo-pkgpath option to be used with gccgo.
	-import_runtime_cgo
		If set (which it is by default) import runtime/cgo in
		generated output.
	-import_syscall
		If set (which it is by default) import syscall in
		generated output.
	-debug-define
		Debugging option. Print #defines.
	-debug-gcc
		Debugging option. Trace C compiler execution and output.
*/
package main

/*
Implementation details.

Cgo provides a way for Go programs to call C code linked into the same
address space. This comment explains the operation of cgo.

Cgo reads a set of Go source files and looks for statements saying
import "C". If the import has a doc comment, that comment is
taken as literal C code to be used as a preamble to any C code
generated by cgo. A typical preamble #includes necessary definitions:

	// #include <stdio.h>
	import "C"

For more details about the usage of cgo, see the documentation
comment at the top of this file.

Understanding C

Cgo scans the Go source files that import "C" for uses of that
package, such as C.puts. It collects all such identifiers. The next
step is to determine each kind of name. In C.xxx the xxx might refer
to a type, a function, a constant, or a global variable. Cgo must
decide which.

The obvious thing for cgo to do is to process the preamble, expanding
#includes and processing the corresponding C code. That would require
a full C parser and type checker that was also aware of any extensions
known to the system compiler (for example, all the GNU C extensions) as
well as the system-specific header locations and system-specific
pre-#defined macros. This is certainly possible to do, but it is an
enormous amount of work.

Cgo takes a different approach. It determines the meaning of C
identifiers not by parsing C code but by feeding carefully constructed
programs into the system C compiler and interpreting the generated
error messages, debug information, and object files. In practice,
parsing these is significantly less work and more robust than parsing
C source.

Cgo first invokes gcc -E -dM on the preamble, in order to find out
about simple #defines for constants and the like. These are recorded
for later use.

Next, cgo needs to identify the kinds for each identifier. For the
identifiers C.foo and C.bar, cgo generates this C program:

	<preamble>
	#line 1 "not-declared"
	void __cgo_f_xxx_1(void) { __typeof__(foo) *__cgo_undefined__; }
	#line 1 "not-type"
	void __cgo_f_xxx_2(void) { foo *__cgo_undefined__; }
	#line 1 "not-const"
	void __cgo_f_xxx_3(void) { enum { __cgo_undefined__ = (foo)*1 }; }
	#line 2 "not-declared"
	void __cgo_f_xxx_1(void) { __typeof__(bar) *__cgo_undefined__; }
	#line 2 "not-type"
	void __cgo_f_xxx_2(void) { bar *__cgo_undefined__; }
	#line 2 "not-const"
	void __cgo_f_xxx_3(void) { enum { __cgo_undefined__ = (bar)*1 }; }

This program will not compile, but cgo can use the presence or absence
of an error message on a given line to deduce the information it
needs. The program is syntactically valid regardless of whether each
name is a type or an ordinary identifier, so there will be no syntax
errors that might stop parsing early.

An error on not-declared:1 indicates that foo is undeclared.
An error on not-type:1 indicates that foo is not a type (if declared at all, it is an identifier).
An error on not-const:1 indicates that foo is not an integer constant.

The line number specifies the name involved. In the example, 1 is foo and 2 is bar.

Next, cgo must learn the details of each type, variable, function, or
constant. It can do this by reading object files. If cgo has decided
that t1 is a type, v2 and v3 are variables or functions, and c4, c5,
and c6 are constants, it generates:

	<preamble>
	__typeof__(t1) *__cgo__1;
	__typeof__(v2) *__cgo__2;
	__typeof__(v3) *__cgo__3;
	__typeof__(c4) *__cgo__4;
	enum { __cgo_enum__4 = c4 };
	__typeof__(c5) *__cgo__5;
	enum { __cgo_enum__5 = c5 };
	__typeof__(c6) *__cgo__6;
	enum { __cgo_enum__6 = c6 };

	long long __cgo_debug_data[] = {
		0, // t1
		0, // v2
		0, // v3
		c4,
		c5,
		c6,
		1
	};

and again invokes the system C compiler, to produce an object file
containing debug information. Cgo parses the DWARF debug information
for __cgo__N to learn the type of each identifier. (The types also
distinguish functions from global variables.) If using a standard gcc,
cgo can parse the DWARF debug information for the __cgo_enum__N to
learn the identifier's value. The LLVM-based gcc on OS X emits
incomplete DWARF information for enums; in that case cgo reads the
constant values from the __cgo_debug_data from the object file's data
segment.

At this point cgo knows the meaning of each C.xxx well enough to start
the translation process.

Translating Go

Given the input Go files x.go and y.go, cgo generates these source
files:

	x.cgo1.go       # for gc (cmd/compile)
	y.cgo1.go       # for gc
	_cgo_gotypes.go # for gc
	_cgo_import.go  # for gc (if -dynout _cgo_import.go)
	x.cgo2.c        # for gcc
	y.cgo2.c        # for gcc
	_cgo_defun.c    # for gcc (if -gccgo)
	_cgo_export.c   # for gcc
	_cgo_export.h   # for gcc
	_cgo_main.c     # for gcc
	_cgo_flags      # for alternative build tools

The file x.cgo1.go is a copy of x.go with the import "C" removed and
references to C.xxx replaced with names like _Cfunc_xxx or _Ctype_xxx.
The definitions of those identifiers, written as Go functions, types,
or variables, are provided in _cgo_gotypes.go.

Here is a _cgo_gotypes.go containing definitions for needed C types:

	type _Ctype_char int8
	type _Ctype_int int32
	type _Ctype_void [0]byte

The _cgo_gotypes.go file also contains the definitions of the
functions.  They all have similar bodies that invoke runtime·cgocall
to make a switch from the Go runtime world to the system C (GCC-based)
world.

For example, here is the definition of _Cfunc_puts:

	//go:cgo_import_static _cgo_be59f0f25121_Cfunc_puts
	//go:linkname __cgofn__cgo_be59f0f25121_Cfunc_puts _cgo_be59f0f25121_Cfunc_puts
	var __cgofn__cgo_be59f0f25121_Cfunc_puts byte
	var _cgo_be59f0f25121_Cfunc_puts = unsafe.Pointer(&__cgofn__cgo_be59f0f25121_Cfunc_puts)

	func _Cfunc_puts(p0 *_Ctype_char) (r1 _Ctype_int) {
		_cgo_runtime_cgocall(_cgo_be59f0f25121_Cfunc_puts, uintptr(unsafe.Pointer(&p0)))
		return
	}

The hexadecimal number is a hash of cgo's input, chosen to be
deterministic yet unlikely to collide with other uses. The actual
function _cgo_be59f0f25121_Cfunc_puts is implemented in a C source
file compiled by gcc, the file x.cgo2.c:

	void
	_cgo_be59f0f25121_Cfunc_puts(void *v)
	{
		_cgo_wait_runtime_init_done();
		struct {
			char* p0;
			int r;
			char __pad12[4];
		} __attribute__((__packed__, __gcc_struct__)) *a = v;
		a->r = puts((void*)a->p0);
	}

It waits for Go runtime to be initialized (required for shared libraries),
extracts the arguments from the pointer to _Cfunc_puts's argument
frame, invokes the system C function (in this case, puts), stores the
result in the frame, and returns.

Linking

Once the _cgo_export.c and *.cgo2.c files have been compiled with gcc,
they need to be linked into the final binary, along with the libraries
they might depend on (in the case of puts, stdio). cmd/link has been
extended to understand basic ELF files, but it does not understand ELF
in the full complexity that modern C libraries embrace, so it cannot
in general generate direct references to the system libraries.

Instead, the build process generates an object file using dynamic
linkage to the desired libraries. The main function is provided by
_cgo_main.c:

	int main() { return 0; }
	void crosscall2(void(*fn)(void*, int), void *a, int c) { }
	void _cgo_wait_runtime_init_done() { }
	void _cgo_allocate(void *a, int c) { }
	void _cgo_panic(void *a, int c) { }

The extra functions here are stubs to satisfy the references in the C
code generated for gcc. The build process links this stub, along with
_cgo_export.c and *.cgo2.c, into a dynamic executable and then lets
cgo examine the executable. Cgo records the list of shared library
references and resolved names and writes them into a new file
_cgo_import.go, which looks like:

	//go:cgo_dynamic_linker "/lib64/ld-linux-x86-64.so.2"
	//go:cgo_import_dynamic puts puts#GLIBC_2.2.5 "libc.so.6"
	//go:cgo_import_dynamic __libc_start_main __libc_start_main#GLIBC_2.2.5 "libc.so.6"
	//go:cgo_import_dynamic stdout stdout#GLIBC_2.2.5 "libc.so.6"
	//go:cgo_import_dynamic fflush fflush#GLIBC_2.2.5 "libc.so.6"
	//go:cgo_import_dynamic _ _ "libpthread.so.0"
	//go:cgo_import_dynamic _ _ "libc.so.6"

In the end, the compiled Go package, which will eventually be
presented to cmd/link as part of a larger program, contains:

	_go_.o        # gc-compiled object for _cgo_gotypes.go, _cgo_import.go, *.cgo1.go
	_all.o        # gcc-compiled object for _cgo_export.c, *.cgo2.c

The final program will be a dynamic executable, so that cmd/link can avoid
needing to process arbitrary .o files. It only needs to process the .o
files generated from C files that cgo writes, and those are much more
limited in the ELF or other features that they use.

In essence, the _cgo_import.o file includes the extra linking
directives that cmd/link is not sophisticated enough to derive from _all.o
on its own. Similarly, the _all.o uses dynamic references to real
system object code because cmd/link is not sophisticated enough to process
the real code.

The main benefits of this system are that cmd/link remains relatively simple
(it does not need to implement a complete ELF and Mach-O linker) and
that gcc is not needed after the package is compiled. For example,
package net uses cgo for access to name resolution functions provided
by libc. Although gcc is needed to compile package net, gcc is not
needed to link programs that import package net.

Runtime

When using cgo, Go must not assume that it owns all details of the
process. In particular it needs to coordinate with C in the use of
threads and thread-local storage. The runtime package declares a few
variables:

	var (
		iscgo             bool
		_cgo_init         unsafe.Pointer
		_cgo_thread_start unsafe.Pointer
	)

Any package using cgo imports "runtime/cgo", which provides
initializations for these variables. It sets iscgo to true, _cgo_init
to a gcc-compiled function that can be called early during program
startup, and _cgo_thread_start to a gcc-compiled function that can be
used to create a new thread, in place of the runtime's usual direct
system calls.

Internal and External Linking

The text above describes "internal" linking, in which cmd/link parses and
links host object files (ELF, Mach-O, PE, and so on) into the final
executable itself. Keeping cmd/link simple means we cannot possibly
implement the full semantics of the host linker, so the kinds of
objects that can be linked directly into the binary is limited (other
code can only be used as a dynamic library). On the other hand, when
using internal linking, cmd/link can generate Go binaries by itself.

In order to allow linking arbitrary object files without requiring
dynamic libraries, cgo supports an "external" linking mode too. In
external linking mode, cmd/link does not process any host object files.
Instead, it collects all the Go code and writes a single go.o object
file containing it. Then it invokes the host linker (usually gcc) to
combine the go.o object file and any supporting non-Go code into a
final executable. External linking avoids the dynamic library
requirement but introduces a requirement that the host linker be
present to create such a binary.

Most builds both compile source code and invoke the linker to create a
binary. When cgo is involved, the compile step already requires gcc, so
it is not problematic for the link step to require gcc too.

An important exception is builds using a pre-compiled copy of the
standard library. In particular, package net uses cgo on most systems,
and we want to preserve the ability to compile pure Go code that
imports net without requiring gcc to be present at link time. (In this
case, the dynamic library requirement is less significant, because the
only library involved is libc.so, which can usually be assumed
present.)

This conflict between functionality and the gcc requirement means we
must support both internal and external linking, depending on the
circumstances: if net is the only cgo-using package, then internal
linking is probably fine, but if other packages are involved, so that there
are dependencies on libraries beyond libc, external linking is likely
to work better. The compilation of a package records the relevant
information to support both linking modes, leaving the decision
to be made when linking the final binary.

Linking Directives

In either linking mode, package-specific directives must be passed
through to cmd/link. These are communicated by writing //go: directives in a
Go source file compiled by gc. The directives are copied into the .o
object file and then processed by the linker.

The directives are:

//go:cgo_import_dynamic <local> [<remote> ["<library>"]]

	In internal linking mode, allow an unresolved reference to
	<local>, assuming it will be resolved by a dynamic library
	symbol. The optional <remote> specifies the symbol's name and
	possibly version in the dynamic library, and the optional "<library>"
	names the specific library where the symbol should be found.

	In the <remote>, # or @ can be used to introduce a symbol version.

	Examples:
	//go:cgo_import_dynamic puts
	//go:cgo_import_dynamic puts puts#GLIBC_2.2.5
	//go:cgo_import_dynamic puts puts#GLIBC_2.2.5 "libc.so.6"

	A side effect of the cgo_import_dynamic directive with a
	library is to make the final binary depend on that dynamic
	library. To get the dependency without importing any specific
	symbols, use _ for local and remote.

	Example:
	//go:cgo_import_dynamic _ _ "libc.so.6"

	For compatibility with current versions of SWIG,
	#pragma dynimport is an alias for //go:cgo_import_dynamic.

//go:cgo_dynamic_linker "<path>"

	In internal linking mode, use "<path>" as the dynamic linker
	in the final binary. This directive is only needed from one
	package when constructing a binary; by convention it is
	supplied by runtime/cgo.

	Example:
	//go:cgo_dynamic_linker "/lib/ld-linux.so.2"

//go:cgo_export_dynamic <local> <remote>

	In internal linking mode, put the Go symbol
	named <local> into the program's exported symbol table as
	<remote>, so that C code can refer to it by that name. This
	mechanism makes it possible for C code to call back into Go or
	to share Go's data.

	For compatibility with current versions of SWIG,
	#pragma dynexport is an alias for //go:cgo_export_dynamic.

//go:cgo_import_static <local>

	In external linking mode, allow unresolved references to
	<local> in the go.o object file prepared for the host linker,
	under the assumption that <local> will be supplied by the
	other object files that will be linked with go.o.

	Example:
	//go:cgo_import_static puts_wrapper

//go:cgo_export_static <local> <remote>

	In external linking mode, put the Go symbol
	named <local> into the program's exported symbol table as
	<remote>, so that C code can refer to it by that name. This
	mechanism makes it possible for C code to call back into Go or
	to share Go's data.

//go:cgo_ldflag "<arg>"

	In external linking mode, invoke the host linker (usually gcc)
	with "<arg>" as a command-line argument following the .o files.
	Note that the arguments are for "gcc", not "ld".

	Example:
	//go:cgo_ldflag "-lpthread"
	//go:cgo_ldflag "-L/usr/local/sqlite3/lib"

A package compiled with cgo will include directives for both
internal and external linking; the linker will select the appropriate
subset for the chosen linking mode.

Example

As a simple example, consider a package that uses cgo to call C.sin.
The following code will be generated by cgo:

	// compiled by gc

	//go:cgo_ldflag "-lm"

	type _Ctype_double float64

	//go:cgo_import_static _cgo_gcc_Cfunc_sin
	//go:linkname __cgo_gcc_Cfunc_sin _cgo_gcc_Cfunc_sin
	var __cgo_gcc_Cfunc_sin byte
	var _cgo_gcc_Cfunc_sin = unsafe.Pointer(&__cgo_gcc_Cfunc_sin)

	func _Cfunc_sin(p0 _Ctype_double) (r1 _Ctype_double) {
		_cgo_runtime_cgocall(_cgo_gcc_Cfunc_sin, uintptr(unsafe.Pointer(&p0)))
		return
	}

	// compiled by gcc, into foo.cgo2.o

	void
	_cgo_gcc_Cfunc_sin(void *v)
	{
		struct {
			double p0;
			double r;
		} __attribute__((__packed__)) *a = v;
		a->r = sin(a->p0);
	}

What happens at link time depends on whether the final binary is linked
using the internal or external mode. If other packages are compiled in
"external only" mode, then the final link will be an external one.
Otherwise the link will be an internal one.

The linking directives are used according to the kind of final link
used.

In internal mode, cmd/link itself processes all the host object files, in
particular foo.cgo2.o. To do so, it uses the cgo_import_dynamic and
cgo_dynamic_linker directives to learn that the otherwise undefined
reference to sin in foo.cgo2.o should be rewritten to refer to the
symbol sin with version GLIBC_2.2.5 from the dynamic library
"libm.so.6", and the binary should request "/lib/ld-linux.so.2" as its
runtime dynamic linker.

In external mode, cmd/link does not process any host object files, in
particular foo.cgo2.o. It links together the gc-generated object
files, along with any other Go code, into a go.o file. While doing
that, cmd/link will discover that there is no definition for
_cgo_gcc_Cfunc_sin, referred to by the gc-compiled source file. This
is okay, because cmd/link also processes the cgo_import_static directive and
knows that _cgo_gcc_Cfunc_sin is expected to be supplied by a host
object file, so cmd/link does not treat the missing symbol as an error when
creating go.o. Indeed, the definition for _cgo_gcc_Cfunc_sin will be
provided to the host linker by foo2.cgo.o, which in turn will need the
symbol 'sin'. cmd/link also processes the cgo_ldflag directives, so that it
knows that the eventual host link command must include the -lm
argument, so that the host linker will be able to find 'sin' in the
math library.

cmd/link Command Line Interface

The go command and any other Go-aware build systems invoke cmd/link
to link a collection of packages into a single binary. By default, cmd/link will
present the same interface it does today:

	cmd/link main.a

produces a file named a.out, even if cmd/link does so by invoking the host
linker in external linking mode.

By default, cmd/link will decide the linking mode as follows: if the only
packages using cgo are those on a whitelist of standard library
packages (net, os/user, runtime/cgo), cmd/link will use internal linking
mode. Otherwise, there are non-standard cgo packages involved, and cmd/link
will use external linking mode. The first rule means that a build of
the godoc binary, which uses net but no other cgo, can run without
needing gcc available. The second rule means that a build of a
cgo-wrapped library like sqlite3 can generate a standalone executable
instead of needing to refer to a dynamic library. The specific choice
can be overridden using a command line flag: cmd/link -linkmode=internal or
cmd/link -linkmode=external.

In an external link, cmd/link will create a temporary directory, write any
host object files found in package archives to that directory (renamed
to avoid conflicts), write the go.o file to that directory, and invoke
the host linker. The default value for the host linker is $CC, split
into fields, or else "gcc". The specific host linker command line can
be overridden using command line flags: cmd/link -extld=clang
-extldflags='-ggdb -O3'.  If any package in a build includes a .cc or
other file compiled by the C++ compiler, the go tool will use the
-extld option to set the host linker to the C++ compiler.

These defaults mean that Go-aware build systems can ignore the linking
changes and keep running plain 'cmd/link' and get reasonable results, but
they can also control the linking details if desired.

*/
