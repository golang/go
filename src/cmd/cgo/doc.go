// Copyright 2009 The Go Authors.  All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

/*

Cgo enables the creation of Go packages that call C code.

Usage: cgo [compiler options] file.go

The compiler options are passed through uninterpreted when
invoking gcc to compile the C parts of the package.

The input file.go is a syntactically valid Go source file that imports
the pseudo-package "C" and then refers to types such as C.size_t,
variables such as C.stdout, or functions such as C.putchar.

If the import of "C" is immediately preceded by a comment, that
comment is used as a header when compiling the C parts of
the package.  For example:

	// #include <stdio.h>
	// #include <errno.h>
	import "C"

Cgo transforms the input file into four output files: two Go source
files, a C file for 6c (or 8c or 5c), and a C file for gcc.

The standard package makefile rules in Make.pkg automate the
process of using cgo.  See $GOROOT/misc/cgo/stdio and
$GOROOT/misc/cgo/gmp for examples.

Cgo does not yet work with gccgo.
*/
package documentation
