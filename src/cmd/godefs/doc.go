// Copyright 2009 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

/*

Godefs is a bootstrapping tool for porting the Go runtime to new systems.
It translates C type declarations into C or Go type declarations
with the same memory layout. 

Usage: godefs [-g package] [-c cc] [-f cc-arg]... [defs.c ...]

Godefs takes as input a host-compilable C file that includes
standard system headers.  From that input file, it generates
a standalone (no #includes) C or Go file containing equivalent
definitions.

The input to godefs is a C input file that can be compiled by
the host system's standard C compiler (typically gcc).
This file is expected to define new types and enumerated constants
whose names begin with $ (a legal identifier character in gcc). 
Godefs compile the given input file with the host compiler and
then parses the debug info embedded in the assembly output.
This is far easier than reading system headers on most machines.

The output from godefs is either C output intended for the
Plan 9 C compiler tool chain (6c, 8c, or 5c) or Go output.	

The options are:

	-g package
		generate Go output using the given package name.
		In the Go output, struct fields have leading xx_ prefixes
		removed and the first character capitalized (exported).

	-c cc
		set the name of the host system's C compiler (default "gcc")
	
	-f cc-arg
		add cc-arg to the command line when invoking the system C compiler
		(for example, -f -m64 to invoke gcc -m64).
		Repeating this option adds multiple flags to the command line.

For example, if this is x.c:

	#include <sys/stat.h>

	typedef struct timespec $Timespec;
	enum {
		$S_IFMT = S_IFMT,
		$S_IFIFO = S_IFIFO,
		$S_IFCHR = S_IFCHR,
	};

then "godefs x.c" generates:

	// godefs x.c
	// MACHINE GENERATED - DO NOT EDIT.
	
	// Constants
	enum {
		S_IFMT = 0xf000,
		S_IFIFO = 0x1000,
		S_IFCHR = 0x2000,
	};
	
	// Types
	#pragma pack on
	
	typedef struct Timespec Timespec;
	struct Timespec {
		int64 tv_sec;
		int64 tv_nsec;
	};
	#pragma pack off

and "godefs -g MyPackage x.c" generates:

	// godefs -g MyPackage x.c
	// MACHINE GENERATED - DO NOT EDIT.
	
	package MyPackage
	
	// Constants
	const (
		S_IFMT = 0xf000;
		S_IFIFO = 0x1000;
		S_IFCHR = 0x2000;
	)
	
	// Types
	
	type Timespec struct {
		Sec int64;
		Nsec int64;
	}

*/
package documentation
