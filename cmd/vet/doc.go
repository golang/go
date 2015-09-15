// Copyright 2010 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

/*

Vet examines Go source code and reports suspicious constructs, such as Printf
calls whose arguments do not align with the format string. Vet uses heuristics
that do not guarantee all reports are genuine problems, but it can find errors
not caught by the compilers.

It can be invoked three ways:

By package, from the go tool:
	go vet package/path/name
vets the package whose path is provided.

By files:
	go tool vet source/directory/*.go
vets the files named, all of which must be in the same package.

By directory:
	go tool vet source/directory
recursively descends the directory, vetting each package it finds.

Vet's exit code is 2 for erroneous invocation of the tool, 1 if a
problem was reported, and 0 otherwise. Note that the tool does not
check every possible problem and depends on unreliable heuristics
so it should be used as guidance only, not as a firm indicator of
program correctness.

By default all checks are performed. If any flags are explicitly set
to true, only those tests are run. Conversely, if any flag is
explicitly set to false, only those tests are disabled.
Thus -printf=true runs the printf check, -printf=false runs all checks
except the printf check.

Available checks:

Assembly declarations

Flag: -asmdecl

Mismatches between assembly files and Go function declarations.

Useless assignments

Flag: -assign

Check for useless assignments.

Atomic mistakes

Flag: -atomic

Common mistaken usages of the sync/atomic package.

Boolean conditions

Flag: -bool

Mistakes involving boolean operators.

Build tags

Flag: -buildtags

Badly formed or misplaced +build tags.

Unkeyed composite literals

Flag: -composites

Composite struct literals that do not use the field-keyed syntax.

Copying locks

Flag: -copylocks

Locks that are erroneously passed by value.

Documentation examples

Flag: -example

Mistakes involving example tests, including examples with incorrect names or
function signatures, or that document identifiers not in the package.

Methods

Flag: -methods

Non-standard signatures for methods with familiar names, including:
	Format GobEncode GobDecode MarshalJSON MarshalXML
	Peek ReadByte ReadFrom ReadRune Scan Seek
	UnmarshalJSON UnreadByte UnreadRune WriteByte
	WriteTo

Nil function comparison

Flag: -nilfunc

Comparisons between functions and nil.

Printf family

Flag: -printf

Suspicious calls to functions in the Printf family, including any functions
with these names, disregarding case:
	Print Printf Println
	Fprint Fprintf Fprintln
	Sprint Sprintf Sprintln
	Error Errorf
	Fatal Fatalf
	Log Logf
	Panic Panicf Panicln
If the function name ends with an 'f', the function is assumed to take
a format descriptor string in the manner of fmt.Printf. If not, vet
complains about arguments that look like format descriptor strings.

It also checks for errors such as using a Writer as the first argument of
Printf.

Range loop variables

Flag: -rangeloops

Incorrect uses of range loop variables in closures.

Shadowed variables

Flag: -shadow=false (experimental; must be set explicitly)

Variables that may have been unintentionally shadowed.

Shifts

Flag: -shift

Shifts equal to or longer than the variable's length.

Struct tags

Flag: -structtags

Struct tags that do not follow the format understood by reflect.StructTag.Get.
Well-known encoding struct tags (json, xml) used with unexported fields.

Unreachable code

Flag: -unreachable

Unreachable code.

Misuse of unsafe Pointers

Flag: -unsafeptr

Likely incorrect uses of unsafe.Pointer to convert integers to pointers.
A conversion from uintptr to unsafe.Pointer is invalid if it implies that
there is a uintptr-typed word in memory that holds a pointer value,
because that word will be invisible to stack copying and to the garbage
collector.

Unused result of certain function calls

Flag: -unusedresult

Calls to well-known functions and methods that return a value that is
discarded.  By default, this includes functions like fmt.Errorf and
fmt.Sprintf and methods like String and Error. The flags -unusedfuncs
and -unusedstringmethods control the set.

Other flags

These flags configure the behavior of vet:

	-all (default true)
		Check everything; disabled if any explicit check is requested.
	-v
		Verbose mode
	-printfuncs
		A comma-separated list of print-like functions to supplement
		the standard list.  Each entry is in the form Name:N where N
		is the zero-based argument position of the first argument
		involved in the print: either the format or the first print
		argument for non-formatted prints.  For example,
		if you have Warn and Warnf functions that take an
		io.Writer as their first argument, like Fprintf,
			-printfuncs=Warn:1,Warnf:1
	-shadowstrict
		Whether to be strict about shadowing; can be noisy.
	-test
		For testing only: sets -all and -shadow.
*/
package main // import "golang.org/x/tools/cmd/vet"
