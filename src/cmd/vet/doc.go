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

By default the -all flag is set so all checks are performed.
If any flags are explicitly set to true, only those tests are run. Conversely, if
any flag is explicitly set to false, only those tests are disabled.  Thus -printf=true
runs the printf check, -printf=false runs all checks except the printf check.

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

Invalid uses of cgo

Flag: -cgocall

Detect some violations of the cgo pointer passing rules.

Unkeyed composite literals

Flag: -composites

Composite struct literals that do not use the field-keyed syntax.

Copying locks

Flag: -copylocks

Locks that are erroneously passed by value.

HTTP responses used incorrectly

Flag: -httpresponse

Mistakes deferring a function call on an HTTP response before
checking whether the error returned with the response was nil.

Failure to call the cancelation function returned by WithCancel

Flag: -lostcancel

The cancelation function returned by context.WithCancel, WithTimeout,
and WithDeadline must be called or the new context will remain live
until its parent context is cancelled.
(The background context is never cancelled.)

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
The -printfuncs flag can be used to redefine this list.
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

Tests and documentation examples

Flag: -tests

Mistakes involving tests including functions with incorrect names or signatures
and example tests that document identifiers not in the package.

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
		Enable all non-experimental checks.
	-v
		Verbose mode
	-printfuncs
		A comma-separated list of print-like function names
		to supplement the standard list.
		For more information, see the discussion of the -printf flag.
	-shadowstrict
		Whether to be strict about shadowing; can be noisy.
*/
package main
