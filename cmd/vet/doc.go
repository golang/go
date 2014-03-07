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
recursively descends the directory, vetting each file in isolation.
Package-level type-checking is disabled, so the vetting is weaker.

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

1. Printf family

Flag -printf

Suspicious calls to functions in the Printf family, including any functions
with these names:
	Print Printf Println
	Fprint Fprintf Fprintln
	Sprint Sprintf Sprintln
	Error Errorf
	Fatal Fatalf
	Panic Panicf Panicln
If the function name ends with an 'f', the function is assumed to take
a format descriptor string in the manner of fmt.Printf. If not, vet
complains about arguments that look like format descriptor strings.

It also checks for errors such as using a Writer as the first argument of
Printf.

2. Methods

Flag -methods

Non-standard signatures for methods with familiar names, including:
	Format GobEncode GobDecode MarshalJSON MarshalXML
	Peek ReadByte ReadFrom ReadRune Scan Seek
	UnmarshalJSON UnreadByte UnreadRune WriteByte
	WriteTo

3. Struct tags

Flag -structtags

Struct tags that do not follow the format understood by reflect.StructTag.Get.

4. Unkeyed composite literals

Flag -composites

Composite struct literals that do not use the field-keyed syntax.

5. Assembly declarations

Flag -asmdecl

Mismatches between assembly files and Go function declarations.

6. Useless assignments

Flag -assign

Check for useless assignments.

7. Atomic mistakes

Flag -atomic

Common mistaken usages of the sync/atomic package.

8. Build tags

Flag -buildtags

Badly formed or misplaced +build tags.

9. Copying locks

Flag -copylocks

Locks that are erroneously passed by value.

10. Nil function comparison

Flag -nilfunc

Comparisons between functions and nil.

11. Range loop variables

Flag -rangeloops

Incorrect uses of range loop variables in closures.

12. Unreachable code

Flag -unreachable

Unreachable code.

13. Shadowed variables

Flag -shadow=false (experimental; must be set explicitly)

Variables that may have been unintentionally shadowed.


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
package main
