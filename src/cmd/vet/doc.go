// Copyright 2010 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

/*

Vet examines Go source code and reports suspicious constructs, such as Printf
calls whose arguments do not align with the format string. Vet uses heuristics
that do not guarantee all reports are genuine problems, but it can find errors
not caught by the compilers.

Available checks:

1. Printf family

Suspicious calls to functions in the Printf familiy, including any functions
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

Non-standard signatures for methods with familiar names, including:
	Format GobEncode GobDecode MarshalJSON MarshalXML
	Peek ReadByte ReadFrom ReadRune Scan Seek 
	UnmarshalJSON UnreadByte UnreadRune WriteByte
	WriteTo

3. Struct tags

Struct tags that do not follow the format understood by reflect.StructTag.Get.

Usage:

	go tool vet [flag] [file.go ...]
	go tool vet [flag] [directory ...] # Scan all .go files under directory, recursively

The flags are:
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

*/
package documentation
