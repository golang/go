// Copyright 2010 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

/*

Vet does simple checking of Go source code.

It checks for simple errors in calls to functions named
	Print Printf Println
	Fprint Fprintf Fprintln
	Sprint Sprintf Sprintln
	Error Errorf
	Fatal Fatalf
If the function name ends with an 'f', the function is assumed to take
a format descriptor string in the manner of fmt.Printf. If not, vet
complains about arguments that look like format descriptor strings.

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
