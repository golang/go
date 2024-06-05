// Copyright 2023 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

/*
Package raw provides an interface to interpret and emit Go execution traces.
It can interpret and emit execution traces in its wire format as well as a
bespoke but simple text format.

The readers and writers in this package perform no validation on or ordering of
the input, and so are generally unsuitable for analysis. However, they're very
useful for testing and debugging the tracer in the runtime and more sophisticated
trace parsers.

# Text format specification

The trace text format produced and consumed by this package is a line-oriented
format.

The first line in each text trace is the header line.

	Trace Go1.XX

Following that is a series of event lines. Each event begins with an
event name, followed by zero or more named unsigned integer arguments.
Names are separated from their integer values by an '=' sign. Names can
consist of any UTF-8 character except '='.

For example:

	EventName arg1=23 arg2=55 arg3=53

Any amount of whitespace is allowed to separate each token. Whitespace
is identified via unicode.IsSpace.

Some events have additional data on following lines. There are two such
special cases.

The first special case consists of events with trailing byte-oriented data.
The trailer begins on the following line from the event. That line consists
of a single argument 'data' and a Go-quoted string representing the byte data
within. Note: an explicit argument for the length is elided, because it's
just the length of the unquoted string.

For example:

	String id=5
		data="hello world\x00"

These events are identified in their spec by the HasData flag.

The second special case consists of stack events. These events are identified
by the IsStack flag. These events also have a trailing unsigned integer argument
describing the number of stack frame descriptors that follow. Each stack frame
descriptor is on its own line following the event, consisting of four signed
integer arguments: the PC, an integer describing the function name, an integer
describing the file name, and the line number in that file that function was at
at the time the stack trace was taken.

For example:

	Stack id=5 n=2
		pc=1241251 func=3 file=6 line=124
		pc=7534345 func=6 file=3 line=64
*/
package raw
