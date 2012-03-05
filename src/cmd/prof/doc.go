// Copyright 2009 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

/*

Prof is a rudimentary real-time profiler.

Given a command to run or the process id (pid) of a command already
running, it samples the program's state at regular intervals and reports
on its behavior.  With no options, it prints a histogram of the locations
in the code that were sampled during execution.

Since it is a real-time profiler, unlike a traditional profiler it samples
the program's state even when it is not running, such as when it is
asleep or waiting for I/O.  Each thread contributes equally to the
statistics.

Usage:
	go tool prof -p pid [-t total_secs] [-d delta_msec] [6.out args ...]

The output modes (default -h) are:

	-P file.prof:
		Write the profile information to file.prof, in the format used by pprof.
		At the moment, this only works on Linux amd64 binaries and requires that the
		binary be written using 6l -e to produce ELF debug info.
		See http://code.google.com/p/google-perftools for details.
	-h: histograms
		How many times a sample occurred at each location.
	-f: dynamic functions
		At each sample period, print the name of the executing function.
	-l: dynamic file and line numbers
		At each sample period, print the file and line number of the executing instruction.
	-r: dynamic registers
		At each sample period, print the register contents.
	-s: dynamic function stack traces
		At each sample period, print the symbolic stack trace.

Flag -t sets the maximum real time to sample, in seconds, and -d
sets the sampling interval in milliseconds.  The default is to sample
every 100ms until the program completes.

It is installed as go tool prof and is architecture-independent.

*/
package documentation
