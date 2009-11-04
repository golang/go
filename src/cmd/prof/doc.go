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


Usage: prof -p pid [-t total_secs] [-d delta_msec] [6.out args ...]

The formats (default -h) are:

	-h: histograms
		How many times a sample occurred at each location
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

For reasons of disambiguation it is installed as 6prof although it also serves
as an 8prof and a 5prof.

*/
package documentation
