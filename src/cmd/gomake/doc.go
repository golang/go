// Copyright 2010 The Go Authors.  All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

/*
The gomake command runs GNU make with an appropriate environment
for using the conventional Go makefiles.  If $GOROOT is already
set in the environment, running gomake is exactly the same
as running make (or, on BSD systems, running gmake).

Usage: gomake [ target ... ]

Common targets are:

	all (default)
		build the package or command, but do not install it.

	install
		build and install the package or command

	test
		run the tests (packages only)

	bench
		run benchmarks (packages only)

	clean
		remove object files from the current directory

	nuke
		make clean and remove the installed package or command

See http://golang.org/doc/code.html for information about
writing makefiles.
*/
package documentation
