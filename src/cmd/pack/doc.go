// Copyright 2014 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

/*

Pack is a simple version of the traditional Unix ar tool.
It implements only the operations needed by Go.

Usage:
	go tool pack op file.a [name...]

Pack applies the operation to the archive, using the names as arguments to the operation.

The operation op is given by one of these letters:

	c	append files (from the file system) to a new archive
	p	print files from the archive
	r	append files (from the file system) to the archive
	t	list files from the archive
	x	extract files from the archive

The archive argument to the c command must be non-existent or a
valid archive file, which will be cleared before adding new entries. It
is an error if the file exists but is not an archive.

For the p, t, and x commands, listing no names on the command line
causes the operation to apply to all files in the archive.

In contrast to Unix ar, the r operation always appends to the archive,
even if a file with the given name already exists in the archive. In this way
pack's r operation is more like Unix ar's rq operation.

Adding the letter v to an operation, as in pv or rv, enables verbose operation:
For the c and r commands, names are printed as files are added.
For the p command, each file is prefixed by the name on a line by itself.
For the t command, the listing includes additional file metadata.
For the x command, names are printed as files are extracted.

*/
package main
