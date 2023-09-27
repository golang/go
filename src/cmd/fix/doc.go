// Copyright 2011 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

/*
Fix finds Go programs that use old APIs and rewrites them to use
newer ones.  After you update to a new Go release, fix helps make
the necessary changes to your programs.

Usage:

	go tool fix [-r name,...] [path ...]

Without an explicit path, fix reads standard input and writes the
result to standard output.

If the named path is a file, fix rewrites the named files in place.
If the named path is a directory, fix rewrites all .go files in that
directory tree.  When fix rewrites a file, it prints a line to standard
error giving the name of the file and the rewrite applied.

If the -diff flag is set, no files are rewritten. Instead fix prints
the differences a rewrite would introduce.

The -r flag restricts the set of rewrites considered to those in the
named list.  By default fix considers all known rewrites.  Fix's
rewrites are idempotent, so that it is safe to apply fix to updated
or partially updated code even without using the -r flag.

Fix prints the full list of fixes it can apply in its help output;
to see them, run go tool fix -help.

Fix does not make backup copies of the files that it edits.
Instead, use a version control system's “diff” functionality to inspect
the changes that fix makes before committing them.
*/
package main
