// Copyright 2011 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

/*
Gofix finds Go programs that use old APIs and rewrites them to use
newer ones.  After you update to a new Go release, gofix helps make
the necessary changes to your programs.

Usage:
	gofix [-r name,...] [path ...]

Without an explicit path, gofix reads standard input and writes the
result to standard output.

If the named path is a file, gofix rewrites the named files in place.
If the named path is a directory, gofix rewrites all .go files in that
directory tree.  When gofix rewrites a file, it prints a line to standard
error giving the name of the file and the rewrite applied.

The -r flag restricts the set of rewrites considered to those in the
named list.  By default gofix considers all known rewrites.  Gofix's
rewrites are idempotent, so that it is safe to apply gofix to updated
or partially updated code even without using the -r flag.

Gofix prints the full list of fixes it can apply in its help output;
to see them, run gofix -?.

Gofix does not make backup copies of the files that it edits.
Instead, use a version control system's ``diff'' functionality to inspect
the changes that gofix makes before committing them.

*/
package documentation
