// Copyright 2009 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

/*

Gopack program is a variant of the Plan 9 ar tool.  The original is documented at

	http://plan9.bell-labs.com/magic/man2html/1/ar

It adds a special Go-specific section __.PKGDEF that collects all the
Go type information from the files in the archive; that section is
used by the compiler when importing the package during compilation.

Usage: gopack [uvnbailo][mrxtdpq] archive files ...

The new option 'g' causes gopack to maintain the __.PKGDEF section
as files are added to the archive.

*/
package documentation
