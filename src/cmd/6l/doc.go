// Copyright 2009 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

/*

6l is a modified version of the Plan 9 loader documented at

	http://plan9.bell-labs.com/magic/man2html/1/2l

Its target architecture is the x86-64, referred to by these tools as amd64.

Major changes include:
	- support for ELF and Mach-O binary files 
	- support for segmented stacks (this feature is implemented here, not in the compilers).


Original options are listed in the link above.

Options new in this version:

-H6
	Write Apple Mach-O binaries (default when $GOOS is darwin)
-H7
	Write Linux ELF binaries (default when $GOOS is linux)
-L dir1,dir2,..
	Search for libraries (package files) in the comma-separated list of directories.
	The default is the single location $GOROOT/pkg/$GOOS_amd64.


*/
package documentation
