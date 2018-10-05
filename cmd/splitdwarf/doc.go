// Copyright 2018 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

/*

Splitdwarf uncompresses and copies the DWARF segment of a Mach-O
executable into the "dSYM" file expected by lldb and ports of gdb
on OSX.

Usage: splitdwarf osxMachoFile [ osxDsymFile ]

Unless a dSYM file name is provided on the command line,
splitdwarf will place it where the OSX tools expect it, in
"<osxMachoFile>.dSYM/Contents/Resources/DWARF/<osxMachoFile>",
creating directories as necessary.

*/
package main // import "golang.org/x/tools/cmd/splitdwarf"
