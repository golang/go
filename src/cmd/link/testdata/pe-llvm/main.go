// Copyright 2020 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

// Test that a PE rsrc section is handled correctly, when the object files
// have been created by llvm-rc or msvc's rc.exe, which means there's the
// @feat.00 symbol as well as split .rsrc$00 and .rsrc$01 section to deal with.
//
// rsrc.syso is created using llvm with:
//    {i686,x86_64,armv7,arm64}-w64-mingw32-windres -i a.rc -o rsrc_$GOARCH.syso -O coff
// where this windres calls into llvm-rc and llvm-cvtres. The source file,
// a.rc, simply contains a reference to its own bytes:
//
//    resname RCDATA a.rc
//
// Object dumping the resultant rsrc.syso, we can see the split sections and
// the @feat.00 SEH symbol:
//
//     rsrc.syso:      file format coff-x86-64
//
//     architecture: x86_64
//     start address: 0x0000000000000000
//
//     Export Table:
//     Sections:
//     Idx Name          Size     VMA              Type
//       0 .rsrc$01      00000068 0000000000000000 DATA
//       1 .rsrc$02      00000018 0000000000000000 DATA
//
//     SYMBOL TABLE:
//     [ 0](sec -1)(fl 0x00)(ty   0)(scl   3) (nx 0) 0x00000011 @feat.00
//     [ 1](sec  1)(fl 0x00)(ty   0)(scl   3) (nx 1) 0x00000000 .rsrc$01
//     AUX scnlen 0x68 nreloc 1 nlnno 0 checksum 0x0 assoc 0 comdat 0
//     [ 3](sec  2)(fl 0x00)(ty   0)(scl   3) (nx 1) 0x00000000 .rsrc$02
//     AUX scnlen 0x18 nreloc 0 nlnno 0 checksum 0x0 assoc 0 comdat 0
//     [ 5](sec  2)(fl 0x00)(ty   0)(scl   3) (nx 0) 0x00000000 $R000000
//     RELOCATION RECORDS FOR [.rsrc$01]:
//     OFFSET           TYPE                     VALUE
//     0000000000000048 IMAGE_REL_AMD64_ADDR32NB $R000000

package main

func main() {}
