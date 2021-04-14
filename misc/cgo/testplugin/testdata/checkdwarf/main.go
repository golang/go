// Copyright 2018 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

// Usage:
//
//  checkdwarf <exe> <suffix>
//
// Opens <exe>, which must be an executable or a library and checks that
// there is an entry in .debug_info whose name ends in <suffix>

package main

import (
	"debug/dwarf"
	"debug/elf"
	"debug/macho"
	"debug/pe"
	"fmt"
	"os"
	"strings"
)

func usage() {
	fmt.Fprintf(os.Stderr, "checkdwarf executable-or-library DIE-suffix\n")
}

type dwarfer interface {
	DWARF() (*dwarf.Data, error)
}

func openElf(path string) dwarfer {
	exe, err := elf.Open(path)
	if err != nil {
		return nil
	}
	return exe
}

func openMacho(path string) dwarfer {
	exe, err := macho.Open(path)
	if err != nil {
		return nil
	}
	return exe
}

func openPE(path string) dwarfer {
	exe, err := pe.Open(path)
	if err != nil {
		return nil
	}
	return exe
}

func main() {
	if len(os.Args) != 3 {
		usage()
	}

	exePath := os.Args[1]
	dieSuffix := os.Args[2]

	var exe dwarfer

	for _, openfn := range []func(string) dwarfer{openMacho, openPE, openElf} {
		exe = openfn(exePath)
		if exe != nil {
			break
		}
	}

	if exe == nil {
		fmt.Fprintf(os.Stderr, "could not open %s\n", exePath)
		os.Exit(1)
	}

	data, err := exe.DWARF()
	if err != nil {
		fmt.Fprintf(os.Stderr, "%s: error opening DWARF: %v\n", exePath, err)
		os.Exit(1)
	}

	rdr := data.Reader()
	for {
		e, err := rdr.Next()
		if err != nil {
			fmt.Fprintf(os.Stderr, "%s: error reading DWARF: %v\n", exePath, err)
			os.Exit(1)
		}
		if e == nil {
			break
		}
		name, hasname := e.Val(dwarf.AttrName).(string)
		if !hasname {
			continue
		}
		if strings.HasSuffix(name, dieSuffix) {
			// found
			os.Exit(0)
		}
	}

	fmt.Fprintf(os.Stderr, "%s: no entry with a name ending in %q was found\n", exePath, dieSuffix)
	os.Exit(1)
}
