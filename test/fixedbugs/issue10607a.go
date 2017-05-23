// skip

// Copyright 2015 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

// This is built by issue10607.go with a -B option.
// Verify that we have one build-id note with the expected value.

package main

import (
	"bytes"
	"debug/elf"
	"fmt"
	"os"
)

func main() {
	f, err := elf.Open("/proc/self/exe")
	if err != nil {
		if os.IsNotExist(err) {
			return
		}
		fmt.Fprintln(os.Stderr, "opening /proc/self/exe:", err)
		os.Exit(1)
	}

	c := 0
	fail := false
	for i, s := range f.Sections {
		if s.Type != elf.SHT_NOTE {
			continue
		}

		d, err := s.Data()
		if err != nil {
			fmt.Fprintln(os.Stderr, "reading data of note section %d: %v", i, err)
			continue
		}

		for len(d) > 0 {
			namesz := f.ByteOrder.Uint32(d)
			descsz := f.ByteOrder.Uint32(d[4:])
			typ := f.ByteOrder.Uint32(d[8:])

			an := (namesz + 3) &^ 3
			ad := (descsz + 3) &^ 3

			if int(12+an+ad) > len(d) {
				fmt.Fprintf(os.Stderr, "note section %d too short for header (%d < 12 + align(%d,4) + align(%d,4))\n", i, len(d), namesz, descsz)
				break
			}

			// 3 == NT_GNU_BUILD_ID
			if typ == 3 && namesz == 4 && bytes.Equal(d[12:16], []byte("GNU\000")) {
				id := string(d[12+an:12+an+descsz])
				if id == "\x12\x34\x56\x78" {
					c++
				} else {
					fmt.Fprintf(os.Stderr, "wrong build ID data: %q\n", id)
					fail = true
				}
			}

			d = d[12+an+ad:]
		}
	}

	if c == 0 {
		fmt.Fprintln(os.Stderr, "no build-id note")
		fail = true
	} else if c > 1 {
		fmt.Fprintln(os.Stderr, c, "build-id notes")
		fail = true
	}

	if fail {
		os.Exit(1)
	}
}
