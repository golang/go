// Copyright 2014 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

package cgotest

// Test that we have no more than one build ID.  In the past we used
// to generate a separate build ID for each package using cgo, and the
// linker concatenated them all.  We don't want that--we only want
// one.

import (
	"bytes"
	"debug/elf"
	"os"
	"testing"
)

func testBuildID(t *testing.T) {
	f, err := elf.Open("/proc/self/exe")
	if err != nil {
		if os.IsNotExist(err) {
			t.Skip("no /proc/self/exe")
		}
		t.Fatal("opening /proc/self/exe: ", err)
	}
	defer f.Close()

	c := 0
	for i, s := range f.Sections {
		if s.Type != elf.SHT_NOTE {
			continue
		}

		d, err := s.Data()
		if err != nil {
			t.Logf("reading data of note section %d: %v", i, err)
			continue
		}

		for len(d) > 0 {

			// ELF standards differ as to the sizes in
			// note sections.  Both the GNU linker and
			// gold always generate 32-bit sizes, so that
			// is what we assume here.

			if len(d) < 12 {
				t.Logf("note section %d too short (%d < 12)", i, len(d))
				continue
			}

			namesz := f.ByteOrder.Uint32(d)
			descsz := f.ByteOrder.Uint32(d[4:])
			typ := f.ByteOrder.Uint32(d[8:])

			an := (namesz + 3) &^ 3
			ad := (descsz + 3) &^ 3

			if int(12+an+ad) > len(d) {
				t.Logf("note section %d too short for header (%d < 12 + align(%d,4) + align(%d,4))", i, len(d), namesz, descsz)
				continue
			}

			// 3 == NT_GNU_BUILD_ID
			if typ == 3 && namesz == 4 && bytes.Equal(d[12:16], []byte("GNU\000")) {
				c++
			}

			d = d[12+an+ad:]
		}
	}

	if c > 1 {
		t.Errorf("found %d build ID notes", c)
	}
}
