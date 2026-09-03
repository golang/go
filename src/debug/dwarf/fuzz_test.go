// Copyright 2026 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

package dwarf_test

import (
	"debug/dwarf"
	"debug/elf"
	"debug/macho"
	"os"
	"path/filepath"
	"strings"
	"testing"
)

func FuzzReader(f *testing.F) {
	if testing.Short() {
		f.Skip("Skipping in short mode")
	}

	dwarfSuffix := func(name string) string {
		switch {
		// ELF prefixes, see dwarfSuffix in debug/elf.DWARF
		case strings.HasPrefix(name, ".debug_"):
			return name[7:]
		case strings.HasPrefix(name, ".zdebug_"):
			return name[8:]
		// machO prefixes, see dwarfSuffix in debug/macho.DWARF
		case strings.HasPrefix(name, "__debug_"):
			return name[8:]
		case strings.HasPrefix(name, "__zdebug_"):
			return name[9:]
		default:
			return ""
		}
	}

	testdata, err := os.ReadDir("testdata")
	if err != nil {
		f.Fatalf("failed to read testdata directory: %s", err)
	}
	for _, de := range testdata {
		if de.IsDir() {
			continue
		}

		var abbrev, info, line []byte

		ext := filepath.Ext(de.Name())
		name := filepath.Join("testdata", de.Name())

		switch {
		case strings.HasPrefix(ext, ".elf"):
			e, err := elf.Open(name)
			if err != nil {
				f.Fatal(err)
			}

			for _, s := range e.Sections {
				switch dwarfSuffix(s.Name) {
				case "abbrev":
					abbrev, _ = s.Data()
				case "info":
					info, _ = s.Data()
				case "line":
					line, _ = s.Data()
				}
			}
		case strings.HasPrefix(ext, ".macho"):
			m, err := macho.Open(name)
			if err != nil {
				f.Fatal(err)
			}

			for _, s := range m.Sections {
				switch dwarfSuffix(s.Name) {
				case "abbrev":
					abbrev, _ = s.Data()
				case "info":
					info, _ = s.Data()
				case "line":
					line, _ = s.Data()
				}
			}
		default:
			continue
		}

		if len(abbrev) > 0 || len(info) > 0 || len(line) > 0 {
			f.Add(abbrev, info, line)
		}
	}

	// A DWARF 4 compile unit with a line program containing a
	// DW_LNS_set_file instruction with a file index of math.MaxUint64.
	f.Add(
		[]byte{
			1, 0x11, 0,
			0x10, 0x17,
			0, 0,
			0,
		},
		[]byte{
			0x0c, 0, 0, 0,
			4, 0,
			0, 0, 0, 0,
			8,
			1,
			0, 0, 0, 0,
		},
		[]byte{
			0x25, 0, 0, 0,
			4, 0,
			0x14, 0, 0, 0,
			1, 1, 1, 0, 1, 13,
			0, 1, 1, 1, 1, 0, 0, 0, 1, 0, 0, 1,
			0, 0,
			4,
			0xff, 0xff, 0xff, 0xff, 0xff, 0xff, 0xff, 0xff, 0xff, 1,
		},
	)

	f.Fuzz(func(t *testing.T, abbrev, info, line []byte) {
		d, err := dwarf.New(abbrev, nil, nil, info, line, nil, nil, nil)
		if err != nil {
			return
		}

		r := d.Reader()
		for {
			e, err := r.Next()
			if err != nil || e == nil {
				return
			}
			if e.Tag != dwarf.TagCompileUnit {
				continue
			}
			lr, err := d.LineReader(e)
			if err != nil || lr == nil {
				continue
			}
			var lineEntry dwarf.LineEntry
			for {
				if err := lr.Next(&lineEntry); err != nil {
					break
				}
			}
		}
	})
}
