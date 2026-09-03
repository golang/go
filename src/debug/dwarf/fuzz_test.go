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
