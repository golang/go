// Copyright 2026 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

package elf_test

import (
	"bytes"
	"compress/gzip"
	"debug/elf"
	"io"
	"os"
	"path/filepath"
	"testing"
)

func FuzzReader(f *testing.F) {
	if testing.Short() {
		f.Skip("Skipping in short mode")
	}

	testdata, err := os.ReadDir("testdata")
	if err != nil {
		f.Fatalf("failed to read testdata directory: %s", err)
	}
	for _, de := range testdata {
		if de.IsDir() || filepath.Ext(de.Name()) == ".c" {
			continue
		}
		b, err := os.ReadFile(filepath.Join("testdata", de.Name()))
		if err != nil {
			f.Fatalf("failed to read testdata: %s", err)
		}
		if filepath.Ext(de.Name()) == ".gz" {
			gz, err := gzip.NewReader(bytes.NewBuffer(b))
			if err != nil {
				f.Fatalf("failed to read gzip testdata: %s", err)
			}
			b, err = io.ReadAll(gz)
			if err != nil {
				f.Fatalf("failed to read gzip testdata: %s", err)
			}
		}
		f.Add(b)
	}

	f.Fuzz(func(t *testing.T, data []byte) {
		f, err := elf.NewFile(bytes.NewReader(data))
		if err != nil {
			return
		}
		defer f.Close()
		f.DynString(elf.DT_SONAME)
		f.DynString(elf.DT_RPATH)
		f.DynValue(elf.DT_FLAGS)
		f.DynValue(elf.DT_VERNEEDNUM)
		f.DynamicSymbols()
		f.DynamicVersionNeeds()
		f.DynamicVersions()
		f.ImportedLibraries()
		f.ImportedSymbols()
		f.Section(".data")
		f.Section(".text")
		f.SectionByType(elf.SHT_GNU_VERSYM)
		f.Symbols()
		dw, err := f.DWARF()
		if err != nil {
			return
		}
		dr := dw.Reader()
		for {
			e, _ := dr.Next()
			if e == nil {
				break
			}
		}
	})
}
