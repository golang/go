// Copyright 2026 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

package pe_test

import (
	"bytes"
	"debug/pe"
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
		f.Add(b)
	}

	f.Fuzz(func(t *testing.T, data []byte) {
		f, err := pe.NewFile(bytes.NewReader(data))
		if err != nil {
			return
		}
		defer f.Close()
		f.ImportedLibraries()
		f.ImportedSymbols()
		f.Section(".data")
		f.Section(".text")
		for _, c := range f.COFFSymbols {
			_, err := c.FullName(f.StringTable)
			if err != nil {
				continue
			}
		}
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
