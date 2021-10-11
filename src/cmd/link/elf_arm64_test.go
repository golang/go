// Copyright 2021 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

//go:build linux && arm64
// +build linux,arm64

package main

import (
	"debug/elf"
	"internal/testenv"
	"io/ioutil"
	"os"
	"os/exec"
	"path/filepath"
	"sort"
	"sync"
	"testing"
)

func TestMappingSymbols(t *testing.T) {
	testenv.MustHaveGoBuild(t)
	t.Parallel()

	buildmodes := []string{"exe", "pie"}

	var wg sync.WaitGroup
	wg.Add(len(buildmodes))

	for _, buildmode := range buildmodes {
		go func(mode string) {
			defer wg.Done()
			symbols := buildSymbols(t, mode)
			checkMappingSymbols(t, symbols)
		}(buildmode)
	}

	wg.Wait()
}

var goProgram = `
package main
func main() {}
`

// Builds a simple program, then returns a corresponding symbol table for that binary
func buildSymbols(t *testing.T, mode string) []elf.Symbol {
	goTool := testenv.GoToolPath(t)

	dir := t.TempDir()

	gopath := filepath.Join(dir, "GOPATH")
	env := append(os.Environ(), "GOPATH="+gopath)

	goFile := filepath.Join(dir, "main.go")
	if err := ioutil.WriteFile(goFile, []byte(goProgram), 0444); err != nil {
		t.Fatal(err)
	}

	cmd := exec.Command(goTool, "build", "-o", mode, "-buildmode="+mode)
	cmd.Dir = dir
	cmd.Env = env

	t.Logf("%s build", goTool)
	if out, err := cmd.CombinedOutput(); err != nil {
		t.Logf("%s", out)
		t.Fatal(err)
	}

	bin := filepath.Join(dir, mode)

	elfexe, err := elf.Open(bin)
	if err != nil {
		t.Fatal(err)
	}

	symbols, err := elfexe.Symbols()
	if err != nil {
		t.Fatal(err)
	}

	return symbols
}

// Checks that mapping symbols are inserted correctly inside a symbol table.
func checkMappingSymbols(t *testing.T, symbols []elf.Symbol) {
	// textValues variable stores addresses of function symbols,
	// it helps ensuring that "$x" symbol has a correct place (at the beginning of a function)
	textValues := make(map[uint64]struct{})
	for _, symbol := range symbols {
		if elf.ST_TYPE(symbol.Info) == elf.STT_FUNC {
			textValues[symbol.Value] = struct{}{}
		}
	}

	// mappingSymbols variable keeps only "$x" and "$d" symbols sorted by their position.
	var mappingSymbols []elf.Symbol
	for _, symbol := range symbols {
		if symbol.Name == "$x" || symbol.Name == "$d" {
			if elf.ST_TYPE(symbol.Info) != elf.STT_NOTYPE || elf.ST_BIND(symbol.Info) != elf.STB_LOCAL {
				t.Fatalf("met \"%v\" symbol at %v position with incorrect info %v", symbol.Name, symbol.Value, symbol.Info)
			}
			mappingSymbols = append(mappingSymbols, symbol)
		}
	}
	sort.Slice(mappingSymbols, func(i, j int) bool {
		return mappingSymbols[i].Value < mappingSymbols[j].Value
	})

	hasData := false
	hasText := false

	needCodeSymb := true
	for _, symbol := range mappingSymbols {
		if symbol.Name == "$x" {
			hasText = true

			_, has := textValues[symbol.Value]
			if !has {
				t.Fatalf("met \"$x\" symbol at %v position which is not a beginning of the function", symbol.Value)
			}

			needCodeSymb = false
			continue
		}

		hasData = true

		if needCodeSymb {
			t.Fatalf("met unexpected \"$d\" symbol at %v position, (\"$x\" should always go after \"$d\", not another \"$d\")", symbol.Value)
		}
		needCodeSymb = true
	}

	if !hasText || !hasData {
		t.Fatal("binary does not have mapping symbols")
	}
}

