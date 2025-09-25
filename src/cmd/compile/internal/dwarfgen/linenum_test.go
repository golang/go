// Copyright 2025 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

package dwarfgen

import (
	"debug/dwarf"
	"internal/platform"
	"internal/testenv"
	"io"
	"runtime"
	"testing"
)

func TestIssue75249(t *testing.T) {
	testenv.MustHaveGoRun(t)
	t.Parallel()

	if !platform.ExecutableHasDWARF(runtime.GOOS, runtime.GOARCH) {
		t.Skipf("skipping on %s/%s: no DWARF symbol table in executables", runtime.GOOS, runtime.GOARCH)
	}

	code := `
package main

type Data struct {
        Field1 int
        Field2 *int
        Field3 int
        Field4 *int
        Field5 int
        Field6 *int
        Field7 int
        Field8 *int
}

//go:noinline
func InitializeData(d *Data) {
        d.Field1++            // line 16
        d.Field2 = d.Field4
        d.Field3++
        d.Field4 = d.Field6
        d.Field5++
        d.Field6 = d.Field8
        d.Field7++
        d.Field8 = d.Field2   // line 23
}

func main() {
        var data Data
        InitializeData(&data)
}
`

	_, f := gobuild(t, t.TempDir(), true, []testline{{line: code}})
	defer f.Close()

	dwarfData, err := f.DWARF()
	if err != nil {
		t.Fatal(err)
	}
	dwarfReader := dwarfData.Reader()

	for {
		entry, err := dwarfReader.Next()
		if err != nil {
			t.Fatal(err)
		}
		if entry == nil {
			break
		}
		if entry.Tag != dwarf.TagCompileUnit {
			continue
		}
		name := entry.AttrField(dwarf.AttrName)
		if name == nil || name.Class != dwarf.ClassString || name.Val != "main" {
			continue
		}
		lr, err := dwarfData.LineReader(entry)
		if err != nil {
			t.Fatal(err)
		}
		stmts := map[int]bool{}
		for {
			var le dwarf.LineEntry
			err := lr.Next(&le)
			if err == io.EOF {
				break
			}
			if err != nil {
				t.Fatal(err)
			}
			if !le.IsStmt {
				continue
			}
			stmts[le.Line] = true
		}
		for i := 16; i <= 23; i++ {
			if !stmts[i] {
				t.Errorf("missing statement at line %d", i)
			}
		}
	}
}
