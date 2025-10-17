// Copyright 2025 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

package main

import (
	"bytes"
	"os"
	"testing"
)

func TestNoChange(t *testing.T) {
	classes := makeClasses()
	sizeToSizeClass := makeSizeToSizeClass(classes)

	outfile := "../malloc_generated.go"
	want, err := os.ReadFile(outfile)
	if err != nil {
		t.Fatal(err)
	}
	got := mustFormat(inline(specializedMallocConfig(classes, sizeToSizeClass)))
	if !bytes.Equal(want, got) {
		t.Fatalf("want:\n%s\ngot:\n%s\n", withLineNumbers(want), withLineNumbers(got))
	}

	tablefile := "../malloc_tables_generated.go"
	wanttable, err := os.ReadFile(tablefile)
	if err != nil {
		t.Fatal(err)
	}
	gotTable := mustFormat(generateTable(sizeToSizeClass))
	if !bytes.Equal(wanttable, gotTable) {
		t.Fatalf("want:\n%s\ngot:\n%s\n", withLineNumbers(wanttable), withLineNumbers(gotTable))
	}
}
