// Copyright 2013 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

// Support for test coverage.

package testing

import (
	"fmt"
	"os"
)

// CoverBlock records the coverage data for a single basic block.
// NOTE: This struct is internal to the testing infrastructure and may change.
// It is not covered (yet) by the Go 1 compatibility guidelines.
type CoverBlock struct {
	Line0 uint32
	Col0  uint16
	Line1 uint32
	Col1  uint16
	Stmts uint16
}

var (
	coverCounters map[string][]uint32
	coverBlocks   map[string][]CoverBlock
)

// RegisterCover records the coverage data accumulators for the tests.
// NOTE: This struct is internal to the testing infrastructure and may change.
// It is not covered (yet) by the Go 1 compatibility guidelines.
func RegisterCover(c map[string][]uint32, b map[string][]CoverBlock) {
	coverCounters = c
	coverBlocks = b
}

// mustBeNil checks the error and, if present, reports it and exits.
func mustBeNil(err error) {
	if err != nil {
		fmt.Fprintf(os.Stderr, "testing: %s\n", err)
		os.Exit(2)
	}
}

// coverReport reports the coverage percentage and writes a coverage profile if requested.
func coverReport() {
	var f *os.File
	var err error
	if *coverProfile != "" {
		f, err = os.Create(toOutputDir(*coverProfile))
		mustBeNil(err)
		defer func() { mustBeNil(f.Close()) }()
	}

	var active, total int64
	packageName := ""
	for name, counts := range coverCounters {
		if packageName == "" {
			// Package name ends at last slash.
			for i, c := range name {
				if c == '/' {
					packageName = name[:i]
				}
			}
		}
		blocks := coverBlocks[name]
		for i, count := range counts {
			stmts := int64(blocks[i].Stmts)
			total += stmts
			if count > 0 {
				active += stmts
			}
			if f != nil {
				_, err := fmt.Fprintf(f, "%s:%d.%d,%d.%d %d %d\n", name,
					blocks[i].Line0, blocks[i].Col0,
					blocks[i].Line1, blocks[i].Col1,
					stmts,
					count)
				mustBeNil(err)
			}
		}
	}
	if total == 0 {
		total = 1
	}
	if packageName == "" {
		packageName = "package"
	}
	fmt.Printf("test coverage for %s: %.1f%% of statements\n", packageName, 100*float64(active)/float64(total))
}
