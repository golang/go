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

var (
	testedPackage  string // The package being tested.
	coveredPackage string // List of the package[s] being covered, if distinct from the tested package.
)

// RegisterCover records the coverage data accumulators for the tests.
// NOTE: This struct is internal to the testing infrastructure and may change.
// It is not covered (yet) by the Go 1 compatibility guidelines.
func RegisterCover(c map[string][]uint32, b map[string][]CoverBlock) {
	coverCounters = c
	coverBlocks = b
}

// CoveredPackage records the names of the packages being tested and covered.
// NOTE: This function is internal to the testing infrastructure and may change.
// It is not covered (yet) by the Go 1 compatibility guidelines.
func CoveredPackage(tested, covered string) {
	testedPackage = tested
	coveredPackage = covered
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
		fmt.Fprintf(f, "mode: %s\n", *coverMode)
		defer func() { mustBeNil(f.Close()) }()
	}

	var active, total int64
	for name, counts := range coverCounters {
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
	fmt.Printf("coverage for %s: %.1f%% of statements%s\n", testedPackage, 100*float64(active)/float64(total), coveredPackage)
}
