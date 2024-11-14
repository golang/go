// Copyright 2023 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

package zstd

import (
	"slices"
	"testing"
)

// literalPredefinedDistribution is the predefined distribution table
// for literal lengths. RFC 3.1.1.3.2.2.1.
var literalPredefinedDistribution = []int16{
	4, 3, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 1, 1, 1,
	2, 2, 2, 2, 2, 2, 2, 2, 2, 3, 2, 1, 1, 1, 1, 1,
	-1, -1, -1, -1,
}

// offsetPredefinedDistribution is the predefined distribution table
// for offsets. RFC 3.1.1.3.2.2.3.
var offsetPredefinedDistribution = []int16{
	1, 1, 1, 1, 1, 1, 2, 2, 2, 1, 1, 1, 1, 1, 1, 1,
	1, 1, 1, 1, 1, 1, 1, 1, -1, -1, -1, -1, -1,
}

// matchPredefinedDistribution is the predefined distribution table
// for match lengths. RFC 3.1.1.3.2.2.2.
var matchPredefinedDistribution = []int16{
	1, 4, 3, 2, 2, 2, 2, 2, 2, 1, 1, 1, 1, 1, 1, 1,
	1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1,
	1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, -1, -1,
	-1, -1, -1, -1, -1,
}

// TestPredefinedTables verifies that we can generate the predefined
// literal/offset/match tables from the input data in RFC 8878.
// This serves as a test of the predefined tables, and also of buildFSE
// and the functions that make baseline FSE tables.
func TestPredefinedTables(t *testing.T) {
	tests := []struct {
		name         string
		distribution []int16
		tableBits    int
		toBaseline   func(*Reader, int, []fseEntry, []fseBaselineEntry) error
		predef       []fseBaselineEntry
	}{
		{
			name:         "literal",
			distribution: literalPredefinedDistribution,
			tableBits:    6,
			toBaseline:   (*Reader).makeLiteralBaselineFSE,
			predef:       predefinedLiteralTable[:],
		},
		{
			name:         "offset",
			distribution: offsetPredefinedDistribution,
			tableBits:    5,
			toBaseline:   (*Reader).makeOffsetBaselineFSE,
			predef:       predefinedOffsetTable[:],
		},
		{
			name:         "match",
			distribution: matchPredefinedDistribution,
			tableBits:    6,
			toBaseline:   (*Reader).makeMatchBaselineFSE,
			predef:       predefinedMatchTable[:],
		},
	}
	for _, test := range tests {
		test := test
		t.Run(test.name, func { t ->
			var r Reader
			table := make([]fseEntry, 1<<test.tableBits)
			if err := r.buildFSE(0, test.distribution, table, test.tableBits); err != nil {
				t.Fatal(err)
			}

			baselineTable := make([]fseBaselineEntry, len(table))
			if err := test.toBaseline(&r, 0, table, baselineTable); err != nil {
				t.Fatal(err)
			}

			if !slices.Equal(baselineTable, test.predef) {
				t.Errorf("got %v, want %v", baselineTable, test.predef)
			}
		})
	}
}
