// Copyright 2019 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

package cache

import (
	"strings"
	"testing"
)

func TestParseErrorMessage(t *testing.T) {
	tests := []struct {
		name             string
		in               string
		expectedFileName string
		expectedLine     int
		expectedColumn   int
	}{
		{
			name:             "from go list output",
			in:               "\nattributes.go:13:1: expected 'package', found 'type'",
			expectedFileName: "attributes.go",
			expectedLine:     13,
			expectedColumn:   1,
		},
	}

	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			spn := parseGoListError(tt.in, ".")
			fn := spn.URI().Filename()

			if !strings.HasSuffix(fn, tt.expectedFileName) {
				t.Errorf("expected filename with suffix %v but got %v", tt.expectedFileName, fn)
			}

			if !spn.HasPosition() {
				t.Fatalf("expected span to have position")
			}

			pos := spn.Start()
			if pos.Line() != tt.expectedLine {
				t.Errorf("expected line %v but got %v", tt.expectedLine, pos.Line())
			}

			if pos.Column() != tt.expectedColumn {
				t.Errorf("expected line %v but got %v", tt.expectedLine, pos.Line())
			}
		})
	}
}
