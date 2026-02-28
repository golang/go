// Copyright 2019 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

package csv

import (
	"bytes"
	"reflect"
	"slices"
	"strings"
	"testing"
)

func FuzzRoundtrip(f *testing.F) {
	f.Fuzz(func(t *testing.T, in []byte) {
		buf := new(bytes.Buffer)

		t.Logf("input = %q", in)
		for _, tt := range []Reader{
			{Comma: ','},
			{Comma: ';'},
			{Comma: '\t'},
			{Comma: ',', LazyQuotes: true},
			{Comma: ',', TrimLeadingSpace: true},
			{Comma: ',', Comment: '#'},
			{Comma: ',', Comment: ';'},
		} {
			t.Logf("With options:")
			t.Logf("  Comma            = %q", tt.Comma)
			t.Logf("  LazyQuotes       = %t", tt.LazyQuotes)
			t.Logf("  TrimLeadingSpace = %t", tt.TrimLeadingSpace)
			t.Logf("  Comment          = %q", tt.Comment)
			r := NewReader(bytes.NewReader(in))
			r.Comma = tt.Comma
			r.Comment = tt.Comment
			r.LazyQuotes = tt.LazyQuotes
			r.TrimLeadingSpace = tt.TrimLeadingSpace

			records, err := r.ReadAll()
			if err != nil {
				continue
			}
			t.Logf("first records = %#v", records)

			buf.Reset()
			w := NewWriter(buf)
			w.Comma = tt.Comma
			err = w.WriteAll(records)
			if err != nil {
				t.Logf("writer  = %#v\n", w)
				t.Logf("records = %v\n", records)
				t.Fatal(err)
			}
			if tt.Comment != 0 {
				// Writer doesn't support comments, so it can turn the quoted record "#"
				// into a non-quoted comment line, failing the roundtrip check below.
				continue
			}
			t.Logf("second input = %q", buf.Bytes())

			r = NewReader(buf)
			r.Comma = tt.Comma
			r.Comment = tt.Comment
			r.LazyQuotes = tt.LazyQuotes
			r.TrimLeadingSpace = tt.TrimLeadingSpace
			result, err := r.ReadAll()
			if err != nil {
				t.Logf("reader  = %#v\n", r)
				t.Logf("records = %v\n", records)
				t.Fatal(err)
			}

			// The reader turns \r\n into \n.
			for _, record := range records {
				for i, s := range record {
					record[i] = strings.ReplaceAll(s, "\r\n", "\n")
				}
			}
			// Note that the reader parses the quoted record "" as an empty string,
			// and the writer turns that into an empty line, which the reader skips over.
			// Filter those out to avoid false positives.
			records = slices.DeleteFunc(records, func(record []string) bool {
				return len(record) == 1 && record[0] == ""
			})
			// The reader uses nil when returning no records at all.
			if len(records) == 0 {
				records = nil
			}

			if !reflect.DeepEqual(records, result) {
				t.Fatalf("first read got %#v, second got %#v", records, result)
			}
		}
	})
}
