// Copyright 2019 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

// +build gofuzz

package csv

import (
	"bytes"
	"fmt"
	"reflect"
)

func Fuzz(data []byte) int {
	score := 0
	buf := new(bytes.Buffer)

	for _, tt := range []Reader{
		Reader{},
		Reader{Comma: ';'},
		Reader{Comma: '\t'},
		Reader{LazyQuotes: true},
		Reader{TrimLeadingSpace: true},
		Reader{Comment: '#'},
		Reader{Comment: ';'},
	} {
		r := NewReader(bytes.NewReader(data))
		r.Comma = tt.Comma
		r.Comment = tt.Comment
		r.LazyQuotes = tt.LazyQuotes
		r.TrimLeadingSpace = tt.TrimLeadingSpace

		records, err := r.ReadAll()
		if err != nil {
			continue
		}
		score = 1

		buf.Reset()
		w := NewWriter(buf)
		w.Comma = tt.Comma
		err = w.WriteAll(records)
		if err != nil {
			fmt.Printf("writer  = %#v\n", w)
			fmt.Printf("records = %v\n", records)
			panic(err)
		}

		r = NewReader(buf)
		r.Comma = tt.Comma
		r.Comment = tt.Comment
		r.LazyQuotes = tt.LazyQuotes
		r.TrimLeadingSpace = tt.TrimLeadingSpace
		result, err := r.ReadAll()
		if err != nil {
			fmt.Printf("reader  = %#v\n", r)
			fmt.Printf("records = %v\n", records)
			panic(err)
		}

		if !reflect.DeepEqual(records, result) {
			fmt.Println("records = \n", records)
			fmt.Println("result  = \n", records)
			panic("not equal")
		}
	}

	return score
}
