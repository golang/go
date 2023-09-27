// Copyright 2018 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

// This file provides a simple framework to add benchmarks
// based on generated input (source) files.

package format_test

import (
	"bytes"
	"flag"
	"fmt"
	"go/format"
	"os"
	"testing"
)

var debug = flag.Bool("debug", false, "write .src files containing formatting input; for debugging")

// array1 generates an array literal with n elements of the form:
//
// var _ = [...]byte{
//
//	// 0
//	0x00, 0x01, 0x02, 0x03, 0x04, 0x05, 0x06, 0x07,
//	0x08, 0x09, 0x0a, 0x0b, 0x0c, 0x0d, 0x0e, 0x0f,
//	0x10, 0x11, 0x12, 0x13, 0x14, 0x15, 0x16, 0x17,
//	0x18, 0x19, 0x1a, 0x1b, 0x1c, 0x1d, 0x1e, 0x1f,
//	0x20, 0x21, 0x22, 0x23, 0x24, 0x25, 0x26, 0x27,
//	// 40
//	0x28, 0x29, 0x2a, 0x2b, 0x2c, 0x2d, 0x2e, 0x2f,
//	0x30, 0x31, 0x32, 0x33, 0x34, 0x35, 0x36, 0x37,
//	...
func array1(buf *bytes.Buffer, n int) {
	buf.WriteString("var _ = [...]byte{\n")
	for i := 0; i < n; {
		if i%10 == 0 {
			fmt.Fprintf(buf, "\t// %d\n", i)
		}
		buf.WriteByte('\t')
		for j := 0; j < 8; j++ {
			fmt.Fprintf(buf, "0x%02x, ", byte(i))
			i++
		}
		buf.WriteString("\n")
	}
	buf.WriteString("}\n")
}

var tests = []struct {
	name string
	gen  func(*bytes.Buffer, int)
	n    int
}{
	{"array1", array1, 10000},
	// add new test cases here as needed
}

func BenchmarkFormat(b *testing.B) {
	var src bytes.Buffer
	for _, t := range tests {
		src.Reset()
		src.WriteString("package p\n")
		t.gen(&src, t.n)
		data := src.Bytes()

		if *debug {
			filename := t.name + ".src"
			err := os.WriteFile(filename, data, 0660)
			if err != nil {
				b.Fatalf("couldn't write %s: %v", filename, err)
			}
		}

		b.Run(fmt.Sprintf("%s-%d", t.name, t.n), func(b *testing.B) {
			b.SetBytes(int64(len(data)))
			b.ReportAllocs()
			b.ResetTimer()
			for i := 0; i < b.N; i++ {
				var err error
				sink, err = format.Source(data)
				if err != nil {
					b.Fatal(err)
				}
			}
		})
	}
}

var sink []byte
