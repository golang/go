// Copyright 2011 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

package norm

import (
	"bytes"
	"fmt"
	"os"
	"strings"
	"testing"
)

var ioTests = []AppendTest{
	{"", strings.Repeat("a\u0316\u0300", 6), strings.Repeat("\u00E0\u0316", 6)},
	{"", strings.Repeat("a\u0300\u0316", 4000), strings.Repeat("\u00E0\u0316", 4000)},
	{"", strings.Repeat("\x80\x80", 4000), strings.Repeat("\x80\x80", 4000)},
	{"", "\u0041\u0307\u0304", "\u01E0"},
}

var bufSizes = []int{1, 2, 3, 4, 5, 6, 7, 8, 100, 101, 102, 103, 4000, 4001, 4002, 4003}

func readFunc(size int) appendFunc {
	return func(f Form, out []byte, s string) []byte {
		out = append(out, s...)
		r := f.Reader(bytes.NewBuffer(out))
		buf := make([]byte, size)
		result := []byte{}
		for n, err := 0, os.Error(nil); err == nil; {
			n, err = r.Read(buf)
			result = append(result, buf[:n]...)
		}
		return result
	}
}

func TestReader(t *testing.T) {
	for _, s := range bufSizes {
		name := fmt.Sprintf("TestReader%da", s)
		runAppendTests(t, name, NFKC, readFunc(s), appendTests)
		name = fmt.Sprintf("TestReader%db", s)
		runAppendTests(t, name, NFKC, readFunc(s), ioTests)
	}
}

func writeFunc(size int) appendFunc {
	return func(f Form, out []byte, s string) []byte {
		in := append(out, s...)
		result := new(bytes.Buffer)
		w := f.Writer(result)
		buf := make([]byte, size)
		for n := 0; len(in) > 0; in = in[n:] {
			n = copy(buf, in)
			_, _ = w.Write(buf[:n])
		}
		w.Close()
		return result.Bytes()
	}
}

func TestWriter(t *testing.T) {
	for _, s := range bufSizes {
		name := fmt.Sprintf("TestWriter%da", s)
		runAppendTests(t, name, NFKC, writeFunc(s), appendTests)
		name = fmt.Sprintf("TestWriter%db", s)
		runAppendTests(t, name, NFKC, writeFunc(s), ioTests)
	}
}
