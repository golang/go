// Copyright 2012 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

package zlib_test

import (
	"bytes"
	"compress/zlib"
	"fmt"
	"io"
	"os"
)

func ExampleNewWriter() {
	var b bytes.Buffer

	w := zlib.NewWriter(&b)
	w.Write([]byte("hello, world\n"))
	w.Close()
	fmt.Println(b.Bytes())
	// Output: [120 156 0 13 0 242 255 104 101 108 108 111 44 32 119 111 114 108 100 10 3 0 33 231 4 147]
}

func ExampleNewReader() {
	buff := []byte{120, 156, 0, 13, 0, 242, 255, 104, 101, 108, 108, 111,
		44, 32, 119, 111, 114, 108, 100, 10, 3, 0, 33, 231, 4, 147}
	b := bytes.NewReader(buff)

	r, err := zlib.NewReader(b)
	if err != nil {
		panic(err)
	}
	io.Copy(os.Stdout, r)
	r.Close()
	// Output:
	// hello, world
}
