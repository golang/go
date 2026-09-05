// Copyright 2026 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

package textproto_test

import (
	"bufio"
	"fmt"
	"net/textproto"
	"strings"
)

func ExampleReader_ReadMIMEHeader() {
	input := "content-type: text/plain\r\n" +
		"x-tag: one\r\n" +
		"x-tag: two\r\n" +
		"long-key: Even\r\n" +
		"\tLonger Value\r\n\r\n"
	r := textproto.NewReader(bufio.NewReader(strings.NewReader(input)))

	header, err := r.ReadMIMEHeader()
	if err != nil {
		panic(err)
	}
	fmt.Println(header["Content-Type"][0])
	fmt.Println(header["X-Tag"])
	fmt.Println(header["Long-Key"])

	// Output:
	// text/plain
	// [one two]
	// [Even Longer Value]
}
