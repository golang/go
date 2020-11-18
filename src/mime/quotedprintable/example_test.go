// Copyright 2016 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

package quotedprintable_test

import (
	"fmt"
	"io"
	"mime/quotedprintable"
	"os"
	"strings"
)

func ExampleNewReader() {
	for _, s := range []string{
		`=48=65=6C=6C=6F=2C=20=47=6F=70=68=65=72=73=21`,
		`invalid escape: <b style="font-size: 200%">hello</b>`,
		"Hello, Gophers! This symbol will be unescaped: =3D and this will be written in =\r\none line.",
	} {
		b, err := io.ReadAll(quotedprintable.NewReader(strings.NewReader(s)))
		fmt.Printf("%s %v\n", b, err)
	}
	// Output:
	// Hello, Gophers! <nil>
	// invalid escape: <b style="font-size: 200%">hello</b> <nil>
	// Hello, Gophers! This symbol will be unescaped: = and this will be written in one line. <nil>
}

func ExampleNewWriter() {
	w := quotedprintable.NewWriter(os.Stdout)
	w.Write([]byte("These symbols will be escaped: = \t"))
	w.Close()

	// Output:
	// These symbols will be escaped: =3D =09
}
