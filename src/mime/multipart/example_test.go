// Copyright 2014 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

package multipart_test

import (
	"fmt"
	"io"
	"io/ioutil"
	"log"
	"mime"
	"mime/multipart"
	"net/mail"
	"strings"
)

func ExampleNewReader() {
	msg := &mail.Message{
		Header: map[string][]string{
			"Content-Type": {"multipart/mixed; boundary=foo"},
		},
		Body: strings.NewReader(
			"--foo\r\nFoo: one\r\n\r\nA section\r\n" +
				"--foo\r\nFoo: two\r\n\r\nAnd another\r\n" +
				"--foo--\r\n"),
	}
	mediaType, params, err := mime.ParseMediaType(msg.Header.Get("Content-Type"))
	if err != nil {
		log.Fatal(err)
	}
	if strings.HasPrefix(mediaType, "multipart/") {
		mr := multipart.NewReader(msg.Body, params["boundary"])
		for {
			p, err := mr.NextPart()
			if err == io.EOF {
				return
			}
			if err != nil {
				log.Fatal(err)
			}
			slurp, err := ioutil.ReadAll(p)
			if err != nil {
				log.Fatal(err)
			}
			fmt.Printf("Part %q: %q\n", p.Header.Get("Foo"), slurp)
		}
	}

	// Output:
	// Part "one": "A section"
	// Part "two": "And another"
}
