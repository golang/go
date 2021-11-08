// Copyright 2015 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

package mail_test

import (
	"fmt"
	"io"
	"log"
	"net/mail"
	"strings"
)

func ExampleParseAddressList() {
	const list = "Alice <alice@example.com>, Bob <bob@example.com>, Eve <eve@example.com>"
	emails, err := mail.ParseAddressList(list)
	if err != nil {
		log.Fatal(err)
	}

	for _, v := range emails {
		fmt.Println(v.Name, v.Address)
	}

	// Output:
	// Alice alice@example.com
	// Bob bob@example.com
	// Eve eve@example.com
}

func ExampleParseAddress() {
	e, err := mail.ParseAddress("Alice <alice@example.com>")
	if err != nil {
		log.Fatal(err)
	}

	fmt.Println(e.Name, e.Address)

	// Output:
	// Alice alice@example.com
}

func ExampleReadMessage() {
	msg := `Date: Mon, 23 Jun 2015 11:40:36 -0400
From: Gopher <from@example.com>
To: Another Gopher <to@example.com>
Subject: Gophers at Gophercon

Message body
`

	r := strings.NewReader(msg)
	m, err := mail.ReadMessage(r)
	if err != nil {
		log.Fatal(err)
	}

	header := m.Header
	fmt.Println("Date:", header.Get("Date"))
	fmt.Println("From:", header.Get("From"))
	fmt.Println("To:", header.Get("To"))
	fmt.Println("Subject:", header.Get("Subject"))

	body, err := io.ReadAll(m.Body)
	if err != nil {
		log.Fatal(err)
	}
	fmt.Printf("%s", body)

	// Output:
	// Date: Mon, 23 Jun 2015 11:40:36 -0400
	// From: Gopher <from@example.com>
	// To: Another Gopher <to@example.com>
	// Subject: Gophers at Gophercon
	// Message body
}
