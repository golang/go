// Copyright 2010 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

package smtp

import (
	"bufio"
	"bytes"
	"io"
	"net/textproto"
	"os"
	"strings"
	"testing"
)

type authTest struct {
	auth       Auth
	challenges []string
	name       string
	responses  []string
}

var authTests = []authTest{
	{PlainAuth("", "user", "pass", "testserver"), []string{}, "PLAIN", []string{"\x00user\x00pass"}},
	{PlainAuth("foo", "bar", "baz", "testserver"), []string{}, "PLAIN", []string{"foo\x00bar\x00baz"}},
}

func TestAuth(t *testing.T) {
testLoop:
	for i, test := range authTests {
		name, resp, err := test.auth.Start(&ServerInfo{"testserver", true, nil})
		if name != test.name {
			t.Errorf("#%d got name %s, expected %s", i, name, test.name)
		}
		if !bytes.Equal(resp, []byte(test.responses[0])) {
			t.Errorf("#%d got response %s, expected %s", i, resp, test.responses[0])
		}
		if err != nil {
			t.Errorf("#%d error: %s", i, err.String())
		}
		for j := range test.challenges {
			challenge := []byte(test.challenges[j])
			expected := []byte(test.responses[j+1])
			resp, err := test.auth.Next(challenge, true)
			if err != nil {
				t.Errorf("#%d error: %s", i, err.String())
				continue testLoop
			}
			if !bytes.Equal(resp, expected) {
				t.Errorf("#%d got %s, expected %s", i, resp, expected)
				continue testLoop
			}
		}
	}
}

type faker struct {
	io.ReadWriter
}

func (f faker) Close() os.Error {
	return nil
}

func TestBasic(t *testing.T) {
	basicServer = strings.Join(strings.Split(basicServer, "\n", -1), "\r\n")
	basicClient = strings.Join(strings.Split(basicClient, "\n", -1), "\r\n")

	var cmdbuf bytes.Buffer
	bcmdbuf := bufio.NewWriter(&cmdbuf)
	var fake faker
	fake.ReadWriter = bufio.NewReadWriter(bufio.NewReader(strings.NewReader(basicServer)), bcmdbuf)
	c := &Client{Text: textproto.NewConn(fake)}

	if err := c.helo(); err != nil {
		t.Fatalf("HELO failed: %s", err.String())
	}
	if err := c.ehlo(); err == nil {
		t.Fatalf("Expected first EHLO to fail")
	}
	if err := c.ehlo(); err != nil {
		t.Fatalf("Second EHLO failed: %s", err.String())
	}

	if ok, args := c.Extension("aUtH"); !ok || args != "LOGIN PLAIN" {
		t.Fatalf("Expected AUTH supported")
	}
	if ok, _ := c.Extension("DSN"); ok {
		t.Fatalf("Shouldn't support DSN")
	}

	if err := c.Mail("user@gmail.com"); err == nil {
		t.Fatalf("MAIL should require authentication")
	}

	if err := c.Verify("user1@gmail.com"); err == nil {
		t.Fatalf("First VRFY: expected no verification")
	}
	if err := c.Verify("user2@gmail.com"); err != nil {
		t.Fatalf("Second VRFY: expected verification, got %s", err)
	}

	// fake TLS so authentication won't complain
	c.tls = true
	c.serverName = "smtp.google.com"
	if err := c.Auth(PlainAuth("", "user", "pass", "smtp.google.com")); err != nil {
		t.Fatalf("AUTH failed: %s", err.String())
	}

	if err := c.Mail("user@gmail.com"); err != nil {
		t.Fatalf("MAIL failed: %s", err.String())
	}
	if err := c.Rcpt("golang-nuts@googlegroups.com"); err != nil {
		t.Fatalf("RCPT failed: %s", err.String())
	}
	msg := `From: user@gmail.com
To: golang-nuts@googlegroups.com
Subject: Hooray for Go

Line 1
.Leading dot line .
Goodbye.`
	w, err := c.Data()
	if err != nil {
		t.Fatalf("DATA failed: %s", err.String())
	}
	if _, err := w.Write([]byte(msg)); err != nil {
		t.Fatalf("Data write failed: %s", err.String())
	}
	if err := w.Close(); err != nil {
		t.Fatalf("Bad data response: %s", err.String())
	}

	if err := c.Quit(); err != nil {
		t.Fatalf("QUIT failed: %s", err.String())
	}

	bcmdbuf.Flush()
	actualcmds := cmdbuf.String()
	if basicClient != actualcmds {
		t.Fatalf("Got:\n%s\nExpected:\n%s", actualcmds, basicClient)
	}
}

var basicServer = `250 mx.google.com at your service
502 Unrecognized command.
250-mx.google.com at your service
250-SIZE 35651584
250-AUTH LOGIN PLAIN
250 8BITMIME
530 Authentication required
252 Send some mail, I'll try my best
250 User is valid
235 Accepted
250 Sender OK
250 Receiver OK
354 Go ahead
250 Data OK
221 OK
`

var basicClient = `HELO localhost
EHLO localhost
EHLO localhost
MAIL FROM:<user@gmail.com> BODY=8BITMIME
VRFY user1@gmail.com
VRFY user2@gmail.com
AUTH PLAIN AHVzZXIAcGFzcw==
MAIL FROM:<user@gmail.com> BODY=8BITMIME
RCPT TO:<golang-nuts@googlegroups.com>
DATA
From: user@gmail.com
To: golang-nuts@googlegroups.com
Subject: Hooray for Go

Line 1
..Leading dot line .
Goodbye.
.
QUIT
`
