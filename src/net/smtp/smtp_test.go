// Copyright 2010 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

package smtp

import (
	"bufio"
	"bytes"
	"crypto/tls"
	"crypto/x509"
	"fmt"
	"internal/testenv"
	"io"
	"net"
	"net/textproto"
	"runtime"
	"strings"
	"testing"
	"time"
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
	{CRAMMD5Auth("user", "pass"), []string{"<123456.1322876914@testserver>"}, "CRAM-MD5", []string{"", "user 287eb355114cf5c471c26a875f1ca4ae"}},
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
			t.Errorf("#%d error: %s", i, err)
		}
		for j := range test.challenges {
			challenge := []byte(test.challenges[j])
			expected := []byte(test.responses[j+1])
			resp, err := test.auth.Next(challenge, true)
			if err != nil {
				t.Errorf("#%d error: %s", i, err)
				continue testLoop
			}
			if !bytes.Equal(resp, expected) {
				t.Errorf("#%d got %s, expected %s", i, resp, expected)
				continue testLoop
			}
		}
	}
}

func TestAuthPlain(t *testing.T) {

	tests := []struct {
		authName string
		server   *ServerInfo
		err      string
	}{
		{
			authName: "servername",
			server:   &ServerInfo{Name: "servername", TLS: true},
		},
		{
			// OK to use PlainAuth on localhost without TLS
			authName: "localhost",
			server:   &ServerInfo{Name: "localhost", TLS: false},
		},
		{
			// NOT OK on non-localhost, even if server says PLAIN is OK.
			// (We don't know that the server is the real server.)
			authName: "servername",
			server:   &ServerInfo{Name: "servername", Auth: []string{"PLAIN"}},
			err:      "unencrypted connection",
		},
		{
			authName: "servername",
			server:   &ServerInfo{Name: "servername", Auth: []string{"CRAM-MD5"}},
			err:      "unencrypted connection",
		},
		{
			authName: "servername",
			server:   &ServerInfo{Name: "attacker", TLS: true},
			err:      "wrong host name",
		},
	}
	for i, tt := range tests {
		auth := PlainAuth("foo", "bar", "baz", tt.authName)
		_, _, err := auth.Start(tt.server)
		got := ""
		if err != nil {
			got = err.Error()
		}
		if got != tt.err {
			t.Errorf("%d. got error = %q; want %q", i, got, tt.err)
		}
	}
}

// Issue 17794: don't send a trailing space on AUTH command when there's no password.
func TestClientAuthTrimSpace(t *testing.T) {
	server := "220 hello world\r\n" +
		"200 some more"
	var wrote strings.Builder
	var fake faker
	fake.ReadWriter = struct {
		io.Reader
		io.Writer
	}{
		strings.NewReader(server),
		&wrote,
	}
	c, err := NewClient(fake, "fake.host")
	if err != nil {
		t.Fatalf("NewClient: %v", err)
	}
	c.tls = true
	c.didHello = true
	c.Auth(toServerEmptyAuth{})
	c.Close()
	if got, want := wrote.String(), "AUTH FOOAUTH\r\n*\r\nQUIT\r\n"; got != want {
		t.Errorf("wrote %q; want %q", got, want)
	}
}

// toServerEmptyAuth is an implementation of Auth that only implements
// the Start method, and returns "FOOAUTH", nil, nil. Notably, it returns
// zero bytes for "toServer" so we can test that we don't send spaces at
// the end of the line. See TestClientAuthTrimSpace.
type toServerEmptyAuth struct{}

func (toServerEmptyAuth) Start(server *ServerInfo) (proto string, toServer []byte, err error) {
	return "FOOAUTH", nil, nil
}

func (toServerEmptyAuth) Next(fromServer []byte, more bool) (toServer []byte, err error) {
	panic("unexpected call")
}

type faker struct {
	io.ReadWriter
}

func (f faker) Close() error                     { return nil }
func (f faker) LocalAddr() net.Addr              { return nil }
func (f faker) RemoteAddr() net.Addr             { return nil }
func (f faker) SetDeadline(time.Time) error      { return nil }
func (f faker) SetReadDeadline(time.Time) error  { return nil }
func (f faker) SetWriteDeadline(time.Time) error { return nil }

func TestBasic(t *testing.T) {
	server := strings.Join(strings.Split(basicServer, "\n"), "\r\n")
	client := strings.Join(strings.Split(basicClient, "\n"), "\r\n")

	var cmdbuf strings.Builder
	bcmdbuf := bufio.NewWriter(&cmdbuf)
	var fake faker
	fake.ReadWriter = bufio.NewReadWriter(bufio.NewReader(strings.NewReader(server)), bcmdbuf)
	c := &Client{Text: textproto.NewConn(fake), localName: "localhost"}

	if err := c.helo(); err != nil {
		t.Fatalf("HELO failed: %s", err)
	}
	if err := c.ehlo(); err == nil {
		t.Fatalf("Expected first EHLO to fail")
	}
	if err := c.ehlo(); err != nil {
		t.Fatalf("Second EHLO failed: %s", err)
	}

	c.didHello = true
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
	if err := c.Verify("user2@gmail.com>\r\nDATA\r\nAnother injected message body\r\n.\r\nQUIT\r\n"); err == nil {
		t.Fatalf("VRFY should have failed due to a message injection attempt")
	}
	if err := c.Verify("user2@gmail.com"); err != nil {
		t.Fatalf("Second VRFY: expected verification, got %s", err)
	}

	// fake TLS so authentication won't complain
	c.tls = true
	c.serverName = "smtp.google.com"
	if err := c.Auth(PlainAuth("", "user", "pass", "smtp.google.com")); err != nil {
		t.Fatalf("AUTH failed: %s", err)
	}

	if err := c.Rcpt("golang-nuts@googlegroups.com>\r\nDATA\r\nInjected message body\r\n.\r\nQUIT\r\n"); err == nil {
		t.Fatalf("RCPT should have failed due to a message injection attempt")
	}
	if err := c.Mail("user@gmail.com>\r\nDATA\r\nAnother injected message body\r\n.\r\nQUIT\r\n"); err == nil {
		t.Fatalf("MAIL should have failed due to a message injection attempt")
	}
	if err := c.Mail("user@gmail.com"); err != nil {
		t.Fatalf("MAIL failed: %s", err)
	}
	if err := c.Rcpt("golang-nuts@googlegroups.com"); err != nil {
		t.Fatalf("RCPT failed: %s", err)
	}
	msg := `From: user@gmail.com
To: golang-nuts@googlegroups.com
Subject: Hooray for Go

Line 1
.Leading dot line .
Goodbye.`
	w, err := c.Data()
	if err != nil {
		t.Fatalf("DATA failed: %s", err)
	}
	if _, err := w.Write([]byte(msg)); err != nil {
		t.Fatalf("Data write failed: %s", err)
	}
	if err := w.Close(); err != nil {
		t.Fatalf("Bad data response: %s", err)
	}

	if err := c.Quit(); err != nil {
		t.Fatalf("QUIT failed: %s", err)
	}

	bcmdbuf.Flush()
	actualcmds := cmdbuf.String()
	if client != actualcmds {
		t.Fatalf("Got:\n%s\nExpected:\n%s", actualcmds, client)
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

func TestHELOFailed(t *testing.T) {
	serverLines := `502 EH?
502 EH?
221 OK
`
	clientLines := `EHLO localhost
HELO localhost
QUIT
`

	server := strings.Join(strings.Split(serverLines, "\n"), "\r\n")
	client := strings.Join(strings.Split(clientLines, "\n"), "\r\n")
	var cmdbuf strings.Builder
	bcmdbuf := bufio.NewWriter(&cmdbuf)
	var fake faker
	fake.ReadWriter = bufio.NewReadWriter(bufio.NewReader(strings.NewReader(server)), bcmdbuf)
	c := &Client{Text: textproto.NewConn(fake), localName: "localhost"}

	if err := c.Hello("localhost"); err == nil {
		t.Fatal("expected EHLO to fail")
	}
	if err := c.Quit(); err != nil {
		t.Errorf("QUIT failed: %s", err)
	}
	bcmdbuf.Flush()
	actual := cmdbuf.String()
	if client != actual {
		t.Errorf("Got:\n%s\nWant:\n%s", actual, client)
	}
}

func TestExtensions(t *testing.T) {
	fake := func(server string) (c *Client, bcmdbuf *bufio.Writer, cmdbuf *strings.Builder) {
		server = strings.Join(strings.Split(server, "\n"), "\r\n")

		cmdbuf = &strings.Builder{}
		bcmdbuf = bufio.NewWriter(cmdbuf)
		var fake faker
		fake.ReadWriter = bufio.NewReadWriter(bufio.NewReader(strings.NewReader(server)), bcmdbuf)
		c = &Client{Text: textproto.NewConn(fake), localName: "localhost"}

		return c, bcmdbuf, cmdbuf
	}

	t.Run("helo", func(t *testing.T) {
		const (
			basicServer = `250 mx.google.com at your service
250 Sender OK
221 Goodbye
`

			basicClient = `HELO localhost
MAIL FROM:<user@gmail.com>
QUIT
`
		)

		c, bcmdbuf, cmdbuf := fake(basicServer)

		if err := c.helo(); err != nil {
			t.Fatalf("HELO failed: %s", err)
		}
		c.didHello = true
		if err := c.Mail("user@gmail.com"); err != nil {
			t.Fatalf("MAIL FROM failed: %s", err)
		}
		if err := c.Quit(); err != nil {
			t.Fatalf("QUIT failed: %s", err)
		}

		bcmdbuf.Flush()
		actualcmds := cmdbuf.String()
		client := strings.Join(strings.Split(basicClient, "\n"), "\r\n")
		if client != actualcmds {
			t.Fatalf("Got:\n%s\nExpected:\n%s", actualcmds, client)
		}
	})

	t.Run("ehlo", func(t *testing.T) {
		const (
			basicServer = `250-mx.google.com at your service
250 SIZE 35651584
250 Sender OK
221 Goodbye
`

			basicClient = `EHLO localhost
MAIL FROM:<user@gmail.com>
QUIT
`
		)

		c, bcmdbuf, cmdbuf := fake(basicServer)

		if err := c.Hello("localhost"); err != nil {
			t.Fatalf("EHLO failed: %s", err)
		}
		if ok, _ := c.Extension("8BITMIME"); ok {
			t.Fatalf("Shouldn't support 8BITMIME")
		}
		if ok, _ := c.Extension("SMTPUTF8"); ok {
			t.Fatalf("Shouldn't support SMTPUTF8")
		}
		if err := c.Mail("user@gmail.com"); err != nil {
			t.Fatalf("MAIL FROM failed: %s", err)
		}
		if err := c.Quit(); err != nil {
			t.Fatalf("QUIT failed: %s", err)
		}

		bcmdbuf.Flush()
		actualcmds := cmdbuf.String()
		client := strings.Join(strings.Split(basicClient, "\n"), "\r\n")
		if client != actualcmds {
			t.Fatalf("Got:\n%s\nExpected:\n%s", actualcmds, client)
		}
	})

	t.Run("ehlo 8bitmime", func(t *testing.T) {
		const (
			basicServer = `250-mx.google.com at your service
250-SIZE 35651584
250 8BITMIME
250 Sender OK
221 Goodbye
`

			basicClient = `EHLO localhost
MAIL FROM:<user@gmail.com> BODY=8BITMIME
QUIT
`
		)

		c, bcmdbuf, cmdbuf := fake(basicServer)

		if err := c.Hello("localhost"); err != nil {
			t.Fatalf("EHLO failed: %s", err)
		}
		if ok, _ := c.Extension("8BITMIME"); !ok {
			t.Fatalf("Should support 8BITMIME")
		}
		if ok, _ := c.Extension("SMTPUTF8"); ok {
			t.Fatalf("Shouldn't support SMTPUTF8")
		}
		if err := c.Mail("user@gmail.com"); err != nil {
			t.Fatalf("MAIL FROM failed: %s", err)
		}
		if err := c.Quit(); err != nil {
			t.Fatalf("QUIT failed: %s", err)
		}

		bcmdbuf.Flush()
		actualcmds := cmdbuf.String()
		client := strings.Join(strings.Split(basicClient, "\n"), "\r\n")
		if client != actualcmds {
			t.Fatalf("Got:\n%s\nExpected:\n%s", actualcmds, client)
		}
	})

	t.Run("ehlo smtputf8", func(t *testing.T) {
		const (
			basicServer = `250-mx.google.com at your service
250-SIZE 35651584
250 SMTPUTF8
250 Sender OK
221 Goodbye
`

			basicClient = `EHLO localhost
MAIL FROM:<user+📧@gmail.com> SMTPUTF8
QUIT
`
		)

		c, bcmdbuf, cmdbuf := fake(basicServer)

		if err := c.Hello("localhost"); err != nil {
			t.Fatalf("EHLO failed: %s", err)
		}
		if ok, _ := c.Extension("8BITMIME"); ok {
			t.Fatalf("Shouldn't support 8BITMIME")
		}
		if ok, _ := c.Extension("SMTPUTF8"); !ok {
			t.Fatalf("Should support SMTPUTF8")
		}
		if err := c.Mail("user+📧@gmail.com"); err != nil {
			t.Fatalf("MAIL FROM failed: %s", err)
		}
		if err := c.Quit(); err != nil {
			t.Fatalf("QUIT failed: %s", err)
		}

		bcmdbuf.Flush()
		actualcmds := cmdbuf.String()
		client := strings.Join(strings.Split(basicClient, "\n"), "\r\n")
		if client != actualcmds {
			t.Fatalf("Got:\n%s\nExpected:\n%s", actualcmds, client)
		}
	})

	t.Run("ehlo 8bitmime smtputf8", func(t *testing.T) {
		const (
			basicServer = `250-mx.google.com at your service
250-SIZE 35651584
250-8BITMIME
250 SMTPUTF8
250 Sender OK
221 Goodbye
	`

			basicClient = `EHLO localhost
MAIL FROM:<user+📧@gmail.com> BODY=8BITMIME SMTPUTF8
QUIT
`
		)

		c, bcmdbuf, cmdbuf := fake(basicServer)

		if err := c.Hello("localhost"); err != nil {
			t.Fatalf("EHLO failed: %s", err)
		}
		c.didHello = true
		if ok, _ := c.Extension("8BITMIME"); !ok {
			t.Fatalf("Should support 8BITMIME")
		}
		if ok, _ := c.Extension("SMTPUTF8"); !ok {
			t.Fatalf("Should support SMTPUTF8")
		}
		if err := c.Mail("user+📧@gmail.com"); err != nil {
			t.Fatalf("MAIL FROM failed: %s", err)
		}
		if err := c.Quit(); err != nil {
			t.Fatalf("QUIT failed: %s", err)
		}

		bcmdbuf.Flush()
		actualcmds := cmdbuf.String()
		client := strings.Join(strings.Split(basicClient, "\n"), "\r\n")
		if client != actualcmds {
			t.Fatalf("Got:\n%s\nExpected:\n%s", actualcmds, client)
		}
	})
}

func TestNewClient(t *testing.T) {
	server := strings.Join(strings.Split(newClientServer, "\n"), "\r\n")
	client := strings.Join(strings.Split(newClientClient, "\n"), "\r\n")

	var cmdbuf strings.Builder
	bcmdbuf := bufio.NewWriter(&cmdbuf)
	out := func() string {
		bcmdbuf.Flush()
		return cmdbuf.String()
	}
	var fake faker
	fake.ReadWriter = bufio.NewReadWriter(bufio.NewReader(strings.NewReader(server)), bcmdbuf)
	c, err := NewClient(fake, "fake.host")
	if err != nil {
		t.Fatalf("NewClient: %v\n(after %v)", err, out())
	}
	defer c.Close()
	if ok, args := c.Extension("aUtH"); !ok || args != "LOGIN PLAIN" {
		t.Fatalf("Expected AUTH supported")
	}
	if ok, _ := c.Extension("DSN"); ok {
		t.Fatalf("Shouldn't support DSN")
	}
	if err := c.Quit(); err != nil {
		t.Fatalf("QUIT failed: %s", err)
	}

	actualcmds := out()
	if client != actualcmds {
		t.Fatalf("Got:\n%s\nExpected:\n%s", actualcmds, client)
	}
}

var newClientServer = `220 hello world
250-mx.google.com at your service
250-SIZE 35651584
250-AUTH LOGIN PLAIN
250 8BITMIME
221 OK
`

var newClientClient = `EHLO localhost
QUIT
`

func TestNewClient2(t *testing.T) {
	server := strings.Join(strings.Split(newClient2Server, "\n"), "\r\n")
	client := strings.Join(strings.Split(newClient2Client, "\n"), "\r\n")

	var cmdbuf strings.Builder
	bcmdbuf := bufio.NewWriter(&cmdbuf)
	var fake faker
	fake.ReadWriter = bufio.NewReadWriter(bufio.NewReader(strings.NewReader(server)), bcmdbuf)
	c, err := NewClient(fake, "fake.host")
	if err != nil {
		t.Fatalf("NewClient: %v", err)
	}
	defer c.Close()
	if ok, _ := c.Extension("DSN"); ok {
		t.Fatalf("Shouldn't support DSN")
	}
	if err := c.Quit(); err != nil {
		t.Fatalf("QUIT failed: %s", err)
	}

	bcmdbuf.Flush()
	actualcmds := cmdbuf.String()
	if client != actualcmds {
		t.Fatalf("Got:\n%s\nExpected:\n%s", actualcmds, client)
	}
}

var newClient2Server = `220 hello world
502 EH?
250-mx.google.com at your service
250-SIZE 35651584
250-AUTH LOGIN PLAIN
250 8BITMIME
221 OK
`

var newClient2Client = `EHLO localhost
HELO localhost
QUIT
`

func TestNewClientWithTLS(t *testing.T) {
	cert, err := tls.X509KeyPair(localhostCert, localhostKey)
	if err != nil {
		t.Fatalf("loadcert: %v", err)
	}

	config := tls.Config{Certificates: []tls.Certificate{cert}}

	ln, err := tls.Listen("tcp", "127.0.0.1:0", &config)
	if err != nil {
		ln, err = tls.Listen("tcp", "[::1]:0", &config)
		if err != nil {
			t.Fatalf("server: listen: %v", err)
		}
	}

	go func() {
		conn, err := ln.Accept()
		if err != nil {
			t.Errorf("server: accept: %v", err)
			return
		}
		defer conn.Close()

		_, err = conn.Write([]byte("220 SIGNS\r\n"))
		if err != nil {
			t.Errorf("server: write: %v", err)
			return
		}
	}()

	config.InsecureSkipVerify = true
	conn, err := tls.Dial("tcp", ln.Addr().String(), &config)
	if err != nil {
		t.Fatalf("client: dial: %v", err)
	}
	defer conn.Close()

	client, err := NewClient(conn, ln.Addr().String())
	if err != nil {
		t.Fatalf("smtp: newclient: %v", err)
	}
	if !client.tls {
		t.Errorf("client.tls Got: %t Expected: %t", client.tls, true)
	}
}

func TestHello(t *testing.T) {

	if len(helloServer) != len(helloClient) {
		t.Fatalf("Hello server and client size mismatch")
	}

	for i := 0; i < len(helloServer); i++ {
		server := strings.Join(strings.Split(baseHelloServer+helloServer[i], "\n"), "\r\n")
		client := strings.Join(strings.Split(baseHelloClient+helloClient[i], "\n"), "\r\n")
		var cmdbuf strings.Builder
		bcmdbuf := bufio.NewWriter(&cmdbuf)
		var fake faker
		fake.ReadWriter = bufio.NewReadWriter(bufio.NewReader(strings.NewReader(server)), bcmdbuf)
		c, err := NewClient(fake, "fake.host")
		if err != nil {
			t.Fatalf("NewClient: %v", err)
		}
		defer c.Close()
		c.localName = "customhost"
		err = nil

		switch i {
		case 0:
			err = c.Hello("hostinjection>\n\rDATA\r\nInjected message body\r\n.\r\nQUIT\r\n")
			if err == nil {
				t.Errorf("Expected Hello to be rejected due to a message injection attempt")
			}
			err = c.Hello("customhost")
		case 1:
			err = c.StartTLS(nil)
			if err.Error() == "502 Not implemented" {
				err = nil
			}
		case 2:
			err = c.Verify("test@example.com")
		case 3:
			c.tls = true
			c.serverName = "smtp.google.com"
			err = c.Auth(PlainAuth("", "user", "pass", "smtp.google.com"))
		case 4:
			err = c.Mail("test@example.com")
		case 5:
			ok, _ := c.Extension("feature")
			if ok {
				t.Errorf("Expected FEATURE not to be supported")
			}
		case 6:
			err = c.Reset()
		case 7:
			err = c.Quit()
		case 8:
			err = c.Verify("test@example.com")
			if err != nil {
				err = c.Hello("customhost")
				if err != nil {
					t.Errorf("Want error, got none")
				}
			}
		case 9:
			err = c.Noop()
		default:
			t.Fatalf("Unhandled command")
		}

		if err != nil {
			t.Errorf("Command %d failed: %v", i, err)
		}

		bcmdbuf.Flush()
		actualcmds := cmdbuf.String()
		if client != actualcmds {
			t.Errorf("Got:\n%s\nExpected:\n%s", actualcmds, client)
		}
	}
}

var baseHelloServer = `220 hello world
502 EH?
250-mx.google.com at your service
250 FEATURE
`

var helloServer = []string{
	"",
	"502 Not implemented\n",
	"250 User is valid\n",
	"235 Accepted\n",
	"250 Sender ok\n",
	"",
	"250 Reset ok\n",
	"221 Goodbye\n",
	"250 Sender ok\n",
	"250 ok\n",
}

var baseHelloClient = `EHLO customhost
HELO customhost
`

var helloClient = []string{
	"",
	"STARTTLS\n",
	"VRFY test@example.com\n",
	"AUTH PLAIN AHVzZXIAcGFzcw==\n",
	"MAIL FROM:<test@example.com>\n",
	"",
	"RSET\n",
	"QUIT\n",
	"VRFY test@example.com\n",
	"NOOP\n",
}

func TestSendMail(t *testing.T) {
	server := strings.Join(strings.Split(sendMailServer, "\n"), "\r\n")
	client := strings.Join(strings.Split(sendMailClient, "\n"), "\r\n")
	var cmdbuf strings.Builder
	bcmdbuf := bufio.NewWriter(&cmdbuf)
	l, err := net.Listen("tcp", "127.0.0.1:0")
	if err != nil {
		t.Fatalf("Unable to create listener: %v", err)
	}
	defer l.Close()

	// prevent data race on bcmdbuf
	var done = make(chan struct{})
	go func(data []string) {

		defer close(done)

		conn, err := l.Accept()
		if err != nil {
			t.Errorf("Accept error: %v", err)
			return
		}
		defer conn.Close()

		tc := textproto.NewConn(conn)
		for i := 0; i < len(data) && data[i] != ""; i++ {
			tc.PrintfLine("%s", data[i])
			for len(data[i]) >= 4 && data[i][3] == '-' {
				i++
				tc.PrintfLine("%s", data[i])
			}
			if data[i] == "221 Goodbye" {
				return
			}
			read := false
			for !read || data[i] == "354 Go ahead" {
				msg, err := tc.ReadLine()
				bcmdbuf.Write([]byte(msg + "\r\n"))
				read = true
				if err != nil {
					t.Errorf("Read error: %v", err)
					return
				}
				if data[i] == "354 Go ahead" && msg == "." {
					break
				}
			}
		}
	}(strings.Split(server, "\r\n"))

	err = SendMail(l.Addr().String(), nil, "test@example.com", []string{"other@example.com>\n\rDATA\r\nInjected message body\r\n.\r\nQUIT\r\n"}, []byte(strings.Replace(`From: test@example.com
To: other@example.com
Subject: SendMail test

SendMail is working for me.
`, "\n", "\r\n", -1)))
	if err == nil {
		t.Errorf("Expected SendMail to be rejected due to a message injection attempt")
	}

	err = SendMail(l.Addr().String(), nil, "test@example.com", []string{"other@example.com"}, []byte(strings.Replace(`From: test@example.com
To: other@example.com
Subject: SendMail test

SendMail is working for me.
`, "\n", "\r\n", -1)))

	if err != nil {
		t.Errorf("%v", err)
	}

	<-done
	bcmdbuf.Flush()
	actualcmds := cmdbuf.String()
	if client != actualcmds {
		t.Errorf("Got:\n%s\nExpected:\n%s", actualcmds, client)
	}
}

var sendMailServer = `220 hello world
502 EH?
250 mx.google.com at your service
250 Sender ok
250 Receiver ok
354 Go ahead
250 Data ok
221 Goodbye
`

var sendMailClient = `EHLO localhost
HELO localhost
MAIL FROM:<test@example.com>
RCPT TO:<other@example.com>
DATA
From: test@example.com
To: other@example.com
Subject: SendMail test

SendMail is working for me.
.
QUIT
`

func TestSendMailWithAuth(t *testing.T) {
	l, err := net.Listen("tcp", "127.0.0.1:0")
	if err != nil {
		t.Fatalf("Unable to create listener: %v", err)
	}
	defer l.Close()

	errCh := make(chan error)
	go func() {
		defer close(errCh)
		conn, err := l.Accept()
		if err != nil {
			errCh <- fmt.Errorf("Accept: %v", err)
			return
		}
		defer conn.Close()

		tc := textproto.NewConn(conn)
		tc.PrintfLine("220 hello world")
		msg, err := tc.ReadLine()
		if err != nil {
			errCh <- fmt.Errorf("ReadLine error: %v", err)
			return
		}
		const wantMsg = "EHLO localhost"
		if msg != wantMsg {
			errCh <- fmt.Errorf("unexpected response %q; want %q", msg, wantMsg)
			return
		}
		err = tc.PrintfLine("250 mx.google.com at your service")
		if err != nil {
			errCh <- fmt.Errorf("PrintfLine: %v", err)
			return
		}
	}()

	err = SendMail(l.Addr().String(), PlainAuth("", "user", "pass", "smtp.google.com"), "test@example.com", []string{"other@example.com"}, []byte(strings.Replace(`From: test@example.com
To: other@example.com
Subject: SendMail test

SendMail is working for me.
`, "\n", "\r\n", -1)))
	if err == nil {
		t.Error("SendMail: Server doesn't support AUTH, expected to get an error, but got none ")
	}
	if err.Error() != "smtp: server doesn't support AUTH" {
		t.Errorf("Expected: smtp: server doesn't support AUTH, got: %s", err)
	}
	err = <-errCh
	if err != nil {
		t.Fatalf("server error: %v", err)
	}
}

func TestAuthFailed(t *testing.T) {
	server := strings.Join(strings.Split(authFailedServer, "\n"), "\r\n")
	client := strings.Join(strings.Split(authFailedClient, "\n"), "\r\n")
	var cmdbuf strings.Builder
	bcmdbuf := bufio.NewWriter(&cmdbuf)
	var fake faker
	fake.ReadWriter = bufio.NewReadWriter(bufio.NewReader(strings.NewReader(server)), bcmdbuf)
	c, err := NewClient(fake, "fake.host")
	if err != nil {
		t.Fatalf("NewClient: %v", err)
	}
	defer c.Close()

	c.tls = true
	c.serverName = "smtp.google.com"
	err = c.Auth(PlainAuth("", "user", "pass", "smtp.google.com"))

	if err == nil {
		t.Error("Auth: expected error; got none")
	} else if err.Error() != "535 Invalid credentials\nplease see www.example.com" {
		t.Errorf("Auth: got error: %v, want: %s", err, "535 Invalid credentials\nplease see www.example.com")
	}

	bcmdbuf.Flush()
	actualcmds := cmdbuf.String()
	if client != actualcmds {
		t.Errorf("Got:\n%s\nExpected:\n%s", actualcmds, client)
	}
}

var authFailedServer = `220 hello world
250-mx.google.com at your service
250 AUTH LOGIN PLAIN
535-Invalid credentials
535 please see www.example.com
221 Goodbye
`

var authFailedClient = `EHLO localhost
AUTH PLAIN AHVzZXIAcGFzcw==
*
QUIT
`

func TestTLSClient(t *testing.T) {
	if runtime.GOOS == "freebsd" || runtime.GOOS == "js" || runtime.GOOS == "wasip1" {
		testenv.SkipFlaky(t, 19229)
	}
	ln := newLocalListener(t)
	defer ln.Close()
	errc := make(chan error)
	go func() {
		errc <- sendMail(ln.Addr().String())
	}()
	conn, err := ln.Accept()
	if err != nil {
		t.Fatalf("failed to accept connection: %v", err)
	}
	defer conn.Close()
	if err := serverHandle(conn, t); err != nil {
		t.Fatalf("failed to handle connection: %v", err)
	}
	if err := <-errc; err != nil {
		t.Fatalf("client error: %v", err)
	}
}

func TestTLSConnState(t *testing.T) {
	ln := newLocalListener(t)
	defer ln.Close()
	clientDone := make(chan bool)
	serverDone := make(chan bool)
	go func() {
		defer close(serverDone)
		c, err := ln.Accept()
		if err != nil {
			t.Errorf("Server accept: %v", err)
			return
		}
		defer c.Close()
		if err := serverHandle(c, t); err != nil {
			t.Errorf("server error: %v", err)
		}
	}()
	go func() {
		defer close(clientDone)
		c, err := Dial(ln.Addr().String())
		if err != nil {
			t.Errorf("Client dial: %v", err)
			return
		}
		defer c.Quit()
		cfg := &tls.Config{ServerName: "example.com"}
		testHookStartTLS(cfg) // set the RootCAs
		if err := c.StartTLS(cfg); err != nil {
			t.Errorf("StartTLS: %v", err)
			return
		}
		cs, ok := c.TLSConnectionState()
		if !ok {
			t.Errorf("TLSConnectionState returned ok == false; want true")
			return
		}
		if cs.Version == 0 || !cs.HandshakeComplete {
			t.Errorf("ConnectionState = %#v; expect non-zero Version and HandshakeComplete", cs)
		}
	}()
	<-clientDone
	<-serverDone
}

func newLocalListener(t *testing.T) net.Listener {
	ln, err := net.Listen("tcp", "127.0.0.1:0")
	if err != nil {
		ln, err = net.Listen("tcp6", "[::1]:0")
	}
	if err != nil {
		t.Fatal(err)
	}
	return ln
}

type smtpSender struct {
	w io.Writer
}

func (s smtpSender) send(f string) {
	s.w.Write([]byte(f + "\r\n"))
}

// smtp server, finely tailored to deal with our own client only!
func serverHandle(c net.Conn, t *testing.T) error {
	send := smtpSender{c}.send
	send("220 127.0.0.1 ESMTP service ready")
	s := bufio.NewScanner(c)
	for s.Scan() {
		switch s.Text() {
		case "EHLO localhost":
			send("250-127.0.0.1 ESMTP offers a warm hug of welcome")
			send("250-STARTTLS")
			send("250 Ok")
		case "STARTTLS":
			send("220 Go ahead")
			keypair, err := tls.X509KeyPair(localhostCert, localhostKey)
			if err != nil {
				return err
			}
			config := &tls.Config{Certificates: []tls.Certificate{keypair}}
			c = tls.Server(c, config)
			defer c.Close()
			return serverHandleTLS(c, t)
		default:
			t.Fatalf("unrecognized command: %q", s.Text())
		}
	}
	return s.Err()
}

func serverHandleTLS(c net.Conn, t *testing.T) error {
	send := smtpSender{c}.send
	s := bufio.NewScanner(c)
	for s.Scan() {
		switch s.Text() {
		case "EHLO localhost":
			send("250 Ok")
		case "MAIL FROM:<joe1@example.com>":
			send("250 Ok")
		case "RCPT TO:<joe2@example.com>":
			send("250 Ok")
		case "DATA":
			send("354 send the mail data, end with .")
			send("250 Ok")
		case "Subject: test":
		case "":
		case "howdy!":
		case ".":
		case "QUIT":
			send("221 127.0.0.1 Service closing transmission channel")
			return nil
		default:
			t.Fatalf("unrecognized command during TLS: %q", s.Text())
		}
	}
	return s.Err()
}

func init() {
	testRootCAs := x509.NewCertPool()
	testRootCAs.AppendCertsFromPEM(localhostCert)
	testHookStartTLS = func(config *tls.Config) {
		config.RootCAs = testRootCAs
	}
}

func sendMail(hostPort string) error {
	from := "joe1@example.com"
	to := []string{"joe2@example.com"}
	return SendMail(hostPort, nil, from, to, []byte("Subject: test\n\nhowdy!"))
}

// localhostCert is a PEM-encoded TLS cert generated from src/crypto/tls:
//
//	go run generate_cert.go --rsa-bits 2048 --host 127.0.0.1,::1,example.com \
//		--ca --start-date "Jan 1 00:00:00 1970" --duration=1000000h
var localhostCert = []byte(`
-----BEGIN CERTIFICATE-----
MIIDFDCCAfygAwIBAgIRAPV4ktbcY/mn0oRRjnGAGJgwDQYJKoZIhvcNAQELBQAw
EjEQMA4GA1UEChMHQWNtZSBDbzAeFw0yNTAzMTgxOTI3NTRaFw0yNjAzMTgxOTI3
NTRaMBIxEDAOBgNVBAoTB0FjbWUgQ28wggEiMA0GCSqGSIb3DQEBAQUAA4IBDwAw
ggEKAoIBAQDbsEfk1bK7ozwZlcQM8rBUikC4gwnnw0J1PUlGDGu1Y84dKtulbdWj
yrh88D4fSdtmxFbXE7fhYUJTBmEHSUk9OLHh/Tr+nSC3SfH0I/9y6l9j9vVVYhYJ
C07Z1mZZKVb+gmbbB7LEavGMNaFHjvRJAwBX2TMDbXJceZ9jU/iihILkZbrbG40r
n1mctYVmcR3YqOzI/ynLje97FEvxtsg99OUjzzXyFMqfAl0J3Gc6tzvAER3N+ovK
nudsnMB5Y+InQHHmPeizG4mFyeBYesXNwX6cmI30c8KFiAlKHcsxjJsuoBZ3bSwv
vFdK2hnuCO05HEgCzAQKUlY6Q2F0xJblAgMBAAGjZTBjMA4GA1UdDwEB/wQEAwIF
oDATBgNVHSUEDDAKBggrBgEFBQcDATAMBgNVHRMBAf8EAjAAMC4GA1UdEQQnMCWC
C2V4YW1wbGUuY29thwR/AAABhxAAAAAAAAAAAAAAAAAAAAABMA0GCSqGSIb3DQEB
CwUAA4IBAQBnfO4lYRXR9AdMidpgdITqMEKJik8MvCkpQ+EKQLq3CIGXPt5lkHLs
ysbF9f3VxioKNYzkakJGVGyu51hqyhGqGQ4M7IpOBQkmY24IExWPVEk2wkIV+HTU
+oQVZOIrHF+s9IIFOIh3SIPIsXNvx7rUc5sgF4P+eAnAcv3o1zL7YjGJZ8e27Ai2
uF8iG/po/0Vd93OSB8Tj/Nvg99SSucy7nBYTreSdhUjZWRI0W1oYJX49/fhWljR9
8+f2GqUfLc7iCjcV3wxlfBqEKCdpjXsiqtsb1KrAx7AEOj7XfDjJjyCL4bshLp9x
PbV+kBFCN151iWYtfzhKEplrZFYNXlX2
-----END CERTIFICATE-----`)

// localhostKey is the private key for localhostCert.
var localhostKey = []byte(testingKey(`
-----BEGIN RSA TESTING KEY-----
MIIEvgIBADANBgkqhkiG9w0BAQEFAASCBKgwggSkAgEAAoIBAQDbsEfk1bK7ozwZ
lcQM8rBUikC4gwnnw0J1PUlGDGu1Y84dKtulbdWjyrh88D4fSdtmxFbXE7fhYUJT
BmEHSUk9OLHh/Tr+nSC3SfH0I/9y6l9j9vVVYhYJC07Z1mZZKVb+gmbbB7LEavGM
NaFHjvRJAwBX2TMDbXJceZ9jU/iihILkZbrbG40rn1mctYVmcR3YqOzI/ynLje97
FEvxtsg99OUjzzXyFMqfAl0J3Gc6tzvAER3N+ovKnudsnMB5Y+InQHHmPeizG4mF
yeBYesXNwX6cmI30c8KFiAlKHcsxjJsuoBZ3bSwvvFdK2hnuCO05HEgCzAQKUlY6
Q2F0xJblAgMBAAECggEACzZIOQraBB8M3G5rEtEZBDuJGZGgggpSXDrsQC22mouV
M6JiEuOT5Xfdagz10rF5h9lp6DCqsA8/bA7ViWJpYT1BQNwkdGWvC4Oz3EaxDRue
kjLCqyCmKMCBvfbmAtNsC/G6T5/pNQKTQNlk2YrXd1l2nUUpyBlAHq2bX52jwSGD
bFy5hyzSrzjeLpLUNZ56W/uXCvP0l6PAEvXRn/KG89XLZCtMBvVDMCfjIe77Q1U9
/XzIrnb67RzQwiDelvX+biMkBrjeYw/Gvdo9hNCOfbOZ+SpnfDOLEfAha/XPmr3+
5EeF4emeEhCODvfe7wy/4h1gHEG2N435S61DcV3gQQKBgQD92EJidwriPGDTUSM8
nJrPQ5xwPMKz5hWpfI0zxIYZyqA37eRC5Q9WD3rDbrEZiLCInFh+Ci899iLzEpFZ
dFQAUiRam+zFpDCQcGHr/uytRoTH/nxh2MrYPq8cA5ZGU6oMH+Yl4TynqJm2KN7e
0ocE07QjyK/9nIvEtdibEiFEwQKBgQDdjcgoqHaM49YJ4yxGpjuRdc5a3iuKzZU6
BON4GKqYQ9u/o8/NPaOSQ3vKhwzTjiEoOZImn+eX1cRP0ZskmQ+LyzsdVAHMDydz
9I23dbIywtCXGhKOJRwt9O++8ataWIxi1frjj6BcI+TzGl8LM2lYIfUHzVzfswwE
1EK8ikxnJQKBgBqPKvr0a54aJSNXBPHNjOEMuOyBXvnFpBSUpI17DXDbY4IWkOBy
6PTfL8AM79i1FYtlmFivphu8ihGWqsCKTFOwRH96ev5+3FnweD5h8M98Zl4qgUcX
kLmpbVboBSwcitkz6TejZl5AZLzLb+4uZtQZdmqcD9XgMDuHrz8iWXrBAoGBAMJO
Z34pCRfVddFkGF+5yMJw5FLTSLLKTJb+1JRuZad21BIF0+i3p25OmxHrUXd0zmWd
4CzZzt5eD3bFaOA3EOhUi/rTw2O44qwSjfuZUHiuXQw4RI+/wjAYAe+fud1ZjX3d
FtVfEI/etxvyQ+rp4vj1hxWZqVtThzXxBrqePBW1AoGBAOTC19rFQXtVf9A+8c/w
2ryAY2W9qNKe0xMivTAqau0Kdy2/2toJekR/5qOy3tOF7JasOgG+y3m3gLF47EFF
v75eW4FkiCFvsyl/qv4CO1eKnHlvkRoDsnMb+dA5czst58rO6BK40QvPqwXaSxj1
ee8ReNCDhC0Zidczajm63O1G
-----END RSA TESTING KEY-----`))

func testingKey(s string) string { return strings.ReplaceAll(s, "TESTING KEY", "PRIVATE KEY") }
