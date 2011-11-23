// Copyright 2011 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

package ssh

// direct-tcpip functional tests

import (
	"net"
	"net/http"
	"testing"
)

func TestTCPIPHTTP(t *testing.T) {
	if *sshuser == "" {
		t.Log("ssh.user not defined, skipping test")
		return
	}
	// google.com will generate at least one redirect, possibly three
	// depending on your location.
	doTest(t, "http://google.com")
}

func TestTCPIPHTTPS(t *testing.T) {
	if *sshuser == "" {
		t.Log("ssh.user not defined, skipping test")
		return
	}
	doTest(t, "https://encrypted.google.com/")
}

func doTest(t *testing.T, url string) {
	config := &ClientConfig{
		User: *sshuser,
		Auth: []ClientAuth{
			ClientAuthPassword(password(*sshpass)),
		},
	}
	conn, err := Dial("tcp", "localhost:22", config)
	if err != nil {
		t.Fatalf("Unable to connect: %s", err)
	}
	defer conn.Close()
	tr := &http.Transport{
		Dial: func(n, addr string) (net.Conn, error) {
			return conn.Dial(n, addr)
		},
	}
	client := &http.Client{
		Transport: tr,
	}
	resp, err := client.Get(url)
	if err != nil {
		t.Fatalf("unable to proxy: %s", err)
	}
	// got a body without error
	t.Log(resp)
}
