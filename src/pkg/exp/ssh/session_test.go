// Copyright 2011 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

package ssh

// Session tests.

import (
	"bytes"
	"io"
	"testing"
)

// dial constructs a new test server and returns a *ClientConn.
func dial(t *testing.T) *ClientConn {
	pw := password("tiger")
	serverConfig.PasswordCallback = func(user, pass string) bool {
		return user == "testuser" && pass == string(pw)
	}
	serverConfig.PubKeyCallback = nil

	l, err := Listen("tcp", "127.0.0.1:0", serverConfig)
	if err != nil {
		t.Fatalf("unable to listen: %s", err)
	}
	go func() {
		defer l.Close()
		conn, err := l.Accept()
		if err != nil {
			t.Errorf("Unable to accept: %v", err)
			return
		}
		defer conn.Close()
		if err := conn.Handshake(); err != nil {
			t.Errorf("Unable to handshake: %v", err)
			return
		}
		for {
			ch, err := conn.Accept()
			if err == io.EOF {
				return
			}
			if err != nil {
				t.Errorf("Unable to accept incoming channel request: %v", err)
				return
			}
			if ch.ChannelType() != "session" {
				ch.Reject(UnknownChannelType, "unknown channel type")
				continue
			}
			ch.Accept()
			go func() {
				defer ch.Close()
				// this string is returned to stdout
				shell := NewServerShell(ch, "golang")
				shell.ReadLine()
				type exitMsg struct {
					PeersId   uint32
					Request   string
					WantReply bool
					Status    uint32
				}
				// TODO(dfc) converting to the concrete type should not be
				// necessary to send a packet.
				msg := exitMsg{
					PeersId:   ch.(*channel).theirId,
					Request:   "exit-status",
					WantReply: false,
					Status:    0,
				}
				ch.(*channel).serverConn.writePacket(marshal(msgChannelRequest, msg))
			}()
		}
		t.Log("done")
	}()

	config := &ClientConfig{
		User: "testuser",
		Auth: []ClientAuth{
			ClientAuthPassword(pw),
		},
	}

	c, err := Dial("tcp", l.Addr().String(), config)
	if err != nil {
		t.Fatalf("unable to dial remote side: %s", err)
	}
	return c
}

// Test a simple string is returned to session.Stdout.
func TestSessionShell(t *testing.T) {
	conn := dial(t)
	defer conn.Close()
	session, err := conn.NewSession()
	if err != nil {
		t.Fatalf("Unable to request new session: %s", err)
	}
	defer session.Close()
	stdout := new(bytes.Buffer)
	session.Stdout = stdout
	if err := session.Shell(); err != nil {
		t.Fatalf("Unable to execute command: %s", err)
	}
	if err := session.Wait(); err != nil {
		t.Fatalf("Remote command did not exit cleanly: %s", err)
	}
	actual := stdout.String()
	if actual != "golang" {
		t.Fatalf("Remote shell did not return expected string: expected=golang, actual=%s", actual)
	}
}

// TODO(dfc) add support for Std{in,err}Pipe when the Server supports it.

// Test a simple string is returned via StdoutPipe.
func TestSessionStdoutPipe(t *testing.T) {
	conn := dial(t)
	defer conn.Close()
	session, err := conn.NewSession()
	if err != nil {
		t.Fatalf("Unable to request new session: %s", err)
	}
	defer session.Close()
	stdout, err := session.StdoutPipe()
	if err != nil {
		t.Fatalf("Unable to request StdoutPipe(): %v", err)
	}
	var buf bytes.Buffer
	if err := session.Shell(); err != nil {
		t.Fatalf("Unable to execute command: %s", err)
	}
	done := make(chan bool, 1)
	go func() {
		if _, err := io.Copy(&buf, stdout); err != nil {
			t.Errorf("Copy of stdout failed: %v", err)
		}
		done <- true
	}()
	if err := session.Wait(); err != nil {
		t.Fatalf("Remote command did not exit cleanly: %s", err)
	}
	<-done
	actual := buf.String()
	if actual != "golang" {
		t.Fatalf("Remote shell did not return expected string: expected=golang, actual=%s", actual)
	}
}
