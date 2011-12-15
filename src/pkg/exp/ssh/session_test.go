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

type serverType func(*channel)

// dial constructs a new test server and returns a *ClientConn.
func dial(handler serverType, t *testing.T) *ClientConn {
	pw := password("tiger")
	serverConfig.PasswordCallback = func(user, pass string) bool {
		return user == "testuser" && pass == string(pw)
	}
	serverConfig.PublicKeyCallback = nil

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
			go handler(ch.(*channel))
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
	conn := dial(shellHandler, t)
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
	conn := dial(shellHandler, t)
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

// Test non-0 exit status is returned correctly.
func TestExitStatusNonZero(t *testing.T) {
	conn := dial(exitStatusNonZeroHandler, t)
	defer conn.Close()
	session, err := conn.NewSession()
	if err != nil {
		t.Fatalf("Unable to request new session: %s", err)
	}
	defer session.Close()
	if err := session.Shell(); err != nil {
		t.Fatalf("Unable to execute command: %s", err)
	}
	err = session.Wait()
	if err == nil {
		t.Fatalf("expected command to fail but it didn't")
	}
	e, ok := err.(*ExitError)
	if !ok {
		t.Fatalf("expected *ExitError but got %T", err)
	}
	if e.ExitStatus() != 15 {
		t.Fatalf("expected command to exit with 15 but got %s", e.ExitStatus())
	}
}

// Test 0 exit status is returned correctly.
func TestExitStatusZero(t *testing.T) {
	conn := dial(exitStatusZeroHandler, t)
	defer conn.Close()
	session, err := conn.NewSession()
	if err != nil {
		t.Fatalf("Unable to request new session: %s", err)
	}
	defer session.Close()

	if err := session.Shell(); err != nil {
		t.Fatalf("Unable to execute command: %s", err)
	}
	err = session.Wait()
	if err != nil {
		t.Fatalf("expected nil but got %s", err)
	}
}

// Test exit signal and status are both returned correctly.
func TestExitSignalAndStatus(t *testing.T) {
	conn := dial(exitSignalAndStatusHandler, t)
	defer conn.Close()
	session, err := conn.NewSession()
	if err != nil {
		t.Fatalf("Unable to request new session: %s", err)
	}
	defer session.Close()
	if err := session.Shell(); err != nil {
		t.Fatalf("Unable to execute command: %s", err)
	}
	err = session.Wait()
	if err == nil {
		t.Fatalf("expected command to fail but it didn't")
	}
	e, ok := err.(*ExitError)
	if !ok {
		t.Fatalf("expected *ExitError but got %T", err)
	}
	if e.Signal() != "TERM" || e.ExitStatus() != 15 {
		t.Fatalf("expected command to exit with signal TERM and status 15 but got signal %s and status %v", e.Signal(), e.ExitStatus())
	}
}

// Test exit signal and status are both returned correctly.
func TestKnownExitSignalOnly(t *testing.T) {
	conn := dial(exitSignalHandler, t)
	defer conn.Close()
	session, err := conn.NewSession()
	if err != nil {
		t.Fatalf("Unable to request new session: %s", err)
	}
	defer session.Close()
	if err := session.Shell(); err != nil {
		t.Fatalf("Unable to execute command: %s", err)
	}
	err = session.Wait()
	if err == nil {
		t.Fatalf("expected command to fail but it didn't")
	}
	e, ok := err.(*ExitError)
	if !ok {
		t.Fatalf("expected *ExitError but got %T", err)
	}
	if e.Signal() != "TERM" || e.ExitStatus() != 143 {
		t.Fatalf("expected command to exit with signal TERM and status 143 but got signal %s and status %v", e.Signal(), e.ExitStatus())
	}
}

// Test exit signal and status are both returned correctly.
func TestUnknownExitSignal(t *testing.T) {
	conn := dial(exitSignalUnknownHandler, t)
	defer conn.Close()
	session, err := conn.NewSession()
	if err != nil {
		t.Fatalf("Unable to request new session: %s", err)
	}
	defer session.Close()
	if err := session.Shell(); err != nil {
		t.Fatalf("Unable to execute command: %s", err)
	}
	err = session.Wait()
	if err == nil {
		t.Fatalf("expected command to fail but it didn't")
	}
	e, ok := err.(*ExitError)
	if !ok {
		t.Fatalf("expected *ExitError but got %T", err)
	}
	if e.Signal() != "SYS" || e.ExitStatus() != 128 {
		t.Fatalf("expected command to exit with signal SYS and status 128 but got signal %s and status %v", e.Signal(), e.ExitStatus())
	}
}

// Test WaitMsg is not returned if the channel closes abruptly.
func TestExitWithoutStatusOrSignal(t *testing.T) {
	conn := dial(exitWithoutSignalOrStatus, t)
	defer conn.Close()
	session, err := conn.NewSession()
	if err != nil {
		t.Fatalf("Unable to request new session: %s", err)
	}
	defer session.Close()
	if err := session.Shell(); err != nil {
		t.Fatalf("Unable to execute command: %s", err)
	}
	err = session.Wait()
	if err == nil {
		t.Fatalf("expected command to fail but it didn't")
	}
	_, ok := err.(*ExitError)
	if ok {
		// you can't actually test for errors.errorString
		// because it's not exported.
		t.Fatalf("expected *errorString but got %T", err)
	}
}

type exitStatusMsg struct {
	PeersId   uint32
	Request   string
	WantReply bool
	Status    uint32
}

type exitSignalMsg struct {
	PeersId    uint32
	Request    string
	WantReply  bool
	Signal     string
	CoreDumped bool
	Errmsg     string
	Lang       string
}

func exitStatusZeroHandler(ch *channel) {
	defer ch.Close()
	// this string is returned to stdout
	shell := NewServerShell(ch, "> ")
	shell.ReadLine()
	sendStatus(0, ch)
}

func exitStatusNonZeroHandler(ch *channel) {
	defer ch.Close()
	shell := NewServerShell(ch, "> ")
	shell.ReadLine()
	sendStatus(15, ch)
}

func exitSignalAndStatusHandler(ch *channel) {
	defer ch.Close()
	shell := NewServerShell(ch, "> ")
	shell.ReadLine()
	sendStatus(15, ch)
	sendSignal("TERM", ch)
}

func exitSignalHandler(ch *channel) {
	defer ch.Close()
	shell := NewServerShell(ch, "> ")
	shell.ReadLine()
	sendSignal("TERM", ch)
}

func exitSignalUnknownHandler(ch *channel) {
	defer ch.Close()
	shell := NewServerShell(ch, "> ")
	shell.ReadLine()
	sendSignal("SYS", ch)
}

func exitWithoutSignalOrStatus(ch *channel) {
	defer ch.Close()
	shell := NewServerShell(ch, "> ")
	shell.ReadLine()
}

func shellHandler(ch *channel) {
	defer ch.Close()
	// this string is returned to stdout
	shell := NewServerShell(ch, "golang")
	shell.ReadLine()
	sendStatus(0, ch)
}

func sendStatus(status uint32, ch *channel) {
	msg := exitStatusMsg{
		PeersId:   ch.theirId,
		Request:   "exit-status",
		WantReply: false,
		Status:    status,
	}
	ch.serverConn.writePacket(marshal(msgChannelRequest, msg))
}

func sendSignal(signal string, ch *channel) {
	sig := exitSignalMsg{
		PeersId:    ch.theirId,
		Request:    "exit-signal",
		WantReply:  false,
		Signal:     signal,
		CoreDumped: false,
		Errmsg:     "Process terminated",
		Lang:       "en-GB-oed",
	}
	ch.serverConn.writePacket(marshal(msgChannelRequest, sig))
}
