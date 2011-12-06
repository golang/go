// Copyright 2011 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

package ssh

// Session implements an interactive session described in
// "RFC 4254, section 6".

import (
	"bytes"
	"errors"
	"fmt"
	"io"
	"io/ioutil"
)

type Signal string

// POSIX signals as listed in RFC 4254 Section 6.10.
const (
	SIGABRT Signal = "ABRT"
	SIGALRM Signal = "ALRM"
	SIGFPE  Signal = "FPE"
	SIGHUP  Signal = "HUP"
	SIGILL  Signal = "ILL"
	SIGINT  Signal = "INT"
	SIGKILL Signal = "KILL"
	SIGPIPE Signal = "PIPE"
	SIGQUIT Signal = "QUIT"
	SIGSEGV Signal = "SEGV"
	SIGTERM Signal = "TERM"
	SIGUSR1 Signal = "USR1"
	SIGUSR2 Signal = "USR2"
)

// A Session represents a connection to a remote command or shell.
type Session struct {
	// Stdin specifies the remote process's standard input.
	// If Stdin is nil, the remote process reads from an empty
	// bytes.Buffer.
	Stdin io.Reader

	// Stdout and Stderr specify the remote process's standard
	// output and error.
	//
	// If either is nil, Run connects the corresponding file
	// descriptor to an instance of ioutil.Discard. There is a
	// fixed amount of buffering that is shared for the two streams.
	// If either blocks it may eventually cause the remote
	// command to block.
	Stdout io.Writer
	Stderr io.Writer

	*clientChan // the channel backing this session

	started        bool // true once Start, Run or Shell is invoked.
	closeAfterWait []io.Closer
	copyFuncs      []func() error
	errch          chan error // one send per copyFunc
}

// RFC 4254 Section 6.4.
type setenvRequest struct {
	PeersId   uint32
	Request   string
	WantReply bool
	Name      string
	Value     string
}

// Setenv sets an environment variable that will be applied to any
// command executed by Shell or Run.
func (s *Session) Setenv(name, value string) error {
	req := setenvRequest{
		PeersId:   s.peersId,
		Request:   "env",
		WantReply: true,
		Name:      name,
		Value:     value,
	}
	if err := s.writePacket(marshal(msgChannelRequest, req)); err != nil {
		return err
	}
	return s.waitForResponse()
}

// An empty mode list, see RFC 4254 Section 8.
var emptyModelist = "\x00"

// RFC 4254 Section 6.2.
type ptyRequestMsg struct {
	PeersId   uint32
	Request   string
	WantReply bool
	Term      string
	Columns   uint32
	Rows      uint32
	Width     uint32
	Height    uint32
	Modelist  string
}

// RequestPty requests the association of a pty with the session on the remote host.
func (s *Session) RequestPty(term string, h, w int) error {
	req := ptyRequestMsg{
		PeersId:   s.peersId,
		Request:   "pty-req",
		WantReply: true,
		Term:      term,
		Columns:   uint32(w),
		Rows:      uint32(h),
		Width:     uint32(w * 8),
		Height:    uint32(h * 8),
		Modelist:  emptyModelist,
	}
	if err := s.writePacket(marshal(msgChannelRequest, req)); err != nil {
		return err
	}
	return s.waitForResponse()
}

// RFC 4254 Section 6.9.
type signalMsg struct {
	PeersId   uint32
	Request   string
	WantReply bool
	Signal    string
}

// Signal sends the given signal to the remote process.
// sig is one of the SIG* constants.
func (s *Session) Signal(sig Signal) error {
	req := signalMsg{
		PeersId:   s.peersId,
		Request:   "signal",
		WantReply: false,
		Signal:    string(sig),
	}
	return s.writePacket(marshal(msgChannelRequest, req))
}

// RFC 4254 Section 6.5.
type execMsg struct {
	PeersId   uint32
	Request   string
	WantReply bool
	Command   string
}

// Start runs cmd on the remote host. Typically, the remote
// server passes cmd to the shell for interpretation.
// A Session only accepts one call to Run, Start or Shell.
func (s *Session) Start(cmd string) error {
	if s.started {
		return errors.New("ssh: session already started")
	}
	req := execMsg{
		PeersId:   s.peersId,
		Request:   "exec",
		WantReply: true,
		Command:   cmd,
	}
	if err := s.writePacket(marshal(msgChannelRequest, req)); err != nil {
		return err
	}
	if err := s.waitForResponse(); err != nil {
		return fmt.Errorf("ssh: could not execute command %s: %v", cmd, err)
	}
	return s.start()
}

// Run runs cmd on the remote host and waits for it to terminate.
// Typically, the remote server passes cmd to the shell for
// interpretation. A Session only accepts one call to Run,
// Start or Shell.
func (s *Session) Run(cmd string) error {
	err := s.Start(cmd)
	if err != nil {
		return err
	}
	return s.Wait()
}

// Shell starts a login shell on the remote host. A Session only
// accepts one call to Run, Start or Shell.
func (s *Session) Shell() error {
	if s.started {
		return errors.New("ssh: session already started")
	}
	req := channelRequestMsg{
		PeersId:   s.peersId,
		Request:   "shell",
		WantReply: true,
	}
	if err := s.writePacket(marshal(msgChannelRequest, req)); err != nil {
		return err
	}
	if err := s.waitForResponse(); err != nil {
		return fmt.Errorf("ssh: cound not execute shell: %v", err)
	}
	return s.start()
}

func (s *Session) waitForResponse() error {
	msg := <-s.msg
	switch msg.(type) {
	case *channelRequestSuccessMsg:
		return nil
	case *channelRequestFailureMsg:
		return errors.New("request failed")
	}
	return fmt.Errorf("unknown packet %T received: %v", msg, msg)
}

func (s *Session) start() error {
	s.started = true

	type F func(*Session) error
	for _, setupFd := range []F{(*Session).stdin, (*Session).stdout, (*Session).stderr} {
		if err := setupFd(s); err != nil {
			return err
		}
	}

	s.errch = make(chan error, len(s.copyFuncs))
	for _, fn := range s.copyFuncs {
		go func(fn func() error) {
			s.errch <- fn()
		}(fn)
	}
	return nil
}

// Wait waits for the remote command to exit.
func (s *Session) Wait() error {
	if !s.started {
		return errors.New("ssh: session not started")
	}
	waitErr := s.wait()

	var copyError error
	for _ = range s.copyFuncs {
		if err := <-s.errch; err != nil && copyError == nil {
			copyError = err
		}
	}
	for _, fd := range s.closeAfterWait {
		fd.Close()
	}
	if waitErr != nil {
		return waitErr
	}
	return copyError
}

func (s *Session) wait() error {
	for {
		switch msg := (<-s.msg).(type) {
		case *channelRequestMsg:
			// TODO(dfc) improve this behavior to match os.Waitmsg
			switch msg.Request {
			case "exit-status":
				d := msg.RequestSpecificData
				status := int(d[0])<<24 | int(d[1])<<16 | int(d[2])<<8 | int(d[3])
				if status > 0 {
					return fmt.Errorf("remote process exited with %d", status)
				}
				return nil
			case "exit-signal":
				// TODO(dfc) make a more readable error message
				return fmt.Errorf("%v", msg.RequestSpecificData)
			default:
				return fmt.Errorf("wait: unexpected channel request: %v", msg)
			}
		default:
			return fmt.Errorf("wait: unexpected packet %T received: %v", msg, msg)
		}
	}
	panic("unreachable")
}

func (s *Session) stdin() error {
	if s.Stdin == nil {
		s.Stdin = new(bytes.Buffer)
	}
	s.copyFuncs = append(s.copyFuncs, func() error {
		_, err := io.Copy(s.clientChan.stdin, s.Stdin)
		if err1 := s.clientChan.stdin.Close(); err == nil {
			err = err1
		}
		return err
	})
	return nil
}

func (s *Session) stdout() error {
	if s.Stdout == nil {
		s.Stdout = ioutil.Discard
	}
	s.copyFuncs = append(s.copyFuncs, func() error {
		_, err := io.Copy(s.Stdout, s.clientChan.stdout)
		return err
	})
	return nil
}

func (s *Session) stderr() error {
	if s.Stderr == nil {
		s.Stderr = ioutil.Discard
	}
	s.copyFuncs = append(s.copyFuncs, func() error {
		_, err := io.Copy(s.Stderr, s.clientChan.stderr)
		return err
	})
	return nil
}

// StdinPipe returns a pipe that will be connected to the
// remote command's standard input when the command starts.
func (s *Session) StdinPipe() (io.WriteCloser, error) {
	if s.Stdin != nil {
		return nil, errors.New("ssh: Stdin already set")
	}
	if s.started {
		return nil, errors.New("ssh: StdinPipe after process started")
	}
	pr, pw := io.Pipe()
	s.Stdin = pr
	s.closeAfterWait = append(s.closeAfterWait, pr)
	return pw, nil
}

// StdoutPipe returns a pipe that will be connected to the
// remote command's standard output when the command starts.
// There is a fixed amount of buffering that is shared between
// stdout and stderr streams. If the StdoutPipe reader is
// not serviced fast enought it may eventually cause the
// remote command to block.
func (s *Session) StdoutPipe() (io.ReadCloser, error) {
	if s.Stdout != nil {
		return nil, errors.New("ssh: Stdout already set")
	}
	if s.started {
		return nil, errors.New("ssh: StdoutPipe after process started")
	}
	pr, pw := io.Pipe()
	s.Stdout = pw
	s.closeAfterWait = append(s.closeAfterWait, pw)
	return pr, nil
}

// StderrPipe returns a pipe that will be connected to the
// remote command's standard error when the command starts.
// There is a fixed amount of buffering that is shared between
// stdout and stderr streams. If the StderrPipe reader is
// not serviced fast enought it may eventually cause the
// remote command to block.
func (s *Session) StderrPipe() (io.ReadCloser, error) {
	if s.Stderr != nil {
		return nil, errors.New("ssh: Stderr already set")
	}
	if s.started {
		return nil, errors.New("ssh: StderrPipe after process started")
	}
	pr, pw := io.Pipe()
	s.Stderr = pw
	s.closeAfterWait = append(s.closeAfterWait, pw)
	return pr, nil
}

// TODO(dfc) add Output and CombinedOutput helpers

// NewSession returns a new interactive session on the remote host.
func (c *ClientConn) NewSession() (*Session, error) {
	ch := c.newChan(c.transport)
	if err := c.writePacket(marshal(msgChannelOpen, channelOpenMsg{
		ChanType:      "session",
		PeersId:       ch.id,
		PeersWindow:   1 << 14,
		MaxPacketSize: 1 << 15, // RFC 4253 6.1
	})); err != nil {
		c.chanlist.remove(ch.id)
		return nil, err
	}
	if err := ch.waitForChannelOpenResponse(); err != nil {
		c.chanlist.remove(ch.id)
		return nil, fmt.Errorf("ssh: unable to open session: %v", err)
	}
	return &Session{
		clientChan: ch,
	}, nil
}
