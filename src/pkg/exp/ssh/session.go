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

	started   bool // true once a Shell or Exec is invoked.
	copyFuncs []func() error
	errch     chan error // one send per copyFunc
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
// command executed by Shell or Exec.
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

// RFC 4254 Section 6.5.
type execMsg struct {
	PeersId   uint32
	Request   string
	WantReply bool
	Command   string
}

// Exec runs cmd on the remote host. Typically, the remote 
// server passes cmd to the shell for interpretation. 
// A Session only accepts one call to Exec or Shell.
func (s *Session) Exec(cmd string) error {
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
	if err := s.start(); err != nil {
		return err
	}
	return s.Wait()
}

// Shell starts a login shell on the remote host. A Session only 
// accepts one call to Exec or Shell.
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
		_, err := io.Copy(&chanWriter{
			packetWriter: s,
			peersId:      s.peersId,
			win:          s.win,
		}, s.Stdin)
		return err
	})
	return nil
}

func (s *Session) stdout() error {
	if s.Stdout == nil {
		s.Stdout = ioutil.Discard
	}
	s.copyFuncs = append(s.copyFuncs, func() error {
		_, err := io.Copy(s.Stdout, &chanReader{
			packetWriter: s,
			peersId:      s.peersId,
			data:         s.data,
		})
		return err
	})
	return nil
}

func (s *Session) stderr() error {
	if s.Stderr == nil {
		s.Stderr = ioutil.Discard
	}
	s.copyFuncs = append(s.copyFuncs, func() error {
		_, err := io.Copy(s.Stderr, &chanReader{
			packetWriter: s,
			peersId:      s.peersId,
			data:         s.dataExt,
		})
		return err
	})
	return nil
}

// NewSession returns a new interactive session on the remote host.
func (c *ClientConn) NewSession() (*Session, error) {
	ch, err := c.openChan("session")
	if err != nil {
		return nil, err
	}
	return &Session{
		clientChan: ch,
	}, nil
}
