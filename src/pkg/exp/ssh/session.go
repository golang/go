// Copyright 2011 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

package ssh

// Session implements an interactive session described in
// "RFC 4254, section 6".

import (
	"encoding/binary"
	"errors"
	"io"
)

// A Session represents a connection to a remote command or shell.
type Session struct {
	// Writes to Stdin are made available to the remote command's standard input.
	// Closing Stdin causes the command to observe an EOF on its standard input.
	Stdin io.WriteCloser

	// Reads from Stdout and Stderr consume from the remote command's standard
	// output and error streams, respectively.
	// There is a fixed amount of buffering that is shared for the two streams.
	// Failing to read from either may eventually cause the command to block.
	// Closing Stdout unblocks such writes and causes them to return errors.
	Stdout io.ReadCloser
	Stderr io.Reader

	*clientChan // the channel backing this session

	started bool // started is set to true once a Shell or Exec is invoked.
}

// Setenv sets an environment variable that will be applied to any
// command executed by Shell or Exec.
func (s *Session) Setenv(name, value string) error {
	n, v := []byte(name), []byte(value)
	nlen, vlen := stringLength(n), stringLength(v)
	payload := make([]byte, nlen+vlen)
	marshalString(payload[:nlen], n)
	marshalString(payload[nlen:], v)

	return s.sendChanReq(channelRequestMsg{
		PeersId:             s.id,
		Request:             "env",
		WantReply:           true,
		RequestSpecificData: payload,
	})
}

// An empty mode list (a string of 1 character, opcode 0), see RFC 4254 Section 8.
var emptyModeList = []byte{0, 0, 0, 1, 0}

// RequestPty requests the association of a pty with the session on the remote host.
func (s *Session) RequestPty(term string, h, w int) error {
	buf := make([]byte, 4+len(term)+16+len(emptyModeList))
	b := marshalString(buf, []byte(term))
	binary.BigEndian.PutUint32(b, uint32(h))
	binary.BigEndian.PutUint32(b[4:], uint32(w))
	binary.BigEndian.PutUint32(b[8:], uint32(h*8))
	binary.BigEndian.PutUint32(b[12:], uint32(w*8))
	copy(b[16:], emptyModeList)

	return s.sendChanReq(channelRequestMsg{
		PeersId:             s.id,
		Request:             "pty-req",
		WantReply:           true,
		RequestSpecificData: buf,
	})
}

// Exec runs cmd on the remote host. Typically, the remote 
// server passes cmd to the shell for interpretation. 
// A Session only accepts one call to Exec or Shell.
func (s *Session) Exec(cmd string) error {
	if s.started {
		return errors.New("session already started")
	}
	cmdLen := stringLength([]byte(cmd))
	payload := make([]byte, cmdLen)
	marshalString(payload, []byte(cmd))
	s.started = true

	return s.sendChanReq(channelRequestMsg{
		PeersId:             s.id,
		Request:             "exec",
		WantReply:           true,
		RequestSpecificData: payload,
	})
}

// Shell starts a login shell on the remote host. A Session only 
// accepts one call to Exec or Shell.
func (s *Session) Shell() error {
	if s.started {
		return errors.New("session already started")
	}
	s.started = true

	return s.sendChanReq(channelRequestMsg{
		PeersId:   s.id,
		Request:   "shell",
		WantReply: true,
	})
}

// NewSession returns a new interactive session on the remote host.
func (c *ClientConn) NewSession() (*Session, error) {
	ch, err := c.openChan("session")
	if err != nil {
		return nil, err
	}
	return &Session{
		Stdin: &chanWriter{
			packetWriter: ch,
			id:           ch.id,
			win:          ch.win,
		},
		Stdout: &chanReader{
			packetWriter: ch,
			id:           ch.id,
			data:         ch.data,
		},
		Stderr: &chanReader{
			packetWriter: ch,
			id:           ch.id,
			data:         ch.dataExt,
		},
		clientChan: ch,
	}, nil
}
