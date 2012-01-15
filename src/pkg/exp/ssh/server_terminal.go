// Copyright 2011 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

package ssh

// A Terminal is capable of parsing and generating virtual terminal
// data from an SSH client.
type Terminal interface {
	ReadLine() (line string, err error)
	SetSize(x, y int)
	Write([]byte) (int, error)
}

// ServerTerminal contains the state for running a terminal that is capable of
// reading lines of input.
type ServerTerminal struct {
	Term    Terminal
	Channel Channel
}

// parsePtyRequest parses the payload of the pty-req message and extracts the
// dimensions of the terminal. See RFC 4254, section 6.2.
func parsePtyRequest(s []byte) (width, height int, ok bool) {
	_, s, ok = parseString(s)
	if !ok {
		return
	}
	width32, s, ok := parseUint32(s)
	if !ok {
		return
	}
	height32, _, ok := parseUint32(s)
	width = int(width32)
	height = int(height32)
	if width < 1 {
		ok = false
	}
	if height < 1 {
		ok = false
	}
	return
}

func (ss *ServerTerminal) Write(buf []byte) (n int, err error) {
	return ss.Term.Write(buf)
}

// ReadLine returns a line of input from the terminal.
func (ss *ServerTerminal) ReadLine() (line string, err error) {
	for {
		if line, err = ss.Term.ReadLine(); err == nil {
			return
		}

		req, ok := err.(ChannelRequest)
		if !ok {
			return
		}

		ok = false
		switch req.Request {
		case "pty-req":
			var width, height int
			width, height, ok = parsePtyRequest(req.Payload)
			ss.Term.SetSize(width, height)
		case "shell":
			ok = true
			if len(req.Payload) > 0 {
				// We don't accept any commands, only the default shell.
				ok = false
			}
		case "env":
			ok = true
		}
		if req.WantReply {
			ss.Channel.AckRequest(ok)
		}
	}
	panic("unreachable")
}
