// Copyright 2011 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

package ssh

import "io"

// ServerShell contains the state for running a VT100 terminal that is capable
// of reading lines of input.
type ServerShell struct {
	c      Channel
	prompt string

	// line is the current line being entered.
	line []byte
	// pos is the logical position of the cursor in line
	pos int

	// cursorX contains the current X value of the cursor where the left
	// edge is 0. cursorY contains the row number where the first row of
	// the current line is 0.
	cursorX, cursorY int
	// maxLine is the greatest value of cursorY so far.
	maxLine int

	termWidth, termHeight int

	// outBuf contains the terminal data to be sent.
	outBuf []byte
	// remainder contains the remainder of any partial key sequences after
	// a read. It aliases into inBuf.
	remainder []byte
	inBuf     [256]byte
}

// NewServerShell runs a VT100 terminal on the given channel. prompt is a
// string that is written at the start of each input line. For example: "> ".
func NewServerShell(c Channel, prompt string) *ServerShell {
	return &ServerShell{
		c:          c,
		prompt:     prompt,
		termWidth:  80,
		termHeight: 24,
	}
}

const (
	keyCtrlD     = 4
	keyEnter     = '\r'
	keyEscape    = 27
	keyBackspace = 127
	keyUnknown   = 256 + iota
	keyUp
	keyDown
	keyLeft
	keyRight
	keyAltLeft
	keyAltRight
)

// bytesToKey tries to parse a key sequence from b. If successful, it returns
// the key and the remainder of the input. Otherwise it returns -1.
func bytesToKey(b []byte) (int, []byte) {
	if len(b) == 0 {
		return -1, nil
	}

	if b[0] != keyEscape {
		return int(b[0]), b[1:]
	}

	if len(b) >= 3 && b[0] == keyEscape && b[1] == '[' {
		switch b[2] {
		case 'A':
			return keyUp, b[3:]
		case 'B':
			return keyDown, b[3:]
		case 'C':
			return keyRight, b[3:]
		case 'D':
			return keyLeft, b[3:]
		}
	}

	if len(b) >= 6 && b[0] == keyEscape && b[1] == '[' && b[2] == '1' && b[3] == ';' && b[4] == '3' {
		switch b[5] {
		case 'C':
			return keyAltRight, b[6:]
		case 'D':
			return keyAltLeft, b[6:]
		}
	}

	// If we get here then we have a key that we don't recognise, or a
	// partial sequence. It's not clear how one should find the end of a
	// sequence without knowing them all, but it seems that [a-zA-Z] only
	// appears at the end of a sequence.
	for i, c := range b[0:] {
		if c >= 'a' && c <= 'z' || c >= 'A' && c <= 'Z' {
			return keyUnknown, b[i+1:]
		}
	}

	return -1, b
}

// queue appends data to the end of ss.outBuf
func (ss *ServerShell) queue(data []byte) {
	if len(ss.outBuf)+len(data) > cap(ss.outBuf) {
		newOutBuf := make([]byte, len(ss.outBuf), 2*(len(ss.outBuf)+len(data)))
		copy(newOutBuf, ss.outBuf)
		ss.outBuf = newOutBuf
	}

	oldLen := len(ss.outBuf)
	ss.outBuf = ss.outBuf[:len(ss.outBuf)+len(data)]
	copy(ss.outBuf[oldLen:], data)
}

var eraseUnderCursor = []byte{' ', keyEscape, '[', 'D'}

func isPrintable(key int) bool {
	return key >= 32 && key < 127
}

// moveCursorToPos appends data to ss.outBuf which will move the cursor to the
// given, logical position in the text.
func (ss *ServerShell) moveCursorToPos(pos int) {
	x := len(ss.prompt) + pos
	y := x / ss.termWidth
	x = x % ss.termWidth

	up := 0
	if y < ss.cursorY {
		up = ss.cursorY - y
	}

	down := 0
	if y > ss.cursorY {
		down = y - ss.cursorY
	}

	left := 0
	if x < ss.cursorX {
		left = ss.cursorX - x
	}

	right := 0
	if x > ss.cursorX {
		right = x - ss.cursorX
	}

	movement := make([]byte, 3*(up+down+left+right))
	m := movement
	for i := 0; i < up; i++ {
		m[0] = keyEscape
		m[1] = '['
		m[2] = 'A'
		m = m[3:]
	}
	for i := 0; i < down; i++ {
		m[0] = keyEscape
		m[1] = '['
		m[2] = 'B'
		m = m[3:]
	}
	for i := 0; i < left; i++ {
		m[0] = keyEscape
		m[1] = '['
		m[2] = 'D'
		m = m[3:]
	}
	for i := 0; i < right; i++ {
		m[0] = keyEscape
		m[1] = '['
		m[2] = 'C'
		m = m[3:]
	}

	ss.cursorX = x
	ss.cursorY = y
	ss.queue(movement)
}

const maxLineLength = 4096

// handleKey processes the given key and, optionally, returns a line of text
// that the user has entered.
func (ss *ServerShell) handleKey(key int) (line string, ok bool) {
	switch key {
	case keyBackspace:
		if ss.pos == 0 {
			return
		}
		ss.pos--

		copy(ss.line[ss.pos:], ss.line[1+ss.pos:])
		ss.line = ss.line[:len(ss.line)-1]
		ss.writeLine(ss.line[ss.pos:])
		ss.moveCursorToPos(ss.pos)
		ss.queue(eraseUnderCursor)
	case keyAltLeft:
		// move left by a word.
		if ss.pos == 0 {
			return
		}
		ss.pos--
		for ss.pos > 0 {
			if ss.line[ss.pos] != ' ' {
				break
			}
			ss.pos--
		}
		for ss.pos > 0 {
			if ss.line[ss.pos] == ' ' {
				ss.pos++
				break
			}
			ss.pos--
		}
		ss.moveCursorToPos(ss.pos)
	case keyAltRight:
		// move right by a word.
		for ss.pos < len(ss.line) {
			if ss.line[ss.pos] == ' ' {
				break
			}
			ss.pos++
		}
		for ss.pos < len(ss.line) {
			if ss.line[ss.pos] != ' ' {
				break
			}
			ss.pos++
		}
		ss.moveCursorToPos(ss.pos)
	case keyLeft:
		if ss.pos == 0 {
			return
		}
		ss.pos--
		ss.moveCursorToPos(ss.pos)
	case keyRight:
		if ss.pos == len(ss.line) {
			return
		}
		ss.pos++
		ss.moveCursorToPos(ss.pos)
	case keyEnter:
		ss.moveCursorToPos(len(ss.line))
		ss.queue([]byte("\r\n"))
		line = string(ss.line)
		ok = true
		ss.line = ss.line[:0]
		ss.pos = 0
		ss.cursorX = 0
		ss.cursorY = 0
		ss.maxLine = 0
	default:
		if !isPrintable(key) {
			return
		}
		if len(ss.line) == maxLineLength {
			return
		}
		if len(ss.line) == cap(ss.line) {
			newLine := make([]byte, len(ss.line), 2*(1+len(ss.line)))
			copy(newLine, ss.line)
			ss.line = newLine
		}
		ss.line = ss.line[:len(ss.line)+1]
		copy(ss.line[ss.pos+1:], ss.line[ss.pos:])
		ss.line[ss.pos] = byte(key)
		ss.writeLine(ss.line[ss.pos:])
		ss.pos++
		ss.moveCursorToPos(ss.pos)
	}
	return
}

func (ss *ServerShell) writeLine(line []byte) {
	for len(line) != 0 {
		if ss.cursorX == ss.termWidth {
			ss.queue([]byte("\r\n"))
			ss.cursorX = 0
			ss.cursorY++
			if ss.cursorY > ss.maxLine {
				ss.maxLine = ss.cursorY
			}
		}

		remainingOnLine := ss.termWidth - ss.cursorX
		todo := len(line)
		if todo > remainingOnLine {
			todo = remainingOnLine
		}
		ss.queue(line[:todo])
		ss.cursorX += todo
		line = line[todo:]
	}
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

func (ss *ServerShell) Write(buf []byte) (n int, err error) {
	return ss.c.Write(buf)
}

// ReadLine returns a line of input from the terminal.
func (ss *ServerShell) ReadLine() (line string, err error) {
	ss.writeLine([]byte(ss.prompt))
	ss.c.Write(ss.outBuf)
	ss.outBuf = ss.outBuf[:0]

	for {
		// ss.remainder is a slice at the beginning of ss.inBuf
		// containing a partial key sequence
		readBuf := ss.inBuf[len(ss.remainder):]
		var n int
		n, err = ss.c.Read(readBuf)
		if err == nil {
			ss.remainder = ss.inBuf[:n+len(ss.remainder)]
			rest := ss.remainder
			lineOk := false
			for !lineOk {
				var key int
				key, rest = bytesToKey(rest)
				if key < 0 {
					break
				}
				if key == keyCtrlD {
					return "", io.EOF
				}
				line, lineOk = ss.handleKey(key)
			}
			if len(rest) > 0 {
				n := copy(ss.inBuf[:], rest)
				ss.remainder = ss.inBuf[:n]
			} else {
				ss.remainder = nil
			}
			ss.c.Write(ss.outBuf)
			ss.outBuf = ss.outBuf[:0]
			if lineOk {
				return
			}
			continue
		}

		if req, ok := err.(ChannelRequest); ok {
			ok := false
			switch req.Request {
			case "pty-req":
				ss.termWidth, ss.termHeight, ok = parsePtyRequest(req.Payload)
				if !ok {
					ss.termWidth = 80
					ss.termHeight = 24
				}
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
				ss.c.AckRequest(ok)
			}
		} else {
			return "", err
		}
	}
	panic("unreachable")
}
