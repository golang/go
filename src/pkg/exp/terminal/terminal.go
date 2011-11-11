// Copyright 2011 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

package terminal

import "io"

// Terminal contains the state for running a VT100 terminal that is capable of
// reading lines of input.
type Terminal struct {
	c      io.ReadWriter
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

// NewTerminal runs a VT100 terminal on the given ReadWriter. If the ReadWriter is
// a local terminal, that terminal must first have been put into raw mode.
// prompt is a string that is written at the start of each input line (i.e.
// "> ").
func NewTerminal(c io.ReadWriter, prompt string) *Terminal {
	return &Terminal{
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

// queue appends data to the end of t.outBuf
func (t *Terminal) queue(data []byte) {
	if len(t.outBuf)+len(data) > cap(t.outBuf) {
		newOutBuf := make([]byte, len(t.outBuf), 2*(len(t.outBuf)+len(data)))
		copy(newOutBuf, t.outBuf)
		t.outBuf = newOutBuf
	}

	oldLen := len(t.outBuf)
	t.outBuf = t.outBuf[:len(t.outBuf)+len(data)]
	copy(t.outBuf[oldLen:], data)
}

var eraseUnderCursor = []byte{' ', keyEscape, '[', 'D'}

func isPrintable(key int) bool {
	return key >= 32 && key < 127
}

// moveCursorToPos appends data to t.outBuf which will move the cursor to the
// given, logical position in the text.
func (t *Terminal) moveCursorToPos(pos int) {
	x := len(t.prompt) + pos
	y := x / t.termWidth
	x = x % t.termWidth

	up := 0
	if y < t.cursorY {
		up = t.cursorY - y
	}

	down := 0
	if y > t.cursorY {
		down = y - t.cursorY
	}

	left := 0
	if x < t.cursorX {
		left = t.cursorX - x
	}

	right := 0
	if x > t.cursorX {
		right = x - t.cursorX
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

	t.cursorX = x
	t.cursorY = y
	t.queue(movement)
}

const maxLineLength = 4096

// handleKey processes the given key and, optionally, returns a line of text
// that the user has entered.
func (t *Terminal) handleKey(key int) (line string, ok bool) {
	switch key {
	case keyBackspace:
		if t.pos == 0 {
			return
		}
		t.pos--

		copy(t.line[t.pos:], t.line[1+t.pos:])
		t.line = t.line[:len(t.line)-1]
		t.writeLine(t.line[t.pos:])
		t.moveCursorToPos(t.pos)
		t.queue(eraseUnderCursor)
	case keyAltLeft:
		// move left by a word.
		if t.pos == 0 {
			return
		}
		t.pos--
		for t.pos > 0 {
			if t.line[t.pos] != ' ' {
				break
			}
			t.pos--
		}
		for t.pos > 0 {
			if t.line[t.pos] == ' ' {
				t.pos++
				break
			}
			t.pos--
		}
		t.moveCursorToPos(t.pos)
	case keyAltRight:
		// move right by a word.
		for t.pos < len(t.line) {
			if t.line[t.pos] == ' ' {
				break
			}
			t.pos++
		}
		for t.pos < len(t.line) {
			if t.line[t.pos] != ' ' {
				break
			}
			t.pos++
		}
		t.moveCursorToPos(t.pos)
	case keyLeft:
		if t.pos == 0 {
			return
		}
		t.pos--
		t.moveCursorToPos(t.pos)
	case keyRight:
		if t.pos == len(t.line) {
			return
		}
		t.pos++
		t.moveCursorToPos(t.pos)
	case keyEnter:
		t.moveCursorToPos(len(t.line))
		t.queue([]byte("\r\n"))
		line = string(t.line)
		ok = true
		t.line = t.line[:0]
		t.pos = 0
		t.cursorX = 0
		t.cursorY = 0
		t.maxLine = 0
	default:
		if !isPrintable(key) {
			return
		}
		if len(t.line) == maxLineLength {
			return
		}
		if len(t.line) == cap(t.line) {
			newLine := make([]byte, len(t.line), 2*(1+len(t.line)))
			copy(newLine, t.line)
			t.line = newLine
		}
		t.line = t.line[:len(t.line)+1]
		copy(t.line[t.pos+1:], t.line[t.pos:])
		t.line[t.pos] = byte(key)
		t.writeLine(t.line[t.pos:])
		t.pos++
		t.moveCursorToPos(t.pos)
	}
	return
}

func (t *Terminal) writeLine(line []byte) {
	for len(line) != 0 {
		if t.cursorX == t.termWidth {
			t.queue([]byte("\r\n"))
			t.cursorX = 0
			t.cursorY++
			if t.cursorY > t.maxLine {
				t.maxLine = t.cursorY
			}
		}

		remainingOnLine := t.termWidth - t.cursorX
		todo := len(line)
		if todo > remainingOnLine {
			todo = remainingOnLine
		}
		t.queue(line[:todo])
		t.cursorX += todo
		line = line[todo:]
	}
}

func (t *Terminal) Write(buf []byte) (n int, err error) {
	return t.c.Write(buf)
}

// ReadLine returns a line of input from the terminal.
func (t *Terminal) ReadLine() (line string, err error) {
	if t.cursorX == 0 {
		t.writeLine([]byte(t.prompt))
		t.c.Write(t.outBuf)
		t.outBuf = t.outBuf[:0]
	}

	for {
		// t.remainder is a slice at the beginning of t.inBuf
		// containing a partial key sequence
		readBuf := t.inBuf[len(t.remainder):]
		var n int
		n, err = t.c.Read(readBuf)
		if err != nil {
			return
		}

		if err == nil {
			t.remainder = t.inBuf[:n+len(t.remainder)]
			rest := t.remainder
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
				line, lineOk = t.handleKey(key)
			}
			if len(rest) > 0 {
				n := copy(t.inBuf[:], rest)
				t.remainder = t.inBuf[:n]
			} else {
				t.remainder = nil
			}
			t.c.Write(t.outBuf)
			t.outBuf = t.outBuf[:0]
			if lineOk {
				return
			}
			continue
		}
	}
	panic("unreachable")
}

func (t *Terminal) SetSize(width, height int) {
	t.termWidth, t.termHeight = width, height
}
