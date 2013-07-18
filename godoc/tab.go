// TODO(bradfitz,adg): move to util

package godoc

import "io"

var spaces = []byte("                                ") // 32 spaces seems like a good number

const (
	indenting = iota
	collecting
)

// A tconv is an io.Writer filter for converting leading tabs into spaces.
type tconv struct {
	output io.Writer
	state  int // indenting or collecting
	indent int // valid if state == indenting
	p      *Presentation
}

func (p *tconv) writeIndent() (err error) {
	i := p.indent
	for i >= len(spaces) {
		i -= len(spaces)
		if _, err = p.output.Write(spaces); err != nil {
			return
		}
	}
	// i < len(spaces)
	if i > 0 {
		_, err = p.output.Write(spaces[0:i])
	}
	return
}

func (p *tconv) Write(data []byte) (n int, err error) {
	if len(data) == 0 {
		return
	}
	pos := 0 // valid if p.state == collecting
	var b byte
	for n, b = range data {
		switch p.state {
		case indenting:
			switch b {
			case '\t':
				p.indent += p.p.TabWidth
			case '\n':
				p.indent = 0
				if _, err = p.output.Write(data[n : n+1]); err != nil {
					return
				}
			case ' ':
				p.indent++
			default:
				p.state = collecting
				pos = n
				if err = p.writeIndent(); err != nil {
					return
				}
			}
		case collecting:
			if b == '\n' {
				p.state = indenting
				p.indent = 0
				if _, err = p.output.Write(data[pos : n+1]); err != nil {
					return
				}
			}
		}
	}
	n = len(data)
	if pos < n && p.state == collecting {
		_, err = p.output.Write(data[pos:])
	}
	return
}
