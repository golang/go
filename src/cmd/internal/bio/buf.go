// Copyright 2015 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

// Package bio implements seekable buffered I/O.
package bio

import (
	"bufio"
	"io"
	"log"
	"os"
)

const EOF = -1

// Buf implements a seekable buffered I/O abstraction.
type Buf struct {
	f *os.File
	r *bufio.Reader
	w *bufio.Writer
}

func (b *Buf) Reader() *bufio.Reader { return b.r }
func (b *Buf) Writer() *bufio.Writer { return b.w }

func Create(name string) (*Buf, error) {
	f, err := os.Create(name)
	if err != nil {
		return nil, err
	}
	return &Buf{f: f, w: bufio.NewWriter(f)}, nil
}

func Open(name string) (*Buf, error) {
	f, err := os.Open(name)
	if err != nil {
		return nil, err
	}
	return &Buf{f: f, r: bufio.NewReader(f)}, nil
}

func BufWriter(w io.Writer) *Buf {
	return &Buf{w: bufio.NewWriter(w)}
}

func BufReader(r io.Reader) *Buf {
	return &Buf{r: bufio.NewReader(r)}
}

func (b *Buf) Write(p []byte) (int, error) {
	return b.w.Write(p)
}

func (b *Buf) WriteString(p string) (int, error) {
	return b.w.WriteString(p)
}

func Bseek(b *Buf, offset int64, whence int) int64 {
	if b.w != nil {
		if err := b.w.Flush(); err != nil {
			log.Fatalf("writing output: %v", err)
		}
	} else if b.r != nil {
		if whence == 1 {
			offset -= int64(b.r.Buffered())
		}
	}
	off, err := b.f.Seek(offset, whence)
	if err != nil {
		log.Fatalf("seeking in output: %v", err)
	}
	if b.r != nil {
		b.r.Reset(b.f)
	}
	return off
}

func Boffset(b *Buf) int64 {
	if b.w != nil {
		if err := b.w.Flush(); err != nil {
			log.Fatalf("writing output: %v", err)
		}
	}
	off, err := b.f.Seek(0, 1)
	if err != nil {
		log.Fatalf("seeking in output [0, 1]: %v", err)
	}
	if b.r != nil {
		off -= int64(b.r.Buffered())
	}
	return off
}

func (b *Buf) Flush() error {
	return b.w.Flush()
}

func (b *Buf) WriteByte(c byte) error {
	return b.w.WriteByte(c)
}

func Bread(b *Buf, p []byte) int {
	n, err := io.ReadFull(b.r, p)
	if n == 0 {
		if err != nil && err != io.EOF {
			n = -1
		}
	}
	return n
}

func Bgetc(b *Buf) int {
	c, err := b.r.ReadByte()
	if err != nil {
		if err != io.EOF {
			log.Fatalf("reading input: %v", err)
		}
		return EOF
	}
	return int(c)
}

func (b *Buf) Read(p []byte) (int, error) {
	return b.r.Read(p)
}

func (b *Buf) Peek(n int) ([]byte, error) {
	return b.r.Peek(n)
}

func Brdline(b *Buf, delim int) string {
	s, err := b.r.ReadBytes(byte(delim))
	if err != nil {
		log.Fatalf("reading input: %v", err)
	}
	return string(s)
}

func (b *Buf) Close() error {
	var err error
	if b.w != nil {
		err = b.w.Flush()
	}
	err1 := b.f.Close()
	if err == nil {
		err = err1
	}
	return err
}
