// Copyright 2015 The Go Authors.  All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

package obj

import (
	"bufio"
	"fmt"
	"io"
	"os"
	"strconv"
	"time"
)

var start time.Time

func Cputime() float64 {
	if start.IsZero() {
		start = time.Now()
	}
	return time.Since(start).Seconds()
}

type Biobuf struct {
	unget     int
	haveUnget bool
	f         *os.File
	r         *bufio.Reader
	w         *bufio.Writer
}

func Bopenw(name string) (*Biobuf, error) {
	f, err := os.Open(name)
	if err != nil {
		return nil, err
	}
	return &Biobuf{f: f, w: bufio.NewWriter(f)}, nil
}

func Binitw(w io.Writer) *Biobuf {
	return &Biobuf{w: bufio.NewWriter(w)}
}

func (b *Biobuf) Write(p []byte) (int, error) {
	return b.w.Write(p)
}

func (b *Biobuf) Flush() error {
	return b.w.Flush()
}

func Bwrite(b *Biobuf, p []byte) (int, error) {
	return b.w.Write(p)
}

func Bputc(b *Biobuf, c byte) {
	b.w.WriteByte(c)
}

func Bgetc(b *Biobuf) int {
	if b.haveUnget {
		b.haveUnget = false
		return int(b.unget)
	}
	c, err := b.r.ReadByte()
	if err != nil {
		b.unget = -1
		return -1
	}
	b.unget = int(c)
	return int(c)
}

func Bungetc(b *Biobuf) {
	b.haveUnget = true
}

func Boffset(b *Biobuf) int64 {
	panic("Boffset")
}

func Bflush(b *Biobuf) error {
	return b.w.Flush()
}

func Bterm(b *Biobuf) error {
	err := b.w.Flush()
	err1 := b.f.Close()
	if err == nil {
		err = err1
	}
	return err
}

func envOr(key, value string) string {
	if x := os.Getenv(key); x != "" {
		return x
	}
	return value
}

func Getgoroot() string {
	return envOr("GOROOT", defaultGOROOT)
}

func Getgoarch() string {
	return envOr("GOARCH", defaultGOARCH)
}

func Getgoos() string {
	return envOr("GOOS", defaultGOOS)
}

func Getgoarm() string {
	return envOr("GOARM", defaultGOARM)
}

func Getgoversion() string {
	return version
}

func Atoi(s string) int {
	i, _ := strconv.Atoi(s)
	return i
}

func (p *Prog) Line() string {
	return Linklinefmt(p.Ctxt, int(p.Lineno), false, false)
}

func (p *Prog) String() string {
	if p.Ctxt == nil {
		return fmt.Sprintf("<Prog without ctxt>")
	}
	return p.Ctxt.Arch.Pconv(p)
}

func (ctxt *Link) NewProg() *Prog {
	p := new(Prog) // should be the only call to this; all others should use ctxt.NewProg
	p.Ctxt = ctxt
	return p
}

func (ctxt *Link) Line(n int) string {
	return Linklinefmt(ctxt, n, false, false)
}
