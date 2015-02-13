// Copyright 2015 The Go Authors.  All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

package obj

import (
	"bufio"
	"fmt"
	"io"
	"log"
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
	unget    [2]int
	numUnget int
	f        *os.File
	r        *bufio.Reader
	w        *bufio.Writer
	linelen  int
}

func Bopenw(name string) (*Biobuf, error) {
	f, err := os.Create(name)
	if err != nil {
		return nil, err
	}
	return &Biobuf{f: f, w: bufio.NewWriter(f)}, nil
}

func Bopenr(name string) (*Biobuf, error) {
	f, err := os.Open(name)
	if err != nil {
		return nil, err
	}
	return &Biobuf{f: f, r: bufio.NewReader(f)}, nil
}

func Binitw(w io.Writer) *Biobuf {
	return &Biobuf{w: bufio.NewWriter(w)}
}

func (b *Biobuf) Write(p []byte) (int, error) {
	return b.w.Write(p)
}

func Bwritestring(b *Biobuf, p string) (int, error) {
	return b.w.WriteString(p)
}

func Bseek(b *Biobuf, offset int64, whence int) int64 {
	if b.w != nil {
		if err := b.w.Flush(); err != nil {
			log.Fatal("writing output: %v", err)
		}
	} else if b.r != nil {
		if whence == 1 {
			offset -= int64(b.r.Buffered())
		}
	}
	off, err := b.f.Seek(offset, whence)
	if err != nil {
		log.Fatal("seeking in output: %v", err)
	}
	if b.r != nil {
		b.r.Reset(b.f)
	}
	return off
}

func Boffset(b *Biobuf) int64 {
	if err := b.w.Flush(); err != nil {
		log.Fatal("writing output: %v", err)
	}
	off, err := b.f.Seek(0, 1)
	if err != nil {
		log.Fatal("seeking in output: %v", err)
	}
	return off
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

const Beof = -1

func Bread(b *Biobuf, p []byte) int {
	n, err := io.ReadFull(b.r, p)
	if n == 0 {
		if err != nil && err != io.EOF {
			n = -1
		}
	}
	return n
}

func Bgetc(b *Biobuf) int {
	if b.numUnget > 0 {
		b.numUnget--
		return int(b.unget[b.numUnget])
	}
	c, err := b.r.ReadByte()
	r := int(c)
	if err != nil {
		r = -1
	}
	b.unget[1] = b.unget[0]
	b.unget[0] = r
	return r
}

func Bgetrune(b *Biobuf) int {
	r, _, err := b.r.ReadRune()
	if err != nil {
		return -1
	}
	return int(r)
}

func Bungetrune(b *Biobuf) {
	b.r.UnreadRune()
}

func (b *Biobuf) Read(p []byte) (int, error) {
	return b.r.Read(p)
}

func Brdline(b *Biobuf, delim int) string {
	s, err := b.r.ReadBytes(byte(delim))
	if err != nil {
		log.Fatalf("reading input: %v", err)
	}
	b.linelen = len(s)
	return string(s)
}

func Brdstr(b *Biobuf, delim int, cut int) string {
	s, err := b.r.ReadString(byte(delim))
	if err != nil {
		log.Fatalf("reading input: %v", err)
	}
	if len(s) > 0 && cut > 0 {
		s = s[:len(s)-1]
	}
	return s
}

func Access(name string, mode int) int {
	if mode != 0 {
		panic("bad access")
	}
	_, err := os.Stat(name)
	if err != nil {
		return -1
	}
	return 0
}

func Blinelen(b *Biobuf) int {
	return b.linelen
}

func Bungetc(b *Biobuf) {
	b.numUnget++
}

func Bflush(b *Biobuf) error {
	return b.w.Flush()
}

func Bterm(b *Biobuf) error {
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

func Getgo386() string {
	return envOr("GO386", defaultGO386)
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

func (ctxt *Link) Dconv(a *Addr) string {
	return ctxt.Arch.Dconv(nil, 0, a)
}

func (ctxt *Link) Rconv(reg int) string {
	return ctxt.Arch.Rconv(reg)
}

func Getcallerpc(interface{}) uintptr {
	return 1
}
