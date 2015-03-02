// Copyright 2015 The Go Authors.  All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

package ld

import (
	"bufio"
	"bytes"
	"encoding/binary"
	"io"
	"log"
	"os"
	"runtime/pprof"
	"strings"
	"time"
)

func cstring(x []byte) string {
	i := bytes.IndexByte(x, '\x00')
	if i >= 0 {
		x = x[:i]
	}
	return string(x)
}

func plan9quote(s string) string {
	if s == "" {
		return "'" + strings.Replace(s, "'", "''", -1) + "'"
	}
	for i := 0; i < len(s); i++ {
		if s[i] <= ' ' || s[i] == '\'' {
			return "'" + strings.Replace(s, "'", "''", -1) + "'"
		}
	}
	return s
}

func tokenize(s string) []string {
	var f []string
	for {
		s = strings.TrimLeft(s, " \t\r\n")
		if s == "" {
			break
		}
		quote := false
		i := 0
		for ; i < len(s); i++ {
			if s[i] == '\'' {
				if quote && i+1 < len(s) && s[i+1] == '\'' {
					i++
					continue
				}
				quote = !quote
			}
			if !quote && (s[i] == ' ' || s[i] == '\t' || s[i] == '\r' || s[i] == '\n') {
				break
			}
		}
		next := s[:i]
		s = s[i:]
		if strings.Contains(next, "'") {
			var buf []byte
			quote := false
			for i := 0; i < len(next); i++ {
				if next[i] == '\'' {
					if quote && i+1 < len(next) && next[i+1] == '\'' {
						i++
						buf = append(buf, '\'')
					}
					quote = !quote
					continue
				}
				buf = append(buf, next[i])
			}
			next = string(buf)
		}
		f = append(f, next)
	}
	return f
}

func cutStringAtNUL(s string) string {
	if i := strings.Index(s, "\x00"); i >= 0 {
		s = s[:i]
	}
	return s
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

func Binitw(w *os.File) *Biobuf {
	return &Biobuf{w: bufio.NewWriter(w), f: w}
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
			log.Fatalf("writing output: %v", err)
		}
	} else if b.r != nil {
		if whence == 1 {
			offset -= int64(b.r.Buffered())
		}
	}
	off, err := b.f.Seek(offset, whence)
	if err != nil {
		log.Panicf("seeking in output [%d %d %p]: %v", offset, whence, b.f, err)
	}
	if b.r != nil {
		b.r.Reset(b.f)
	}
	return off
}

func Boffset(b *Biobuf) int64 {
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
	if b.numUnget > 0 {
		Bseek(b, -int64(b.numUnget), 1)
		b.numUnget = 0
	}
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
	if b.numUnget > 0 {
		Bseek(b, -int64(b.numUnget), 1)
		b.numUnget = 0
	}
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
	if b.numUnget > 0 {
		Bseek(b, -int64(b.numUnget), 1)
		b.numUnget = 0
	}
	s, err := b.r.ReadBytes(byte(delim))
	if err != nil {
		log.Fatalf("reading input: %v", err)
	}
	b.linelen = len(s)
	return string(s)
}

func Brdstr(b *Biobuf, delim int, cut int) string {
	if b.numUnget > 0 {
		Bseek(b, -int64(b.numUnget), 1)
		b.numUnget = 0
	}
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

// strings.Compare, introduced in Go 1.5.
func stringsCompare(a, b string) int {
	if a == b {
		return 0
	}
	if a < b {
		return -1
	}
	return +1
}

var atExitFuncs []func()

func AtExit(f func()) {
	atExitFuncs = append(atExitFuncs, f)
}

func Exit(code int) {
	for i := len(atExitFuncs) - 1; i >= 0; i-- {
		f := atExitFuncs[i]
		atExitFuncs = atExitFuncs[:i]
		f()
	}
	os.Exit(code)
}

var cpuprofile string
var memprofile string

func startProfile() {
	if cpuprofile != "" {
		f, err := os.Create(cpuprofile)
		if err != nil {
			log.Fatalf("%v", err)
		}
		if err := pprof.StartCPUProfile(f); err != nil {
			log.Fatalf("%v", err)
		}
		AtExit(pprof.StopCPUProfile)
	}
	if memprofile != "" {
		f, err := os.Create(memprofile)
		if err != nil {
			log.Fatalf("%v", err)
		}
		AtExit(func() {
			if err := pprof.WriteHeapProfile(f); err != nil {
				log.Fatalf("%v", err)
			}
		})
	}
}

func artrim(x []byte) string {
	i := 0
	j := len(x)
	for i < len(x) && x[i] == ' ' {
		i++
	}
	for j > i && x[j-1] == ' ' {
		j--
	}
	return string(x[i:j])
}

func stringtouint32(x []uint32, s string) {
	for i := 0; len(s) > 0; i++ {
		var buf [4]byte
		s = s[copy(buf[:], s):]
		x[i] = binary.LittleEndian.Uint32(buf[:])
	}
}

var start = time.Now()

func elapsed() float64 {
	return time.Since(start).Seconds()
}
