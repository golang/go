// Copyright 2021 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

package template

import (
	"strings"
	"testing"
)

type datum struct {
	buf  string
	cnt  int
	syms []string // the symbols in the parse of buf
}

var tmpl = []datum{{`
{{if (foo .X.Y)}}{{$A := "hi"}}{{.Z $A}}{{else}}
{{$A.X 12}}
{{foo (.X.Y) 23 ($A.Zü)}}
{{end}}`, 1, []string{"{7,3,foo,Function,false}", "{12,1,X,Method,false}",
	"{14,1,Y,Method,false}", "{21,2,$A,Variable,true}", "{26,2,,String,false}",
	"{35,1,Z,Method,false}", "{38,2,$A,Variable,false}",
	"{53,2,$A,Variable,false}", "{56,1,X,Method,false}", "{57,2,,Number,false}",
	"{64,3,foo,Function,false}", "{70,1,X,Method,false}",
	"{72,1,Y,Method,false}", "{75,2,,Number,false}", "{80,2,$A,Variable,false}",
	"{83,2,Zü,Method,false}", "{94,3,,Constant,false}"}},

	{`{{define "zzz"}}{{.}}{{end}}
{{template "zzz"}}`, 2, []string{"{10,3,zzz,Namespace,true}", "{18,1,dot,Variable,false}",
		"{41,3,zzz,Package,false}"}},

	{`{{block "aaa" foo}}b{{end}}`, 2, []string{"{9,3,aaa,Namespace,true}",
		"{9,3,aaa,Package,false}", "{14,3,foo,Function,false}", "{19,1,,Constant,false}"}},
}

func TestSymbols(t *testing.T) {
	for i, x := range tmpl {
		got := parseBuffer([]byte(x.buf))
		if got.ParseErr != nil {
			t.Errorf("error:%v", got.ParseErr)
			continue
		}
		if len(got.named) != x.cnt {
			t.Errorf("%d: got %d, expected %d", i, len(got.named), x.cnt)
		}
		for n, s := range got.symbols {
			if s.String() != x.syms[n] {
				t.Errorf("%d: got %s, expected %s", i, s.String(), x.syms[n])
			}
		}
	}
}

func TestWordAt(t *testing.T) {
	want := []string{"", "", "if", "if", "", "$A", "$A", "", "", "B", "", "", "end", "end", "end", "", ""}
	p := parseBuffer([]byte("{{if $A}}B{{end}}"))
	for i := 0; i < len(want); i++ {
		got := findWordAt(p, i)
		if got != want[i] {
			t.Errorf("for %d, got %q, wanted %q", i, got, want[i])
		}
	}
}

func TestNLS(t *testing.T) {
	buf := `{{if (foÜx .X.Y)}}{{$A := "hi"}}{{.Z $A}}{{else}}
	{{$A.X 12}}
	{{foo (.X.Y) 23 ($A.Z)}}
	{{end}}
	`
	p := parseBuffer([]byte(buf))
	if p.ParseErr != nil {
		t.Fatal(p.ParseErr)
	}
	// line 0 doesn't have a \n in front of it
	for i := 1; i < len(p.nls)-1; i++ {
		if buf[p.nls[i]] != '\n' {
			t.Errorf("line %d got %c", i, buf[p.nls[i]])
		}
	}
	// fake line at end of file
	if p.nls[len(p.nls)-1] != len(buf) {
		t.Errorf("got %d expected %d", p.nls[len(p.nls)-1], len(buf))
	}
}

func TestLineCol(t *testing.T) {
	buf := `{{if (foÜx .X.Y)}}{{$A := "hi"}}{{.Z $A}}{{else}}
	{{$A.X 12}}
	{{foo (.X.Y) 23 ($A.Z)}}
	{{end}}`
	if false {
		t.Error(buf)
	}
	for n, cx := range tmpl {
		buf := cx.buf
		p := parseBuffer([]byte(buf))
		if p.ParseErr != nil {
			t.Fatal(p.ParseErr)
		}
		type loc struct {
			offset int
			l, c   uint32
		}
		saved := []loc{}
		// forwards
		var lastl, lastc uint32
		for offset := range buf {
			l, c := p.LineCol(offset)
			saved = append(saved, loc{offset, l, c})
			if l > lastl {
				lastl = l
				if c != 0 {
					t.Errorf("line %d, got %d instead of 0", l, c)
				}
			}
			if c > lastc {
				lastc = c
			}
		}
		lines := strings.Split(buf, "\n")
		mxlen := -1
		for _, l := range lines {
			if len(l) > mxlen {
				mxlen = len(l)
			}
		}
		if int(lastl) != len(lines)-1 && int(lastc) != mxlen {
			// lastl is 0 if there is only 1 line(?)
			t.Errorf("expected %d, %d, got %d, %d for case %d", len(lines)-1, mxlen, lastl, lastc, n)
		}
		// backwards
		for j := len(saved) - 1; j >= 0; j-- {
			s := saved[j]
			xl, xc := p.LineCol(s.offset)
			if xl != s.l || xc != s.c {
				t.Errorf("at offset %d(%d), got (%d,%d), expected (%d,%d)", s.offset, j, xl, xc, s.l, s.c)
			}
		}
	}
}

func TestPos(t *testing.T) {
	buf := `
	{{if (foÜx .X.Y)}}{{$A := "hi"}}{{.Z $A}}{{else}}
	{{$A.X 12}}
	{{foo (.X.Y) 23 ($A.Z)}}
	{{end}}`
	p := parseBuffer([]byte(buf))
	if p.ParseErr != nil {
		t.Fatal(p.ParseErr)
	}
	for pos, r := range buf {
		if r == '\n' {
			continue
		}
		x := p.Position(pos)
		n := p.FromPosition(x)
		if n != pos {
			// once it's wrong, it will be wrong forever
			t.Fatalf("at pos %d (rune %c) got %d {%#v]", pos, r, n, x)
		}

	}
}
func TestLen(t *testing.T) {
	data := []struct {
		cnt int
		v   string
	}{{1, "a"}, {1, "膈"}, {4, "😆🥸"}, {7, "3😀4567"}}
	p := &Parsed{nonASCII: true}
	for _, d := range data {
		got := p.utf16len([]byte(d.v))
		if got != d.cnt {
			t.Errorf("%v, got %d wanted %d", d, got, d.cnt)
		}
	}
}

func TestUtf16(t *testing.T) {
	buf := `
	{{if (foÜx .X.Y)}}😀{{$A := "hi"}}{{.Z $A}}{{else}}
	{{$A.X 12}}
	{{foo (.X.Y) 23 ($A.Z)}}
	{{end}}`
	p := parseBuffer([]byte(buf))
	if p.nonASCII == false {
		t.Error("expected nonASCII to be true")
	}
}
