// Copyright 2009 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

package obj

import (
	"fmt"
	"path/filepath"
	"strings"
)

const (
	HISTSZ = 10
	NSYM   = 50
)

func Linklinefmt(ctxt *Link, lno0 int, showAll, showFullPath bool) string {
	var a [HISTSZ]struct {
		incl *Hist
		idel int32
		line *Hist
		ldel int32
	}
	lno := int32(lno0)
	lno1 := lno
	var d int32
	n := 0
	for h := ctxt.Hist; h != nil; h = h.Link {
		if h.Offset < 0 {
			continue
		}
		if lno < h.Line {
			break
		}
		if h.Name != "<pop>" {
			if h.Offset > 0 {
				// #line directive
				if n > 0 && n < int(HISTSZ) {
					a[n-1].line = h
					a[n-1].ldel = h.Line - h.Offset + 1
				}
			} else {
				// beginning of file
				if n < int(HISTSZ) {
					a[n].incl = h
					a[n].idel = h.Line
					a[n].line = nil
				}
				n++
			}
			continue
		}
		n--
		if n > 0 && n < int(HISTSZ) {
			d = h.Line - a[n].incl.Line
			a[n-1].ldel += d
			a[n-1].idel += d
		}
	}
	if n > int(HISTSZ) {
		n = int(HISTSZ)
	}
	var fp string
	for i := n - 1; i >= 0; i-- {
		if i != n-1 {
			if !showAll {
				break
			}
			fp += " "
		}
		if ctxt.Debugline != 0 || showFullPath {
			fp += fmt.Sprintf("%s/", ctxt.Pathname)
		}
		if a[i].line != nil {
			fp += fmt.Sprintf("%s:%d[%s:%d]", a[i].line.Name, lno-a[i].ldel+1, a[i].incl.Name, lno-a[i].idel+1)
		} else {
			fp += fmt.Sprintf("%s:%d", a[i].incl.Name, lno-a[i].idel+1)
		}
		lno = a[i].incl.Line - 1 // now print out start of this file
	}
	if n == 0 {
		fp += fmt.Sprintf("<unknown line number %d %d %d %s>", lno1, ctxt.Hist.Offset, ctxt.Hist.Line, ctxt.Hist.Name)
	}
	return fp
}

// Does s have t as a path prefix?
// That is, does s == t or does s begin with t followed by a slash?
// For portability, we allow ASCII case folding, so that haspathprefix("a/b/c", "A/B") is true.
// Similarly, we allow slash folding, so that haspathprefix("a/b/c", "a\\b") is true.
func haspathprefix(s string, t string) bool {
	if len(t) > len(s) {
		return false
	}
	var i int
	var cs int
	var ct int
	for i = 0; i < len(t); i++ {
		cs = int(s[i])
		ct = int(t[i])
		if 'A' <= cs && cs <= 'Z' {
			cs += 'a' - 'A'
		}
		if 'A' <= ct && ct <= 'Z' {
			ct += 'a' - 'A'
		}
		if cs == '\\' {
			cs = '/'
		}
		if ct == '\\' {
			ct = '/'
		}
		if cs != ct {
			return false
		}
	}
	return i >= len(s) || s[i] == '/' || s[i] == '\\'
}

// This is a simplified copy of linklinefmt above.
// It doesn't allow printing the full stack, and it returns the file name and line number separately.
// TODO: Unify with linklinefmt somehow.
func linkgetline(ctxt *Link, line int32, f **LSym, l *int32) {
	var a [HISTSZ]struct {
		incl *Hist
		idel int32
		line *Hist
		ldel int32
	}
	var d int32
	lno := int32(line)
	n := 0
	for h := ctxt.Hist; h != nil; h = h.Link {
		if h.Offset < 0 {
			continue
		}
		if lno < h.Line {
			break
		}
		if h.Name != "<pop>" {
			if h.Offset > 0 {
				// #line directive
				if n > 0 && n < HISTSZ {
					a[n-1].line = h
					a[n-1].ldel = h.Line - h.Offset + 1
				}
			} else {
				// beginning of file
				if n < HISTSZ {
					a[n].incl = h
					a[n].idel = h.Line
					a[n].line = nil
				}
				n++
			}
			continue
		}
		n--
		if n > 0 && n < HISTSZ {
			d = h.Line - a[n].incl.Line
			a[n-1].ldel += d
			a[n-1].idel += d
		}
	}
	if n > HISTSZ {
		n = HISTSZ
	}
	if n <= 0 {
		*f = Linklookup(ctxt, "??", HistVersion)
		*l = 0
		return
	}
	n--
	var dlno int32
	var file string
	if a[n].line != nil {
		file = a[n].line.Name
		dlno = a[n].ldel - 1
	} else {
		file = a[n].incl.Name
		dlno = a[n].idel - 1
	}
	var buf string
	if filepath.IsAbs(file) || strings.HasPrefix(file, "<") {
		buf = fmt.Sprintf("%s", file)
	} else {
		buf = fmt.Sprintf("%s/%s", ctxt.Pathname, file)
	}
	// Remove leading ctxt->trimpath, or else rewrite $GOROOT to $GOROOT_FINAL.
	if ctxt.Trimpath != "" && haspathprefix(buf, ctxt.Trimpath) {
		if len(buf) == len(ctxt.Trimpath) {
			buf = "??"
		} else {
			buf1 := fmt.Sprintf("%s", buf[len(ctxt.Trimpath)+1:])
			if buf1[0] == '\x00' {
				buf1 = "??"
			}
			buf = buf1
		}
	} else if ctxt.Goroot_final != "" && haspathprefix(buf, ctxt.Goroot) {
		buf1 := fmt.Sprintf("%s%s", ctxt.Goroot_final, buf[len(ctxt.Goroot):])
		buf = buf1
	}
	lno -= dlno
	*f = Linklookup(ctxt, buf, HistVersion)
	*l = lno
}

func Linklinehist(ctxt *Link, lineno int, f string, offset int) {
	if false { // debug['f']
		if f != "" {
			if offset != 0 {
				fmt.Printf("%4d: %s (#line %d)\n", lineno, f, offset)
			} else {
				fmt.Printf("%4d: %s\n", lineno, f)
			}
		} else {
			fmt.Printf("%4d: <pop>\n", lineno)
		}
	}

	h := new(Hist)
	*h = Hist{}
	h.Name = f
	h.Line = int32(lineno)
	h.Offset = int32(offset)
	h.Link = nil
	if ctxt.Ehist == nil {
		ctxt.Hist = h
		ctxt.Ehist = h
		return
	}

	ctxt.Ehist.Link = h
	ctxt.Ehist = h
}

func Linkprfile(ctxt *Link, line int) {
	l := int32(line)
	var i int
	var a [HISTSZ]Hist
	var d int32
	n := 0
	for h := ctxt.Hist; h != nil; h = h.Link {
		if l < h.Line {
			break
		}
		if h.Name != "<pop>" {
			if h.Offset == 0 {
				if n >= 0 && n < HISTSZ {
					a[n] = *h
				}
				n++
				continue
			}
			if n > 0 && n < HISTSZ {
				if a[n-1].Offset == 0 {
					a[n] = *h
					n++
				} else {
					a[n-1] = *h
				}
			}
			continue
		}
		n--
		if n >= 0 && n < HISTSZ {
			d = h.Line - a[n].Line
			for i = 0; i < n; i++ {
				a[i].Line += d
			}
		}
	}
	if n > HISTSZ {
		n = HISTSZ
	}
	for i := 0; i < n; i++ {
		fmt.Printf("%s:%d ", a[i].Name, int(l-a[i].Line+a[i].Offset+1))
	}
}

/*
 * start a new Prog list.
 */
func Linknewplist(ctxt *Link) *Plist {
	pl := new(Plist)
	*pl = Plist{}
	if ctxt.Plist == nil {
		ctxt.Plist = pl
	} else {
		ctxt.Plast.Link = pl
	}
	ctxt.Plast = pl

	return pl
}
