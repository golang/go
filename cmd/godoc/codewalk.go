// Copyright 2010 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

// The /doc/codewalk/ tree is synthesized from codewalk descriptions,
// files named $GOROOT/doc/codewalk/*.xml.
// For an example and a description of the format, see
// http://golang.org/doc/codewalk/codewalk or run godoc -http=:6060
// and see http://localhost:6060/doc/codewalk/codewalk .
// That page is itself a codewalk; the source code for it is
// $GOROOT/doc/codewalk/codewalk.xml.

package main

import (
	"bytes"
	"encoding/xml"
	"errors"
	"fmt"
	"io"
	"log"
	"net/http"
	"os"
	pathpkg "path"
	"regexp"
	"sort"
	"strconv"
	"strings"
	"text/template"
	"unicode/utf8"

	"golang.org/x/tools/godoc"
	"golang.org/x/tools/godoc/vfs"
)

var codewalkHTML, codewalkdirHTML *template.Template

// Handler for /doc/codewalk/ and below.
func codewalk(w http.ResponseWriter, r *http.Request) {
	relpath := r.URL.Path[len("/doc/codewalk/"):]
	abspath := r.URL.Path

	r.ParseForm()
	if f := r.FormValue("fileprint"); f != "" {
		codewalkFileprint(w, r, f)
		return
	}

	// If directory exists, serve list of code walks.
	dir, err := fs.Lstat(abspath)
	if err == nil && dir.IsDir() {
		codewalkDir(w, r, relpath, abspath)
		return
	}

	// If file exists, serve using standard file server.
	if err == nil {
		pres.ServeFile(w, r)
		return
	}

	// Otherwise append .xml and hope to find
	// a codewalk description, but before trim
	// the trailing /.
	abspath = strings.TrimRight(abspath, "/")
	cw, err := loadCodewalk(abspath + ".xml")
	if err != nil {
		log.Print(err)
		pres.ServeError(w, r, relpath, err)
		return
	}

	// Canonicalize the path and redirect if changed
	if redir(w, r) {
		return
	}

	pres.ServePage(w, godoc.Page{
		Title:    "Codewalk: " + cw.Title,
		Tabtitle: cw.Title,
		Body:     applyTemplate(codewalkHTML, "codewalk", cw),
	})
}

func redir(w http.ResponseWriter, r *http.Request) (redirected bool) {
	canonical := pathpkg.Clean(r.URL.Path)
	if !strings.HasSuffix(canonical, "/") {
		canonical += "/"
	}
	if r.URL.Path != canonical {
		url := *r.URL
		url.Path = canonical
		http.Redirect(w, r, url.String(), http.StatusMovedPermanently)
		redirected = true
	}
	return
}

func applyTemplate(t *template.Template, name string, data interface{}) []byte {
	var buf bytes.Buffer
	if err := t.Execute(&buf, data); err != nil {
		log.Printf("%s.Execute: %s", name, err)
	}
	return buf.Bytes()
}

// A Codewalk represents a single codewalk read from an XML file.
type Codewalk struct {
	Title string      `xml:"title,attr"`
	File  []string    `xml:"file"`
	Step  []*Codestep `xml:"step"`
}

// A Codestep is a single step in a codewalk.
type Codestep struct {
	// Filled in from XML
	Src   string `xml:"src,attr"`
	Title string `xml:"title,attr"`
	XML   string `xml:",innerxml"`

	// Derived from Src; not in XML.
	Err    error
	File   string
	Lo     int
	LoByte int
	Hi     int
	HiByte int
	Data   []byte
}

// String method for printing in template.
// Formats file address nicely.
func (st *Codestep) String() string {
	s := st.File
	if st.Lo != 0 || st.Hi != 0 {
		s += fmt.Sprintf(":%d", st.Lo)
		if st.Lo != st.Hi {
			s += fmt.Sprintf(",%d", st.Hi)
		}
	}
	return s
}

// loadCodewalk reads a codewalk from the named XML file.
func loadCodewalk(filename string) (*Codewalk, error) {
	f, err := fs.Open(filename)
	if err != nil {
		return nil, err
	}
	defer f.Close()
	cw := new(Codewalk)
	d := xml.NewDecoder(f)
	d.Entity = xml.HTMLEntity
	err = d.Decode(cw)
	if err != nil {
		return nil, &os.PathError{Op: "parsing", Path: filename, Err: err}
	}

	// Compute file list, evaluate line numbers for addresses.
	m := make(map[string]bool)
	for _, st := range cw.Step {
		i := strings.Index(st.Src, ":")
		if i < 0 {
			i = len(st.Src)
		}
		filename := st.Src[0:i]
		data, err := vfs.ReadFile(fs, filename)
		if err != nil {
			st.Err = err
			continue
		}
		if i < len(st.Src) {
			lo, hi, err := addrToByteRange(st.Src[i+1:], 0, data)
			if err != nil {
				st.Err = err
				continue
			}
			// Expand match to line boundaries.
			for lo > 0 && data[lo-1] != '\n' {
				lo--
			}
			for hi < len(data) && (hi == 0 || data[hi-1] != '\n') {
				hi++
			}
			st.Lo = byteToLine(data, lo)
			st.Hi = byteToLine(data, hi-1)
		}
		st.Data = data
		st.File = filename
		m[filename] = true
	}

	// Make list of files
	cw.File = make([]string, len(m))
	i := 0
	for f := range m {
		cw.File[i] = f
		i++
	}
	sort.Strings(cw.File)

	return cw, nil
}

// codewalkDir serves the codewalk directory listing.
// It scans the directory for subdirectories or files named *.xml
// and prepares a table.
func codewalkDir(w http.ResponseWriter, r *http.Request, relpath, abspath string) {
	type elem struct {
		Name  string
		Title string
	}

	dir, err := fs.ReadDir(abspath)
	if err != nil {
		log.Print(err)
		pres.ServeError(w, r, relpath, err)
		return
	}
	var v []interface{}
	for _, fi := range dir {
		name := fi.Name()
		if fi.IsDir() {
			v = append(v, &elem{name + "/", ""})
		} else if strings.HasSuffix(name, ".xml") {
			cw, err := loadCodewalk(abspath + "/" + name)
			if err != nil {
				continue
			}
			v = append(v, &elem{name[0 : len(name)-len(".xml")], cw.Title})
		}
	}

	pres.ServePage(w, godoc.Page{
		Title: "Codewalks",
		Body:  applyTemplate(codewalkdirHTML, "codewalkdir", v),
	})
}

// codewalkFileprint serves requests with ?fileprint=f&lo=lo&hi=hi.
// The filename f has already been retrieved and is passed as an argument.
// Lo and hi are the numbers of the first and last line to highlight
// in the response.  This format is used for the middle window pane
// of the codewalk pages.  It is a separate iframe and does not get
// the usual godoc HTML wrapper.
func codewalkFileprint(w http.ResponseWriter, r *http.Request, f string) {
	abspath := f
	data, err := vfs.ReadFile(fs, abspath)
	if err != nil {
		log.Print(err)
		pres.ServeError(w, r, f, err)
		return
	}
	lo, _ := strconv.Atoi(r.FormValue("lo"))
	hi, _ := strconv.Atoi(r.FormValue("hi"))
	if hi < lo {
		hi = lo
	}
	lo = lineToByte(data, lo)
	hi = lineToByte(data, hi+1)

	// Put the mark 4 lines before lo, so that the iframe
	// shows a few lines of context before the highlighted
	// section.
	n := 4
	mark := lo
	for ; mark > 0 && n > 0; mark-- {
		if data[mark-1] == '\n' {
			if n--; n == 0 {
				break
			}
		}
	}

	io.WriteString(w, `<style type="text/css">@import "/doc/codewalk/codewalk.css";</style><pre>`)
	template.HTMLEscape(w, data[0:mark])
	io.WriteString(w, "<a name='mark'></a>")
	template.HTMLEscape(w, data[mark:lo])
	if lo < hi {
		io.WriteString(w, "<div class='codewalkhighlight'>")
		template.HTMLEscape(w, data[lo:hi])
		io.WriteString(w, "</div>")
	}
	template.HTMLEscape(w, data[hi:])
	io.WriteString(w, "</pre>")
}

// addrToByte evaluates the given address starting at offset start in data.
// It returns the lo and hi byte offset of the matched region within data.
// See http://9p.io/sys/doc/sam/sam.html Table II for details on the syntax.
func addrToByteRange(addr string, start int, data []byte) (lo, hi int, err error) {
	var (
		dir        byte
		prevc      byte
		charOffset bool
	)
	lo = start
	hi = start
	for addr != "" && err == nil {
		c := addr[0]
		switch c {
		default:
			err = errors.New("invalid address syntax near " + string(c))
		case ',':
			if len(addr) == 1 {
				hi = len(data)
			} else {
				_, hi, err = addrToByteRange(addr[1:], hi, data)
			}
			return

		case '+', '-':
			if prevc == '+' || prevc == '-' {
				lo, hi, err = addrNumber(data, lo, hi, prevc, 1, charOffset)
			}
			dir = c

		case '$':
			lo = len(data)
			hi = len(data)
			if len(addr) > 1 {
				dir = '+'
			}

		case '#':
			charOffset = true

		case '0', '1', '2', '3', '4', '5', '6', '7', '8', '9':
			var i int
			for i = 1; i < len(addr); i++ {
				if addr[i] < '0' || addr[i] > '9' {
					break
				}
			}
			var n int
			n, err = strconv.Atoi(addr[0:i])
			if err != nil {
				break
			}
			lo, hi, err = addrNumber(data, lo, hi, dir, n, charOffset)
			dir = 0
			charOffset = false
			prevc = c
			addr = addr[i:]
			continue

		case '/':
			var i, j int
		Regexp:
			for i = 1; i < len(addr); i++ {
				switch addr[i] {
				case '\\':
					i++
				case '/':
					j = i + 1
					break Regexp
				}
			}
			if j == 0 {
				j = i
			}
			pattern := addr[1:i]
			lo, hi, err = addrRegexp(data, lo, hi, dir, pattern)
			prevc = c
			addr = addr[j:]
			continue
		}
		prevc = c
		addr = addr[1:]
	}

	if err == nil && dir != 0 {
		lo, hi, err = addrNumber(data, lo, hi, dir, 1, charOffset)
	}
	if err != nil {
		return 0, 0, err
	}
	return lo, hi, nil
}

// addrNumber applies the given dir, n, and charOffset to the address lo, hi.
// dir is '+' or '-', n is the count, and charOffset is true if the syntax
// used was #n.  Applying +n (or +#n) means to advance n lines
// (or characters) after hi.  Applying -n (or -#n) means to back up n lines
// (or characters) before lo.
// The return value is the new lo, hi.
func addrNumber(data []byte, lo, hi int, dir byte, n int, charOffset bool) (int, int, error) {
	switch dir {
	case 0:
		lo = 0
		hi = 0
		fallthrough

	case '+':
		if charOffset {
			pos := hi
			for ; n > 0 && pos < len(data); n-- {
				_, size := utf8.DecodeRune(data[pos:])
				pos += size
			}
			if n == 0 {
				return pos, pos, nil
			}
			break
		}
		// find next beginning of line
		if hi > 0 {
			for hi < len(data) && data[hi-1] != '\n' {
				hi++
			}
		}
		lo = hi
		if n == 0 {
			return lo, hi, nil
		}
		for ; hi < len(data); hi++ {
			if data[hi] != '\n' {
				continue
			}
			switch n--; n {
			case 1:
				lo = hi + 1
			case 0:
				return lo, hi + 1, nil
			}
		}

	case '-':
		if charOffset {
			// Scan backward for bytes that are not UTF-8 continuation bytes.
			pos := lo
			for ; pos > 0 && n > 0; pos-- {
				if data[pos]&0xc0 != 0x80 {
					n--
				}
			}
			if n == 0 {
				return pos, pos, nil
			}
			break
		}
		// find earlier beginning of line
		for lo > 0 && data[lo-1] != '\n' {
			lo--
		}
		hi = lo
		if n == 0 {
			return lo, hi, nil
		}
		for ; lo >= 0; lo-- {
			if lo > 0 && data[lo-1] != '\n' {
				continue
			}
			switch n--; n {
			case 1:
				hi = lo
			case 0:
				return lo, hi, nil
			}
		}
	}

	return 0, 0, errors.New("address out of range")
}

// addrRegexp searches for pattern in the given direction starting at lo, hi.
// The direction dir is '+' (search forward from hi) or '-' (search backward from lo).
// Backward searches are unimplemented.
func addrRegexp(data []byte, lo, hi int, dir byte, pattern string) (int, int, error) {
	re, err := regexp.Compile(pattern)
	if err != nil {
		return 0, 0, err
	}
	if dir == '-' {
		// Could implement reverse search using binary search
		// through file, but that seems like overkill.
		return 0, 0, errors.New("reverse search not implemented")
	}
	m := re.FindIndex(data[hi:])
	if len(m) > 0 {
		m[0] += hi
		m[1] += hi
	} else if hi > 0 {
		// No match.  Wrap to beginning of data.
		m = re.FindIndex(data)
	}
	if len(m) == 0 {
		return 0, 0, errors.New("no match for " + pattern)
	}
	return m[0], m[1], nil
}

// lineToByte returns the byte index of the first byte of line n.
// Line numbers begin at 1.
func lineToByte(data []byte, n int) int {
	if n <= 1 {
		return 0
	}
	n--
	for i, c := range data {
		if c == '\n' {
			if n--; n == 0 {
				return i + 1
			}
		}
	}
	return len(data)
}

// byteToLine returns the number of the line containing the byte at index i.
func byteToLine(data []byte, i int) int {
	l := 1
	for j, c := range data {
		if j == i {
			return l
		}
		if c == '\n' {
			l++
		}
	}
	return l
}
