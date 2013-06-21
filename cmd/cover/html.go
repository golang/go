// Copyright 2013 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

package main

import (
	"bufio"
	"fmt"
	"go/build"
	"html"
	"io"
	"io/ioutil"
	"os"
	"os/exec"
	"path/filepath"
	"regexp"
	"runtime"
	"sort"
	"strconv"
)

// htmlOutput reads the profile data from profile and generates an HTML
// coverage report, writing it to outfile. If outfile is empty,
// it writes the report to a temporary file and opens it in a web browser.
func htmlOutput(profile, outfile string) error {
	pf, err := os.Open(profile)
	if err != nil {
		return err
	}
	defer pf.Close()

	profiles, err := ParseProfiles(pf)
	if err != nil {
		return err
	}

	var out *os.File
	if outfile == "" {
		var dir string
		dir, err = ioutil.TempDir("", "cover")
		if err != nil {
			return err
		}
		out, err = os.Create(filepath.Join(dir, "coverage.html"))
	} else {
		out, err = os.Create(outfile)
	}
	if err != nil {
		return err
	}

	for fn, profile := range profiles {
		dir, file := filepath.Split(fn)
		pkg, err := build.Import(dir, ".", build.FindOnly)
		if err != nil {
			return fmt.Errorf("can't find %q: %v", fn, err)
		}
		src, err := ioutil.ReadFile(filepath.Join(pkg.Dir, file))
		if err != nil {
			return fmt.Errorf("can't read %q: %v", fn, err)
		}
		err = htmlGen(out, fn, src, profile.Tokens(src))
		if err != nil {
			out.Close()
			return err
		}
	}

	err = out.Close()
	if err != nil {
		return err
	}

	if outfile == "" {
		if !startBrowser("file://" + out.Name()) {
			fmt.Fprintf(os.Stderr, "HTML output written to %s\n", out.Name())
		}
	}

	return nil
}

// Profile represents the profiling data for a specific file.
type Profile struct {
	Blocks []ProfileBlock
}

// ProfileBlock represents a single block of profiling data.
type ProfileBlock struct {
	StartLine, StartCol int
	EndLine, EndCol     int
	NumStmt, Count      int
}

// ParseProfiles parses profile data from the given Reader and returns a
// Profile for each file.
func ParseProfiles(r io.Reader) (map[string]*Profile, error) {
	files := make(map[string]*Profile)
	buf := bufio.NewReader(r)
	// First line is mode.
	mode, err := buf.ReadString('\n')
	if err != nil {
		return nil, err
	}
	_ = mode // TODO: Use the mode to affect the display.
	// Rest of file is in the format
	//	encoding/base64/base64.go:34.44,37.40 3 1
	// where the fields are: name.go:line.column,line.column numberOfStatements count
	s := bufio.NewScanner(buf)
	for s.Scan() {
		line := s.Text()
		m := lineRe.FindStringSubmatch(line)
		if m == nil {
			return nil, fmt.Errorf("line %q doesn't match expected format: %v", m, lineRe)
		}
		fn := m[1]
		p := files[fn]
		if p == nil {
			p = new(Profile)
			files[fn] = p
		}
		p.Blocks = append(p.Blocks, ProfileBlock{
			StartLine: toInt(m[2]),
			StartCol:  toInt(m[3]),
			EndLine:   toInt(m[4]),
			EndCol:    toInt(m[5]),
			NumStmt:   toInt(m[6]),
			Count:     toInt(m[7]),
		})
	}
	if err := s.Err(); err != nil {
		return nil, err
	}
	for _, p := range files {
		sort.Sort(blocksByStart(p.Blocks))
	}
	return files, nil
}

type blocksByStart []ProfileBlock

func (b blocksByStart) Len() int      { return len(b) }
func (b blocksByStart) Swap(i, j int) { b[i], b[j] = b[j], b[i] }
func (b blocksByStart) Less(i, j int) bool {
	return b[i].StartLine < b[j].StartLine || b[i].StartLine == b[j].StartLine && b[i].StartCol < b[j].StartCol
}

var lineRe = regexp.MustCompile(`^(.+):([0-9]+).([0-9]+),([0-9]+).([0-9]+) ([0-9]+) ([0-9]+)$`)

func toInt(s string) int {
	i, err := strconv.ParseInt(s, 10, 64)
	if err != nil {
		panic(err)
	}
	return int(i)
}

// Token represents the position in a source file of an opening or closing
// <span> tag. These are used to colorize the source.
type Token struct {
	Pos   int
	Start bool
	Count int
}

// Tokens returns a Profile as a set of Tokens within the provided src.
func (p *Profile) Tokens(src []byte) (tokens []Token) {
	line, col := 1, 1
	for si, bi := 0, 0; si < len(src) && bi < len(p.Blocks); {
		b := p.Blocks[bi]
		if b.StartLine == line && b.StartCol == col {
			tokens = append(tokens, Token{Pos: si, Start: true, Count: b.Count})
		}
		if b.EndLine == line && b.EndCol == col {
			tokens = append(tokens, Token{Pos: si, Start: false})
			bi++
			continue // Don't advance through src; maybe the next block starts here.
		}
		if src[si] == '\n' {
			line++
			col = 0
		}
		col++
		si++
	}
	sort.Sort(tokensByPos(tokens))
	return
}

type tokensByPos []Token

func (t tokensByPos) Len() int      { return len(t) }
func (t tokensByPos) Swap(i, j int) { t[i], t[j] = t[j], t[i] }
func (t tokensByPos) Less(i, j int) bool {
	if t[i].Pos == t[j].Pos {
		return !t[i].Start && t[j].Start
	}
	return t[i].Pos < t[j].Pos
}

// htmlGen generates an HTML coverage report with the provided filename,
// source code, and tokens, and writes it to the given Writer.
func htmlGen(w io.Writer, filename string, src []byte, tokens []Token) error {
	dst := bufio.NewWriter(w)
	fmt.Fprintf(dst, "<h1>%s</h1>\n<pre>", html.EscapeString(filename))
	for i := range src {
		for len(tokens) > 0 && tokens[0].Pos == i {
			t := tokens[0]
			if t.Start {
				color := "#CFC" // Green
				if t.Count == 0 {
					color = "#FCC" // Red
				}
				fmt.Fprintf(dst, `<span style="background: %v" title="%v">`, color, t.Count)
			} else {
				dst.WriteString("</span>")
			}
			tokens = tokens[1:]
		}
		switch b := src[i]; b {
		case '>':
			dst.WriteString("&gt;")
		case '<':
			dst.WriteString("&lt;")
		case '\t':
			dst.WriteString("        ")
		default:
			dst.WriteByte(b)
		}
	}
	dst.WriteString("</pre>\n")
	return dst.Flush()
}

// startBrowser tries to open the URL in a browser
// and returns whether it succeed.
func startBrowser(url string) bool {
	// try to start the browser
	var args []string
	switch runtime.GOOS {
	case "darwin":
		args = []string{"open"}
	case "windows":
		args = []string{"cmd", "/c", "start"}
	default:
		args = []string{"xdg-open"}
	}
	cmd := exec.Command(args[0], append(args[1:], url)...)
	return cmd.Start() == nil
}
