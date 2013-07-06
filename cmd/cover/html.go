// Copyright 2013 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

package main

import (
	"bufio"
	"bytes"
	"fmt"
	"go/build"
	"html/template"
	"io"
	"io/ioutil"
	"math"
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

	var files []*templateFile

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
		var buf bytes.Buffer
		err = htmlGen(&buf, src, profile.Tokens(src))
		if err != nil {
			return err
		}
		files = append(files, &templateFile{
			Name: fn,
			Body: template.HTML(buf.String()),
		})
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
	err = htmlTemplate.Execute(out, templateData{Files: files})
	if err == nil {
		err = out.Close()
	}
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
	Norm  float64 // count normalized to 0-1
}

// Tokens returns a Profile as a set of Tokens within the provided src.
func (p *Profile) Tokens(src []byte) (tokens []Token) {
	// Find maximum counts.
	max := 0
	for _, b := range p.Blocks {
		if b.Count > max {
			max = b.Count
		}
	}
	// Divisor for normalization.
	divisor := math.Log(float64(max + 1))

	// tok returns a Token, populating the Norm field with a normalized Count.
	tok := func(pos int, start bool, count int) Token {
		t := Token{Pos: pos, Start: start, Count: count}
		if !start || count == 0 {
			return t
		}
		if max == 1 {
			t.Norm = 0.4 // "set" mode; use pale color
		} else {
			t.Norm = math.Log(float64(count+1)) / divisor
		}
		return t
	}

	line, col := 1, 2
	for si, bi := 0, 0; si < len(src) && bi < len(p.Blocks); {
		b := p.Blocks[bi]
		if b.StartLine == line && b.StartCol == col {
			tokens = append(tokens, tok(si, true, b.Count))
		}
		if b.EndLine == line && b.EndCol == col {
			tokens = append(tokens, tok(si, false, 0))
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
func htmlGen(w io.Writer, src []byte, tokens []Token) error {
	dst := bufio.NewWriter(w)
	for i := range src {
		for len(tokens) > 0 && tokens[0].Pos == i {
			t := tokens[0]
			if t.Start {
				n := 0
				if t.Count > 0 {
					n = int(math.Floor(t.Norm*10)) + 1
				}
				fmt.Fprintf(dst, `<span class="cov%v" title="%v">`, n, t.Count)
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
		case '&':
			dst.WriteString("&amp;")
		case '\t':
			dst.WriteString("        ")
		default:
			dst.WriteByte(b)
		}
	}
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

// rgb returns an rgb value for the specified coverage value
// between 0 (no coverage) and 11 (max coverage).
func rgb(n int) string {
	if n == 0 {
		return "rgb(255, 0, 0)" // Red
	}
	// Gradient from pale green (low count)
	// to bright green (high count)
	r := 240 - 22*(n-1)
	b := 170 + r/3
	return fmt.Sprintf("rgb(%v, 255, %v)", r, b)
}

// colors generates the CSS rules for coverage colors.
func colors() template.CSS {
	var buf bytes.Buffer
	for i := 0; i < 12; i++ {
		fmt.Fprintf(&buf, ".cov%v { color: %v }\n", i, rgb(i))
	}
	return template.CSS(buf.String())
}

var htmlTemplate = template.Must(template.New("html").Funcs(template.FuncMap{
	"colors": colors,
}).Parse(tmplHTML))

type templateData struct {
	Files []*templateFile
}

type templateFile struct {
	Name string
	Body template.HTML
}

const tmplHTML = `
<!DOCTYPE html>
<html>
	<head>
		<style>
			body { background: black; color: white; }
			{{colors}}
		</style>
	</head>
	<body>
		<div id="nav">
			<select id="files">
			{{range $i, $f := .Files}}
			<option value="file{{$i}}">{{$f.Name}}</option>
			{{end}}
			</select>
		</div>
		{{range $i, $f := .Files}}
		<pre class="file" id="file{{$i}}" {{if $i}}style="display: none"{{end}}>{{$f.Body}}</pre>
		{{end}}
	</body>
	<script>
	(function() {
		var files = document.getElementById('files');
		var visible = document.getElementById('file0');
		files.addEventListener('change', onChange, false);
		function onChange() {
			visible.style.display = 'none';
			visible = document.getElementById(files.value);
			visible.style.display = 'block';
		}
	})();
	</script>
</html>
`
