// Copyright 2013 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

// Package godoc is a work-in-progress (2013-07-17) package to
// begin splitting up the godoc binary into multiple pieces.
//
// This package comment will evolve over time as this package splits
// into smaller pieces.
package godoc

import (
	"bytes"
	"fmt"
	"go/ast"
	"go/doc"
	"go/format"
	"go/printer"
	"go/token"
	"io"
	"log"
	"os"
	pathpkg "path"
	"regexp"
	"strconv"
	"strings"
	"text/template"
	"time"
	"unicode"
	"unicode/utf8"
)

// Fake relative package path for built-ins. Documentation for all globals
// (not just exported ones) will be shown for packages in this directory.
const builtinPkgPath = "builtin"

// FuncMap defines template functions used in godoc templates.
//
// Convention: template function names ending in "_html" or "_url" produce
//             HTML- or URL-escaped strings; all other function results may
//             require explicit escaping in the template.
func (p *Presentation) FuncMap() template.FuncMap {
	p.initFuncMapOnce.Do(p.initFuncMap)
	return p.funcMap
}

func (p *Presentation) TemplateFuncs() template.FuncMap {
	p.initFuncMapOnce.Do(p.initFuncMap)
	return p.templateFuncs
}

func (p *Presentation) initFuncMap() {
	if p.Corpus == nil {
		panic("nil Presentation.Corpus")
	}
	p.templateFuncs = template.FuncMap{
		"code": p.code,
	}
	p.funcMap = template.FuncMap{
		// various helpers
		"filename": filenameFunc,
		"repeat":   strings.Repeat,

		// access to FileInfos (directory listings)
		"fileInfoName": fileInfoNameFunc,
		"fileInfoTime": fileInfoTimeFunc,

		// access to search result information
		"infoKind_html":    infoKind_htmlFunc,
		"infoLine":         p.infoLineFunc,
		"infoSnippet_html": p.infoSnippet_htmlFunc,

		// formatting of AST nodes
		"node":         p.nodeFunc,
		"node_html":    p.node_htmlFunc,
		"comment_html": comment_htmlFunc,
		"comment_text": comment_textFunc,

		// support for URL attributes
		"pkgLink":     pkgLinkFunc,
		"srcLink":     srcLinkFunc,
		"posLink_url": newPosLink_urlFunc(srcPosLinkFunc),
		"docLink":     docLinkFunc,
		"queryLink":   queryLinkFunc,

		// formatting of Examples
		"example_html":   p.example_htmlFunc,
		"example_text":   p.example_textFunc,
		"example_name":   p.example_nameFunc,
		"example_suffix": p.example_suffixFunc,

		// formatting of Notes
		"noteTitle": noteTitle,
	}
	if p.URLForSrc != nil {
		p.funcMap["srcLink"] = p.URLForSrc
	}
	if p.URLForSrcPos != nil {
		p.funcMap["posLink_url"] = newPosLink_urlFunc(p.URLForSrcPos)
	}
	if p.URLForSrcQuery != nil {
		p.funcMap["queryLink"] = p.URLForSrcQuery
	}
}

func filenameFunc(path string) string {
	_, localname := pathpkg.Split(path)
	return localname
}

func fileInfoNameFunc(fi os.FileInfo) string {
	name := fi.Name()
	if fi.IsDir() {
		name += "/"
	}
	return name
}

func fileInfoTimeFunc(fi os.FileInfo) string {
	if t := fi.ModTime(); t.Unix() != 0 {
		return t.Local().String()
	}
	return "" // don't return epoch if time is obviously not set
}

// The strings in infoKinds must be properly html-escaped.
var infoKinds = [nKinds]string{
	PackageClause: "package&nbsp;clause",
	ImportDecl:    "import&nbsp;decl",
	ConstDecl:     "const&nbsp;decl",
	TypeDecl:      "type&nbsp;decl",
	VarDecl:       "var&nbsp;decl",
	FuncDecl:      "func&nbsp;decl",
	MethodDecl:    "method&nbsp;decl",
	Use:           "use",
}

func infoKind_htmlFunc(info SpotInfo) string {
	return infoKinds[info.Kind()] // infoKind entries are html-escaped
}

func (p *Presentation) infoLineFunc(info SpotInfo) int {
	line := info.Lori()
	if info.IsIndex() {
		index, _ := p.Corpus.searchIndex.Get()
		if index != nil {
			line = index.(*Index).Snippet(line).Line
		} else {
			// no line information available because
			// we don't have an index - this should
			// never happen; be conservative and don't
			// crash
			line = 0
		}
	}
	return line
}

func (p *Presentation) infoSnippet_htmlFunc(info SpotInfo) string {
	if info.IsIndex() {
		index, _ := p.Corpus.searchIndex.Get()
		// Snippet.Text was HTML-escaped when it was generated
		return index.(*Index).Snippet(info.Lori()).Text
	}
	return `<span class="alert">no snippet text available</span>`
}

func (p *Presentation) nodeFunc(info *PageInfo, node interface{}) string {
	var buf bytes.Buffer
	p.writeNode(&buf, info.FSet, node)
	return buf.String()
}

func (p *Presentation) node_htmlFunc(info *PageInfo, node interface{}, linkify bool) string {
	var buf1 bytes.Buffer
	p.writeNode(&buf1, info.FSet, node)

	var buf2 bytes.Buffer
	if n, _ := node.(ast.Node); n != nil && linkify && p.DeclLinks {
		LinkifyText(&buf2, buf1.Bytes(), n)
	} else {
		FormatText(&buf2, buf1.Bytes(), -1, true, "", nil)
	}

	return buf2.String()
}

func comment_htmlFunc(comment string) string {
	var buf bytes.Buffer
	// TODO(gri) Provide list of words (e.g. function parameters)
	//           to be emphasized by ToHTML.
	doc.ToHTML(&buf, comment, nil) // does html-escaping
	return buf.String()
}

// punchCardWidth is the number of columns of fixed-width
// characters to assume when wrapping text.  Very few people
// use terminals or cards smaller than 80 characters, so 80 it is.
// We do not try to sniff the environment or the tty to adapt to
// the situation; instead, by using a constant we make sure that
// godoc always produces the same output regardless of context,
// a consistency that is lost otherwise.  For example, if we sniffed
// the environment or tty, then http://golang.org/pkg/math/?m=text
// would depend on the width of the terminal where godoc started,
// which is clearly bogus.  More generally, the Unix tools that behave
// differently when writing to a tty than when writing to a file have
// a history of causing confusion (compare `ls` and `ls | cat`), and we
// want to avoid that mistake here.
const punchCardWidth = 80

func comment_textFunc(comment, indent, preIndent string) string {
	var buf bytes.Buffer
	doc.ToText(&buf, comment, indent, preIndent, punchCardWidth-2*len(indent))
	return buf.String()
}

type PageInfo struct {
	Dirname string // directory containing the package
	Err     error  // error or nil

	// package info
	FSet     *token.FileSet         // nil if no package documentation
	PDoc     *doc.Package           // nil if no package documentation
	Examples []*doc.Example         // nil if no example code
	Notes    map[string][]*doc.Note // nil if no package Notes
	PAst     *ast.File              // nil if no AST with package exports
	IsMain   bool                   // true for package main

	// directory info
	Dirs    *DirList  // nil if no directory information
	DirTime time.Time // directory time stamp
	DirFlat bool      // if set, show directory in a flat (non-indented) manner
}

func (info *PageInfo) IsEmpty() bool {
	return info.Err != nil || info.PAst == nil && info.PDoc == nil && info.Dirs == nil
}

func pkgLinkFunc(path string) string {
	relpath := path[1:]
	// because of the irregular mapping under goroot
	// we need to correct certain relative paths
	relpath = strings.TrimPrefix(relpath, "src/pkg/")
	return "pkg/" + relpath // remove trailing '/' for relative URL
}

func newPosLink_urlFunc(srcPosLinkFunc func(s string, line, low, high int) string) func(info *PageInfo, n interface{}) string {
	// n must be an ast.Node or a *doc.Note
	return func(info *PageInfo, n interface{}) string {
		var pos, end token.Pos

		switch n := n.(type) {
		case ast.Node:
			pos = n.Pos()
			end = n.End()
		case *doc.Note:
			pos = n.Pos
			end = n.End
		default:
			panic(fmt.Sprintf("wrong type for posLink_url template formatter: %T", n))
		}

		var relpath string
		var line int
		var low, high int // selection offset range

		if pos.IsValid() {
			p := info.FSet.Position(pos)
			relpath = p.Filename
			line = p.Line
			low = p.Offset
		}
		if end.IsValid() {
			high = info.FSet.Position(end).Offset
		}

		return srcPosLinkFunc(relpath, line, low, high)
	}
}

func srcPosLinkFunc(s string, line, low, high int) string {
	var buf bytes.Buffer
	template.HTMLEscape(&buf, []byte(s))
	// selection ranges are of form "s=low:high"
	if low < high {
		fmt.Fprintf(&buf, "?s=%d:%d", low, high) // no need for URL escaping
		// if we have a selection, position the page
		// such that the selection is a bit below the top
		line -= 10
		if line < 1 {
			line = 1
		}
	}
	// line id's in html-printed source are of the
	// form "L%d" where %d stands for the line number
	if line > 0 {
		fmt.Fprintf(&buf, "#L%d", line) // no need for URL escaping
	}
	return buf.String()
}

func srcLinkFunc(s string) string {
	return pathpkg.Clean("/" + s)
}

// queryLinkFunc returns a URL for a line in a source file with a highlighted
// query term.
// s is expected to be a path to a source file.
// query is expected to be a string that has already been appropriately escaped
// for use in a URL query.
func queryLinkFunc(s, query string, line int) string {
	url := pathpkg.Clean("/"+s) + "?h=" + query
	if line > 0 {
		url += "#L" + strconv.Itoa(line)
	}
	return url
}

func docLinkFunc(s string, ident string) string {
	s = strings.TrimPrefix(s, "/src")
	return pathpkg.Clean("/"+s) + "/#" + ident
}

func (p *Presentation) example_textFunc(info *PageInfo, funcName, indent string) string {
	if !p.ShowExamples {
		return ""
	}

	var buf bytes.Buffer
	first := true
	for _, eg := range info.Examples {
		name := stripExampleSuffix(eg.Name)
		if name != funcName {
			continue
		}

		if !first {
			buf.WriteString("\n")
		}
		first = false

		// print code
		cnode := &printer.CommentedNode{Node: eg.Code, Comments: eg.Comments}
		var buf1 bytes.Buffer
		p.writeNode(&buf1, info.FSet, cnode)
		code := buf1.String()
		// Additional formatting if this is a function body.
		if n := len(code); n >= 2 && code[0] == '{' && code[n-1] == '}' {
			// remove surrounding braces
			code = code[1 : n-1]
			// unindent
			code = strings.Replace(code, "\n    ", "\n", -1)
		}
		code = strings.Trim(code, "\n")
		code = strings.Replace(code, "\n", "\n\t", -1)

		buf.WriteString(indent)
		buf.WriteString("Example:\n\t")
		buf.WriteString(code)
		buf.WriteString("\n")
	}
	return buf.String()
}

func (p *Presentation) example_htmlFunc(info *PageInfo, funcName string) string {
	var buf bytes.Buffer
	for _, eg := range info.Examples {
		name := stripExampleSuffix(eg.Name)

		if name != funcName {
			continue
		}

		// print code
		cnode := &printer.CommentedNode{Node: eg.Code, Comments: eg.Comments}
		code := p.node_htmlFunc(info, cnode, true)
		out := eg.Output
		wholeFile := true

		// Additional formatting if this is a function body.
		if n := len(code); n >= 2 && code[0] == '{' && code[n-1] == '}' {
			wholeFile = false
			// remove surrounding braces
			code = code[1 : n-1]
			// unindent
			code = strings.Replace(code, "\n    ", "\n", -1)
			// remove output comment
			if loc := exampleOutputRx.FindStringIndex(code); loc != nil {
				code = strings.TrimSpace(code[:loc[0]])
			}
		}

		// Write out the playground code in standard Go style
		// (use tabs, no comment highlight, etc).
		play := ""
		if eg.Play != nil && p.ShowPlayground {
			var buf bytes.Buffer
			if err := format.Node(&buf, info.FSet, eg.Play); err != nil {
				log.Print(err)
			} else {
				play = buf.String()
			}
		}

		// Drop output, as the output comment will appear in the code.
		if wholeFile && play == "" {
			out = ""
		}

		if p.ExampleHTML == nil {
			out = ""
			return ""
		}

		err := p.ExampleHTML.Execute(&buf, struct {
			Name, Doc, Code, Play, Output string
		}{eg.Name, eg.Doc, code, play, out})
		if err != nil {
			log.Print(err)
		}
	}
	return buf.String()
}

// example_nameFunc takes an example function name and returns its display
// name. For example, "Foo_Bar_quux" becomes "Foo.Bar (Quux)".
func (p *Presentation) example_nameFunc(s string) string {
	name, suffix := splitExampleName(s)
	// replace _ with . for method names
	name = strings.Replace(name, "_", ".", 1)
	// use "Package" if no name provided
	if name == "" {
		name = "Package"
	}
	return name + suffix
}

// example_suffixFunc takes an example function name and returns its suffix in
// parenthesized form. For example, "Foo_Bar_quux" becomes " (Quux)".
func (p *Presentation) example_suffixFunc(name string) string {
	_, suffix := splitExampleName(name)
	return suffix
}

func noteTitle(note string) string {
	return strings.Title(strings.ToLower(note))
}

func startsWithUppercase(s string) bool {
	r, _ := utf8.DecodeRuneInString(s)
	return unicode.IsUpper(r)
}

var exampleOutputRx = regexp.MustCompile(`(?i)//[[:space:]]*output:`)

// stripExampleSuffix strips lowercase braz in Foo_braz or Foo_Bar_braz from name
// while keeping uppercase Braz in Foo_Braz.
func stripExampleSuffix(name string) string {
	if i := strings.LastIndex(name, "_"); i != -1 {
		if i < len(name)-1 && !startsWithUppercase(name[i+1:]) {
			name = name[:i]
		}
	}
	return name
}

func splitExampleName(s string) (name, suffix string) {
	i := strings.LastIndex(s, "_")
	if 0 <= i && i < len(s)-1 && !startsWithUppercase(s[i+1:]) {
		name = s[:i]
		suffix = " (" + strings.Title(s[i+1:]) + ")"
		return
	}
	name = s
	return
}

// Write an AST node to w.
func (p *Presentation) writeNode(w io.Writer, fset *token.FileSet, x interface{}) {
	// convert trailing tabs into spaces using a tconv filter
	// to ensure a good outcome in most browsers (there may still
	// be tabs in comments and strings, but converting those into
	// the right number of spaces is much harder)
	//
	// TODO(gri) rethink printer flags - perhaps tconv can be eliminated
	//           with an another printer mode (which is more efficiently
	//           implemented in the printer than here with another layer)
	mode := printer.TabIndent | printer.UseSpaces
	err := (&printer.Config{Mode: mode, Tabwidth: p.TabWidth}).Fprint(&tconv{p: p, output: w}, fset, x)
	if err != nil {
		log.Print(err)
	}
}

// WriteNote writes x to w.
func (p *Presentation) WriteNode(w io.Writer, fset *token.FileSet, x interface{}) {
	p.writeNode(w, fset, x)
}
