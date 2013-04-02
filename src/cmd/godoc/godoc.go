// Copyright 2009 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

package main

import (
	"bytes"
	"encoding/json"
	"flag"
	"fmt"
	"go/ast"
	"go/build"
	"go/doc"
	"go/format"
	"go/printer"
	"go/token"
	htmlpkg "html"
	"io"
	"io/ioutil"
	"log"
	"net/http"
	"net/url"
	"os"
	pathpkg "path"
	"path/filepath"
	"regexp"
	"runtime"
	"sort"
	"strings"
	"text/template"
	"time"
	"unicode"
	"unicode/utf8"
)

// ----------------------------------------------------------------------------
// Globals

type delayTime struct {
	RWValue
}

func (dt *delayTime) backoff(max time.Duration) {
	dt.mutex.Lock()
	v := dt.value.(time.Duration) * 2
	if v > max {
		v = max
	}
	dt.value = v
	// don't change dt.timestamp - calling backoff indicates an error condition
	dt.mutex.Unlock()
}

var (
	verbose = flag.Bool("v", false, "verbose mode")

	// file system roots
	// TODO(gri) consider the invariant that goroot always end in '/'
	goroot  = flag.String("goroot", runtime.GOROOT(), "Go root directory")
	testDir = flag.String("testdir", "", "Go root subdirectory - for testing only (faster startups)")

	// layout control
	tabwidth       = flag.Int("tabwidth", 4, "tab width")
	showTimestamps = flag.Bool("timestamps", false, "show timestamps with directory listings")
	templateDir    = flag.String("templates", "", "directory containing alternate template files")
	showPlayground = flag.Bool("play", false, "enable playground in web interface")
	showExamples   = flag.Bool("ex", false, "show examples in command line mode")
	declLinks      = flag.Bool("links", true, "link identifiers to their declarations")

	// search index
	indexEnabled = flag.Bool("index", false, "enable search index")
	indexFiles   = flag.String("index_files", "", "glob pattern specifying index files;"+
		"if not empty, the index is read from these files in sorted order")
	maxResults    = flag.Int("maxresults", 10000, "maximum number of full text search results shown")
	indexThrottle = flag.Float64("index_throttle", 0.75, "index throttle value; 0.0 = no time allocated, 1.0 = full throttle")

	// file system information
	fsTree      RWValue // *Directory tree of packages, updated with each sync (but sync code is removed now)
	fsModified  RWValue // timestamp of last call to invalidateIndex
	docMetadata RWValue // mapping from paths to *Metadata

	// http handlers
	fileServer http.Handler // default file server
	cmdHandler docServer
	pkgHandler docServer

	// source code notes
	notes = flag.String("notes", "BUG", "regular expression matching note markers to show")
)

func initHandlers() {
	fileServer = http.FileServer(&httpFS{fs})
	cmdHandler = docServer{"/cmd/", "/src/cmd"}
	pkgHandler = docServer{"/pkg/", "/src/pkg"}
}

func registerPublicHandlers(mux *http.ServeMux) {
	mux.Handle(cmdHandler.pattern, &cmdHandler)
	mux.Handle(pkgHandler.pattern, &pkgHandler)
	mux.HandleFunc("/doc/codewalk/", codewalk)
	mux.Handle("/doc/play/", fileServer)
	mux.HandleFunc("/search", search)
	mux.Handle("/robots.txt", fileServer)
	mux.HandleFunc("/opensearch.xml", serveSearchDesc)
	mux.HandleFunc("/", serveFile)
}

func initFSTree() {
	dir := newDirectory(pathpkg.Join("/", *testDir), -1)
	if dir == nil {
		log.Println("Warning: FSTree is nil")
		return
	}
	fsTree.set(dir)
	invalidateIndex()
}

// ----------------------------------------------------------------------------
// Tab conversion

var spaces = []byte("                                ") // 32 spaces seems like a good number

const (
	indenting = iota
	collecting
)

// A tconv is an io.Writer filter for converting leading tabs into spaces.
type tconv struct {
	output io.Writer
	state  int // indenting or collecting
	indent int // valid if state == indenting
}

func (p *tconv) writeIndent() (err error) {
	i := p.indent
	for i >= len(spaces) {
		i -= len(spaces)
		if _, err = p.output.Write(spaces); err != nil {
			return
		}
	}
	// i < len(spaces)
	if i > 0 {
		_, err = p.output.Write(spaces[0:i])
	}
	return
}

func (p *tconv) Write(data []byte) (n int, err error) {
	if len(data) == 0 {
		return
	}
	pos := 0 // valid if p.state == collecting
	var b byte
	for n, b = range data {
		switch p.state {
		case indenting:
			switch b {
			case '\t':
				p.indent += *tabwidth
			case '\n':
				p.indent = 0
				if _, err = p.output.Write(data[n : n+1]); err != nil {
					return
				}
			case ' ':
				p.indent++
			default:
				p.state = collecting
				pos = n
				if err = p.writeIndent(); err != nil {
					return
				}
			}
		case collecting:
			if b == '\n' {
				p.state = indenting
				p.indent = 0
				if _, err = p.output.Write(data[pos : n+1]); err != nil {
					return
				}
			}
		}
	}
	n = len(data)
	if pos < n && p.state == collecting {
		_, err = p.output.Write(data[pos:])
	}
	return
}

// ----------------------------------------------------------------------------
// Templates

// Write an AST node to w.
func writeNode(w io.Writer, fset *token.FileSet, x interface{}) {
	// convert trailing tabs into spaces using a tconv filter
	// to ensure a good outcome in most browsers (there may still
	// be tabs in comments and strings, but converting those into
	// the right number of spaces is much harder)
	//
	// TODO(gri) rethink printer flags - perhaps tconv can be eliminated
	//           with an another printer mode (which is more efficiently
	//           implemented in the printer than here with another layer)
	mode := printer.TabIndent | printer.UseSpaces
	err := (&printer.Config{Mode: mode, Tabwidth: *tabwidth}).Fprint(&tconv{output: w}, fset, x)
	if err != nil {
		log.Print(err)
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

func infoLineFunc(info SpotInfo) int {
	line := info.Lori()
	if info.IsIndex() {
		index, _ := searchIndex.get()
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

func infoSnippet_htmlFunc(info SpotInfo) string {
	if info.IsIndex() {
		index, _ := searchIndex.get()
		// Snippet.Text was HTML-escaped when it was generated
		return index.(*Index).Snippet(info.Lori()).Text
	}
	return `<span class="alert">no snippet text available</span>`
}

func nodeFunc(info *PageInfo, node interface{}) string {
	var buf bytes.Buffer
	writeNode(&buf, info.FSet, node)
	return buf.String()
}

func node_htmlFunc(info *PageInfo, node interface{}, linkify bool) string {
	var buf1 bytes.Buffer
	writeNode(&buf1, info.FSet, node)

	var buf2 bytes.Buffer
	if n, _ := node.(ast.Node); n != nil && linkify && *declLinks {
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

func example_textFunc(info *PageInfo, funcName, indent string) string {
	if !*showExamples {
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
		writeNode(&buf1, info.FSet, cnode)
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

func example_htmlFunc(info *PageInfo, funcName string) string {
	var buf bytes.Buffer
	for _, eg := range info.Examples {
		name := stripExampleSuffix(eg.Name)

		if name != funcName {
			continue
		}

		// print code
		cnode := &printer.CommentedNode{Node: eg.Code, Comments: eg.Comments}
		code := node_htmlFunc(info, cnode, true)
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
		if eg.Play != nil && *showPlayground {
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

		err := exampleHTML.Execute(&buf, struct {
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
func example_nameFunc(s string) string {
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
func example_suffixFunc(name string) string {
	_, suffix := splitExampleName(name)
	return suffix
}

func noteTitle(note string) string {
	return strings.Title(strings.ToLower(note))
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

func pkgLinkFunc(path string) string {
	relpath := path[1:]
	// because of the irregular mapping under goroot
	// we need to correct certain relative paths
	relpath = strings.TrimPrefix(relpath, "src/pkg/")
	return pkgHandler.pattern[1:] + relpath // remove trailing '/' for relative URL
}

// n must be an ast.Node or a *doc.Note
func posLink_urlFunc(info *PageInfo, n interface{}) string {
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

	var buf bytes.Buffer
	template.HTMLEscape(&buf, []byte(relpath))
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

// fmap describes the template functions installed with all godoc templates.
// Convention: template function names ending in "_html" or "_url" produce
//             HTML- or URL-escaped strings; all other function results may
//             require explicit escaping in the template.
var fmap = template.FuncMap{
	// various helpers
	"filename": filenameFunc,
	"repeat":   strings.Repeat,

	// access to FileInfos (directory listings)
	"fileInfoName": fileInfoNameFunc,
	"fileInfoTime": fileInfoTimeFunc,

	// access to search result information
	"infoKind_html":    infoKind_htmlFunc,
	"infoLine":         infoLineFunc,
	"infoSnippet_html": infoSnippet_htmlFunc,

	// formatting of AST nodes
	"node":         nodeFunc,
	"node_html":    node_htmlFunc,
	"comment_html": comment_htmlFunc,
	"comment_text": comment_textFunc,

	// support for URL attributes
	"pkgLink":     pkgLinkFunc,
	"srcLink":     srcLinkFunc,
	"posLink_url": posLink_urlFunc,

	// formatting of Examples
	"example_html":   example_htmlFunc,
	"example_text":   example_textFunc,
	"example_name":   example_nameFunc,
	"example_suffix": example_suffixFunc,

	// formatting of Notes
	"noteTitle": noteTitle,
}

func readTemplate(name string) *template.Template {
	path := "lib/godoc/" + name

	// use underlying file system fs to read the template file
	// (cannot use template ParseFile functions directly)
	data, err := ReadFile(fs, path)
	if err != nil {
		log.Fatal("readTemplate: ", err)
	}
	// be explicit with errors (for app engine use)
	t, err := template.New(name).Funcs(fmap).Parse(string(data))
	if err != nil {
		log.Fatal("readTemplate: ", err)
	}
	return t
}

var (
	codewalkHTML,
	codewalkdirHTML,
	dirlistHTML,
	errorHTML,
	exampleHTML,
	godocHTML,
	packageHTML,
	packageText,
	searchHTML,
	searchText,
	searchDescXML *template.Template
)

func readTemplates() {
	// have to delay until after flags processing since paths depend on goroot
	codewalkHTML = readTemplate("codewalk.html")
	codewalkdirHTML = readTemplate("codewalkdir.html")
	dirlistHTML = readTemplate("dirlist.html")
	errorHTML = readTemplate("error.html")
	exampleHTML = readTemplate("example.html")
	godocHTML = readTemplate("godoc.html")
	packageHTML = readTemplate("package.html")
	packageText = readTemplate("package.txt")
	searchHTML = readTemplate("search.html")
	searchText = readTemplate("search.txt")
	searchDescXML = readTemplate("opensearch.xml")
}

// ----------------------------------------------------------------------------
// Generic HTML wrapper

// Page describes the contents of the top-level godoc webpage.
type Page struct {
	Title    string
	Tabtitle string
	Subtitle string
	Query    string
	Body     []byte

	// filled in by servePage
	SearchBox  bool
	Playground bool
	Version    string
}

func servePage(w http.ResponseWriter, page Page) {
	if page.Tabtitle == "" {
		page.Tabtitle = page.Title
	}
	page.SearchBox = *indexEnabled
	page.Playground = *showPlayground
	page.Version = runtime.Version()
	if err := godocHTML.Execute(w, page); err != nil {
		log.Printf("godocHTML.Execute: %s", err)
	}
}

func serveText(w http.ResponseWriter, text []byte) {
	w.Header().Set("Content-Type", "text/plain; charset=utf-8")
	w.Write(text)
}

// ----------------------------------------------------------------------------
// Files

var (
	doctype   = []byte("<!DOCTYPE ")
	jsonStart = []byte("<!--{")
	jsonEnd   = []byte("}-->")
)

func serveHTMLDoc(w http.ResponseWriter, r *http.Request, abspath, relpath string) {
	// get HTML body contents
	src, err := ReadFile(fs, abspath)
	if err != nil {
		log.Printf("ReadFile: %s", err)
		serveError(w, r, relpath, err)
		return
	}

	// if it begins with "<!DOCTYPE " assume it is standalone
	// html that doesn't need the template wrapping.
	if bytes.HasPrefix(src, doctype) {
		w.Write(src)
		return
	}

	// if it begins with a JSON blob, read in the metadata.
	meta, src, err := extractMetadata(src)
	if err != nil {
		log.Printf("decoding metadata %s: %v", relpath, err)
	}

	// evaluate as template if indicated
	if meta.Template {
		tmpl, err := template.New("main").Funcs(templateFuncs).Parse(string(src))
		if err != nil {
			log.Printf("parsing template %s: %v", relpath, err)
			serveError(w, r, relpath, err)
			return
		}
		var buf bytes.Buffer
		if err := tmpl.Execute(&buf, nil); err != nil {
			log.Printf("executing template %s: %v", relpath, err)
			serveError(w, r, relpath, err)
			return
		}
		src = buf.Bytes()
	}

	// if it's the language spec, add tags to EBNF productions
	if strings.HasSuffix(abspath, "go_spec.html") {
		var buf bytes.Buffer
		Linkify(&buf, src)
		src = buf.Bytes()
	}

	servePage(w, Page{
		Title:    meta.Title,
		Subtitle: meta.Subtitle,
		Body:     src,
	})
}

func applyTemplate(t *template.Template, name string, data interface{}) []byte {
	var buf bytes.Buffer
	if err := t.Execute(&buf, data); err != nil {
		log.Printf("%s.Execute: %s", name, err)
	}
	return buf.Bytes()
}

func redirect(w http.ResponseWriter, r *http.Request) (redirected bool) {
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

func redirectFile(w http.ResponseWriter, r *http.Request) (redirected bool) {
	c := pathpkg.Clean(r.URL.Path)
	c = strings.TrimRight(c, "/")
	if r.URL.Path != c {
		url := *r.URL
		url.Path = c
		http.Redirect(w, r, url.String(), http.StatusMovedPermanently)
		redirected = true
	}
	return
}

func serveTextFile(w http.ResponseWriter, r *http.Request, abspath, relpath, title string) {
	src, err := ReadFile(fs, abspath)
	if err != nil {
		log.Printf("ReadFile: %s", err)
		serveError(w, r, relpath, err)
		return
	}

	if r.FormValue("m") == "text" {
		serveText(w, src)
		return
	}

	var buf bytes.Buffer
	buf.WriteString("<pre>")
	FormatText(&buf, src, 1, pathpkg.Ext(abspath) == ".go", r.FormValue("h"), rangeSelection(r.FormValue("s")))
	buf.WriteString("</pre>")
	fmt.Fprintf(&buf, `<p><a href="/%s?m=text">View as plain text</a></p>`, htmlpkg.EscapeString(relpath))

	servePage(w, Page{
		Title:    title + " " + relpath,
		Tabtitle: relpath,
		Body:     buf.Bytes(),
	})
}

func serveDirectory(w http.ResponseWriter, r *http.Request, abspath, relpath string) {
	if redirect(w, r) {
		return
	}

	list, err := fs.ReadDir(abspath)
	if err != nil {
		serveError(w, r, relpath, err)
		return
	}

	servePage(w, Page{
		Title:    "Directory " + relpath,
		Tabtitle: relpath,
		Body:     applyTemplate(dirlistHTML, "dirlistHTML", list),
	})
}

func serveFile(w http.ResponseWriter, r *http.Request) {
	relpath := r.URL.Path

	// Check to see if we need to redirect or serve another file.
	if m := metadataFor(relpath); m != nil {
		if m.Path != relpath {
			// Redirect to canonical path.
			http.Redirect(w, r, m.Path, http.StatusMovedPermanently)
			return
		}
		// Serve from the actual filesystem path.
		relpath = m.filePath
	}

	abspath := relpath
	relpath = relpath[1:] // strip leading slash

	switch pathpkg.Ext(relpath) {
	case ".html":
		if strings.HasSuffix(relpath, "/index.html") {
			// We'll show index.html for the directory.
			// Use the dir/ version as canonical instead of dir/index.html.
			http.Redirect(w, r, r.URL.Path[0:len(r.URL.Path)-len("index.html")], http.StatusMovedPermanently)
			return
		}
		serveHTMLDoc(w, r, abspath, relpath)
		return

	case ".go":
		serveTextFile(w, r, abspath, relpath, "Source file")
		return
	}

	dir, err := fs.Lstat(abspath)
	if err != nil {
		log.Print(err)
		serveError(w, r, relpath, err)
		return
	}

	if dir != nil && dir.IsDir() {
		if redirect(w, r) {
			return
		}
		if index := pathpkg.Join(abspath, "index.html"); isTextFile(index) {
			serveHTMLDoc(w, r, index, index)
			return
		}
		serveDirectory(w, r, abspath, relpath)
		return
	}

	if isTextFile(abspath) {
		if redirectFile(w, r) {
			return
		}
		serveTextFile(w, r, abspath, relpath, "Text file")
		return
	}

	fileServer.ServeHTTP(w, r)
}

func serveSearchDesc(w http.ResponseWriter, r *http.Request) {
	w.Header().Set("Content-Type", "application/opensearchdescription+xml")
	data := map[string]interface{}{
		"BaseURL": fmt.Sprintf("http://%s", r.Host),
	}
	if err := searchDescXML.Execute(w, &data); err != nil {
		log.Printf("searchDescXML.Execute: %s", err)
	}
}

// ----------------------------------------------------------------------------
// Packages

// Fake relative package path for built-ins. Documentation for all globals
// (not just exported ones) will be shown for packages in this directory.
const builtinPkgPath = "builtin"

type PageInfoMode uint

const (
	noFiltering PageInfoMode = 1 << iota // do not filter exports
	allMethods                           // show all embedded methods
	showSource                           // show source code, do not extract documentation
	noHtml                               // show result in textual form, do not generate HTML
	flatDir                              // show directory in a flat (non-indented) manner
)

// modeNames defines names for each PageInfoMode flag.
var modeNames = map[string]PageInfoMode{
	"all":     noFiltering,
	"methods": allMethods,
	"src":     showSource,
	"text":    noHtml,
	"flat":    flatDir,
}

// getPageInfoMode computes the PageInfoMode flags by analyzing the request
// URL form value "m". It is value is a comma-separated list of mode names
// as defined by modeNames (e.g.: m=src,text).
func getPageInfoMode(r *http.Request) PageInfoMode {
	var mode PageInfoMode
	for _, k := range strings.Split(r.FormValue("m"), ",") {
		if m, found := modeNames[strings.TrimSpace(k)]; found {
			mode |= m
		}
	}
	return adjustPageInfoMode(r, mode)
}

// Specialized versions of godoc may adjust the PageInfoMode by overriding
// this variable.
var adjustPageInfoMode = func(_ *http.Request, mode PageInfoMode) PageInfoMode {
	return mode
}

// remoteSearchURL returns the search URL for a given query as needed by
// remoteSearch. If html is set, an html result is requested; otherwise
// the result is in textual form.
// Adjust this function as necessary if modeNames or FormValue parameters
// change.
func remoteSearchURL(query string, html bool) string {
	s := "/search?m=text&q="
	if html {
		s = "/search?q="
	}
	return s + url.QueryEscape(query)
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

type docServer struct {
	pattern string // url pattern; e.g. "/pkg/"
	fsRoot  string // file system root to which the pattern is mapped
}

// fsReadDir implements ReadDir for the go/build package.
func fsReadDir(dir string) ([]os.FileInfo, error) {
	return fs.ReadDir(filepath.ToSlash(dir))
}

// fsOpenFile implements OpenFile for the go/build package.
func fsOpenFile(name string) (r io.ReadCloser, err error) {
	data, err := ReadFile(fs, filepath.ToSlash(name))
	if err != nil {
		return nil, err
	}
	return ioutil.NopCloser(bytes.NewReader(data)), nil
}

// packageExports is a local implementation of ast.PackageExports
// which correctly updates each package file's comment list.
// (The ast.PackageExports signature is frozen, hence the local
// implementation).
//
func packageExports(fset *token.FileSet, pkg *ast.Package) {
	for _, src := range pkg.Files {
		cmap := ast.NewCommentMap(fset, src, src.Comments)
		ast.FileExports(src)
		src.Comments = cmap.Filter(src).Comments()
	}
}

// addNames adds the names declared by decl to the names set.
// Method names are added in the form ReceiverTypeName_Method.
func addNames(names map[string]bool, decl ast.Decl) {
	switch d := decl.(type) {
	case *ast.FuncDecl:
		name := d.Name.Name
		if d.Recv != nil {
			var typeName string
			switch r := d.Recv.List[0].Type.(type) {
			case *ast.StarExpr:
				typeName = r.X.(*ast.Ident).Name
			case *ast.Ident:
				typeName = r.Name
			}
			name = typeName + "_" + name
		}
		names[name] = true
	case *ast.GenDecl:
		for _, spec := range d.Specs {
			switch s := spec.(type) {
			case *ast.TypeSpec:
				names[s.Name.Name] = true
			case *ast.ValueSpec:
				for _, id := range s.Names {
					names[id.Name] = true
				}
			}
		}
	}
}

// globalNames returns a set of the names declared by all package-level
// declarations. Method names are returned in the form Receiver_Method.
func globalNames(pkg *ast.Package) map[string]bool {
	names := make(map[string]bool)
	for _, file := range pkg.Files {
		for _, decl := range file.Decls {
			addNames(names, decl)
		}
	}
	return names
}

// collectExamples collects examples for pkg from testfiles.
func collectExamples(pkg *ast.Package, testfiles map[string]*ast.File) []*doc.Example {
	var files []*ast.File
	for _, f := range testfiles {
		files = append(files, f)
	}

	var examples []*doc.Example
	globals := globalNames(pkg)
	for _, e := range doc.Examples(files...) {
		name := stripExampleSuffix(e.Name)
		if name == "" || globals[name] {
			examples = append(examples, e)
		} else {
			log.Printf("skipping example 'Example%s' because '%s' is not a known function or type", e.Name, e.Name)
		}
	}

	return examples
}

// poorMansImporter returns a (dummy) package object named
// by the last path component of the provided package path
// (as is the convention for packages). This is sufficient
// to resolve package identifiers without doing an actual
// import. It never returns an error.
//
func poorMansImporter(imports map[string]*ast.Object, path string) (*ast.Object, error) {
	pkg := imports[path]
	if pkg == nil {
		// note that strings.LastIndex returns -1 if there is no "/"
		pkg = ast.NewObj(ast.Pkg, path[strings.LastIndex(path, "/")+1:])
		pkg.Data = ast.NewScope(nil) // required by ast.NewPackage for dot-import
		imports[path] = pkg
	}
	return pkg, nil
}

// getPageInfo returns the PageInfo for a package directory abspath. If the
// parameter genAST is set, an AST containing only the package exports is
// computed (PageInfo.PAst), otherwise package documentation (PageInfo.Doc)
// is extracted from the AST. If there is no corresponding package in the
// directory, PageInfo.PAst and PageInfo.PDoc are nil. If there are no sub-
// directories, PageInfo.Dirs is nil. If an error occurred, PageInfo.Err is
// set to the respective error but the error is not logged.
//
func (h *docServer) getPageInfo(abspath, relpath string, mode PageInfoMode) *PageInfo {
	info := &PageInfo{Dirname: abspath}

	// Restrict to the package files that would be used when building
	// the package on this system.  This makes sure that if there are
	// separate implementations for, say, Windows vs Unix, we don't
	// jumble them all together.
	// Note: Uses current binary's GOOS/GOARCH.
	// To use different pair, such as if we allowed the user to choose,
	// set ctxt.GOOS and ctxt.GOARCH before calling ctxt.ImportDir.
	ctxt := build.Default
	ctxt.IsAbsPath = pathpkg.IsAbs
	ctxt.ReadDir = fsReadDir
	ctxt.OpenFile = fsOpenFile
	pkginfo, err := ctxt.ImportDir(abspath, 0)
	// continue if there are no Go source files; we still want the directory info
	if _, nogo := err.(*build.NoGoError); err != nil && !nogo {
		info.Err = err
		return info
	}

	// collect package files
	pkgname := pkginfo.Name
	pkgfiles := append(pkginfo.GoFiles, pkginfo.CgoFiles...)
	if len(pkgfiles) == 0 {
		// Commands written in C have no .go files in the build.
		// Instead, documentation may be found in an ignored file.
		// The file may be ignored via an explicit +build ignore
		// constraint (recommended), or by defining the package
		// documentation (historic).
		pkgname = "main" // assume package main since pkginfo.Name == ""
		pkgfiles = pkginfo.IgnoredGoFiles
	}

	// get package information, if any
	if len(pkgfiles) > 0 {
		// build package AST
		fset := token.NewFileSet()
		files, err := parseFiles(fset, abspath, pkgfiles)
		if err != nil {
			info.Err = err
			return info
		}

		// ignore any errors - they are due to unresolved identifiers
		pkg, _ := ast.NewPackage(fset, files, poorMansImporter, nil)

		// extract package documentation
		info.FSet = fset
		if mode&showSource == 0 {
			// show extracted documentation
			var m doc.Mode
			if mode&noFiltering != 0 {
				m = doc.AllDecls
			}
			if mode&allMethods != 0 {
				m |= doc.AllMethods
			}
			info.PDoc = doc.New(pkg, pathpkg.Clean(relpath), m) // no trailing '/' in importpath

			// collect examples
			testfiles := append(pkginfo.TestGoFiles, pkginfo.XTestGoFiles...)
			files, err = parseFiles(fset, abspath, testfiles)
			if err != nil {
				log.Println("parsing examples:", err)
			}
			info.Examples = collectExamples(pkg, files)

			// collect any notes that we want to show
			if info.PDoc.Notes != nil {
				// could regexp.Compile only once per godoc, but probably not worth it
				if rx, err := regexp.Compile(*notes); err == nil {
					for m, n := range info.PDoc.Notes {
						if rx.MatchString(m) {
							if info.Notes == nil {
								info.Notes = make(map[string][]*doc.Note)
							}
							info.Notes[m] = n
						}
					}
				}
			}

		} else {
			// show source code
			// TODO(gri) Consider eliminating export filtering in this mode,
			//           or perhaps eliminating the mode altogether.
			if mode&noFiltering == 0 {
				packageExports(fset, pkg)
			}
			info.PAst = ast.MergePackageFiles(pkg, 0)
		}
		info.IsMain = pkgname == "main"
	}

	// get directory information, if any
	var dir *Directory
	var timestamp time.Time
	if tree, ts := fsTree.get(); tree != nil && tree.(*Directory) != nil {
		// directory tree is present; lookup respective directory
		// (may still fail if the file system was updated and the
		// new directory tree has not yet been computed)
		dir = tree.(*Directory).lookup(abspath)
		timestamp = ts
	}
	if dir == nil {
		// no directory tree present (too early after startup or
		// command-line mode); compute one level for this page
		// note: cannot use path filter here because in general
		//       it doesn't contain the fsTree path
		dir = newDirectory(abspath, 1)
		timestamp = time.Now()
	}
	info.Dirs = dir.listing(true)
	info.DirTime = timestamp
	info.DirFlat = mode&flatDir != 0

	return info
}

func (h *docServer) ServeHTTP(w http.ResponseWriter, r *http.Request) {
	if redirect(w, r) {
		return
	}

	relpath := pathpkg.Clean(r.URL.Path[len(h.pattern):])
	abspath := pathpkg.Join(h.fsRoot, relpath)
	mode := getPageInfoMode(r)
	if relpath == builtinPkgPath {
		mode = noFiltering
	}
	info := h.getPageInfo(abspath, relpath, mode)
	if info.Err != nil {
		log.Print(info.Err)
		serveError(w, r, relpath, info.Err)
		return
	}

	if mode&noHtml != 0 {
		serveText(w, applyTemplate(packageText, "packageText", info))
		return
	}

	var tabtitle, title, subtitle string
	switch {
	case info.PAst != nil:
		tabtitle = info.PAst.Name.Name
	case info.PDoc != nil:
		tabtitle = info.PDoc.Name
	default:
		tabtitle = info.Dirname
		title = "Directory "
		if *showTimestamps {
			subtitle = "Last update: " + info.DirTime.String()
		}
	}
	if title == "" {
		if info.IsMain {
			// assume that the directory name is the command name
			_, tabtitle = pathpkg.Split(relpath)
			title = "Command "
		} else {
			title = "Package "
		}
	}
	title += tabtitle

	// special cases for top-level package/command directories
	switch tabtitle {
	case "/src/pkg":
		tabtitle = "Packages"
	case "/src/cmd":
		tabtitle = "Commands"
	}

	servePage(w, Page{
		Title:    title,
		Tabtitle: tabtitle,
		Subtitle: subtitle,
		Body:     applyTemplate(packageHTML, "packageHTML", info),
	})
}

// ----------------------------------------------------------------------------
// Search

var searchIndex RWValue

type SearchResult struct {
	Query string
	Alert string // error or warning message

	// identifier matches
	Pak HitList       // packages matching Query
	Hit *LookupResult // identifier matches of Query
	Alt *AltWords     // alternative identifiers to look for

	// textual matches
	Found    int         // number of textual occurrences found
	Textual  []FileLines // textual matches of Query
	Complete bool        // true if all textual occurrences of Query are reported
}

func lookup(query string) (result SearchResult) {
	result.Query = query

	index, timestamp := searchIndex.get()
	if index != nil {
		index := index.(*Index)

		// identifier search
		var err error
		result.Pak, result.Hit, result.Alt, err = index.Lookup(query)
		if err != nil && *maxResults <= 0 {
			// ignore the error if full text search is enabled
			// since the query may be a valid regular expression
			result.Alert = "Error in query string: " + err.Error()
			return
		}

		// full text search
		if *maxResults > 0 && query != "" {
			rx, err := regexp.Compile(query)
			if err != nil {
				result.Alert = "Error in query regular expression: " + err.Error()
				return
			}
			// If we get maxResults+1 results we know that there are more than
			// maxResults results and thus the result may be incomplete (to be
			// precise, we should remove one result from the result set, but
			// nobody is going to count the results on the result page).
			result.Found, result.Textual = index.LookupRegexp(rx, *maxResults+1)
			result.Complete = result.Found <= *maxResults
			if !result.Complete {
				result.Found-- // since we looked for maxResults+1
			}
		}
	}

	// is the result accurate?
	if *indexEnabled {
		if _, ts := fsModified.get(); timestamp.Before(ts) {
			// The index is older than the latest file system change under godoc's observation.
			result.Alert = "Indexing in progress: result may be inaccurate"
		}
	} else {
		result.Alert = "Search index disabled: no results available"
	}

	return
}

func search(w http.ResponseWriter, r *http.Request) {
	query := strings.TrimSpace(r.FormValue("q"))
	result := lookup(query)

	if getPageInfoMode(r)&noHtml != 0 {
		serveText(w, applyTemplate(searchText, "searchText", result))
		return
	}

	var title string
	if result.Hit != nil || len(result.Textual) > 0 {
		title = fmt.Sprintf(`Results for query %q`, query)
	} else {
		title = fmt.Sprintf(`No results found for query %q`, query)
	}

	servePage(w, Page{
		Title:    title,
		Tabtitle: query,
		Query:    query,
		Body:     applyTemplate(searchHTML, "searchHTML", result),
	})
}

// ----------------------------------------------------------------------------
// Documentation Metadata

type Metadata struct {
	Title    string
	Subtitle string
	Template bool   // execute as template
	Path     string // canonical path for this page
	filePath string // filesystem path relative to goroot
}

// extractMetadata extracts the Metadata from a byte slice.
// It returns the Metadata value and the remaining data.
// If no metadata is present the original byte slice is returned.
//
func extractMetadata(b []byte) (meta Metadata, tail []byte, err error) {
	tail = b
	if !bytes.HasPrefix(b, jsonStart) {
		return
	}
	end := bytes.Index(b, jsonEnd)
	if end < 0 {
		return
	}
	b = b[len(jsonStart)-1 : end+1] // drop leading <!-- and include trailing }
	if err = json.Unmarshal(b, &meta); err != nil {
		return
	}
	tail = tail[end+len(jsonEnd):]
	return
}

// updateMetadata scans $GOROOT/doc for HTML files, reads their metadata,
// and updates the docMetadata map.
//
func updateMetadata() {
	metadata := make(map[string]*Metadata)
	var scan func(string) // scan is recursive
	scan = func(dir string) {
		fis, err := fs.ReadDir(dir)
		if err != nil {
			log.Println("updateMetadata:", err)
			return
		}
		for _, fi := range fis {
			name := pathpkg.Join(dir, fi.Name())
			if fi.IsDir() {
				scan(name) // recurse
				continue
			}
			if !strings.HasSuffix(name, ".html") {
				continue
			}
			// Extract metadata from the file.
			b, err := ReadFile(fs, name)
			if err != nil {
				log.Printf("updateMetadata %s: %v", name, err)
				continue
			}
			meta, _, err := extractMetadata(b)
			if err != nil {
				log.Printf("updateMetadata: %s: %v", name, err)
				continue
			}
			// Store relative filesystem path in Metadata.
			meta.filePath = name
			if meta.Path == "" {
				// If no Path, canonical path is actual path.
				meta.Path = meta.filePath
			}
			// Store under both paths.
			metadata[meta.Path] = &meta
			metadata[meta.filePath] = &meta
		}
	}
	scan("/doc")
	docMetadata.set(metadata)
}

// Send a value on this channel to trigger a metadata refresh.
// It is buffered so that if a signal is not lost if sent during a refresh.
//
var refreshMetadataSignal = make(chan bool, 1)

// refreshMetadata sends a signal to update docMetadata. If a refresh is in
// progress the metadata will be refreshed again afterward.
//
func refreshMetadata() {
	select {
	case refreshMetadataSignal <- true:
	default:
	}
}

// refreshMetadataLoop runs forever, updating docMetadata when the underlying
// file system changes. It should be launched in a goroutine by main.
//
func refreshMetadataLoop() {
	for {
		<-refreshMetadataSignal
		updateMetadata()
		time.Sleep(10 * time.Second) // at most once every 10 seconds
	}
}

// metadataFor returns the *Metadata for a given relative path or nil if none
// exists.
//
func metadataFor(relpath string) *Metadata {
	if m, _ := docMetadata.get(); m != nil {
		meta := m.(map[string]*Metadata)
		// If metadata for this relpath exists, return it.
		if p := meta[relpath]; p != nil {
			return p
		}
		// Try with or without trailing slash.
		if strings.HasSuffix(relpath, "/") {
			relpath = relpath[:len(relpath)-1]
		} else {
			relpath = relpath + "/"
		}
		return meta[relpath]
	}
	return nil
}

// ----------------------------------------------------------------------------
// Indexer

// invalidateIndex should be called whenever any of the file systems
// under godoc's observation change so that the indexer is kicked on.
//
func invalidateIndex() {
	fsModified.set(nil)
	refreshMetadata()
}

// indexUpToDate() returns true if the search index is not older
// than any of the file systems under godoc's observation.
//
func indexUpToDate() bool {
	_, fsTime := fsModified.get()
	_, siTime := searchIndex.get()
	return !fsTime.After(siTime)
}

// feedDirnames feeds the directory names of all directories
// under the file system given by root to channel c.
//
func feedDirnames(root *RWValue, c chan<- string) {
	if dir, _ := root.get(); dir != nil {
		for d := range dir.(*Directory).iter(false) {
			c <- d.Path
		}
	}
}

// fsDirnames() returns a channel sending all directory names
// of all the file systems under godoc's observation.
//
func fsDirnames() <-chan string {
	c := make(chan string, 256) // buffered for fewer context switches
	go func() {
		feedDirnames(&fsTree, c)
		close(c)
	}()
	return c
}

func readIndex(filenames string) error {
	matches, err := filepath.Glob(filenames)
	if err != nil {
		return err
	} else if matches == nil {
		return fmt.Errorf("no index files match %q", filenames)
	}
	sort.Strings(matches) // make sure files are in the right order
	files := make([]io.Reader, 0, len(matches))
	for _, filename := range matches {
		f, err := os.Open(filename)
		if err != nil {
			return err
		}
		defer f.Close()
		files = append(files, f)
	}
	x := new(Index)
	if err := x.Read(io.MultiReader(files...)); err != nil {
		return err
	}
	searchIndex.set(x)
	return nil
}

func updateIndex() {
	if *verbose {
		log.Printf("updating index...")
	}
	start := time.Now()
	index := NewIndex(fsDirnames(), *maxResults > 0, *indexThrottle)
	stop := time.Now()
	searchIndex.set(index)
	if *verbose {
		secs := stop.Sub(start).Seconds()
		stats := index.Stats()
		log.Printf("index updated (%gs, %d bytes of source, %d files, %d lines, %d unique words, %d spots)",
			secs, stats.Bytes, stats.Files, stats.Lines, stats.Words, stats.Spots)
	}
	memstats := new(runtime.MemStats)
	runtime.ReadMemStats(memstats)
	log.Printf("before GC: bytes = %d footprint = %d", memstats.HeapAlloc, memstats.Sys)
	runtime.GC()
	runtime.ReadMemStats(memstats)
	log.Printf("after  GC: bytes = %d footprint = %d", memstats.HeapAlloc, memstats.Sys)
}

func indexer() {
	// initialize the index from disk if possible
	if *indexFiles != "" {
		if err := readIndex(*indexFiles); err != nil {
			log.Printf("error reading index: %s", err)
		}
	}

	// repeatedly update the index when it goes out of date
	for {
		if !indexUpToDate() {
			// index possibly out of date - make a new one
			updateIndex()
		}
		delay := 60 * time.Second // by default, try every 60s
		if *testDir != "" {
			// in test mode, try once a second for fast startup
			delay = 1 * time.Second
		}
		time.Sleep(delay)
	}
}
