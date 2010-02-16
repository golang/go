// Copyright 2009 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

package main

import (
	"bytes"
	"flag"
	"fmt"
	"go/ast"
	"go/doc"
	"go/parser"
	"go/printer"
	"go/token"
	"http"
	"io"
	"io/ioutil"
	"log"
	"os"
	pathutil "path"
	"strings"
	"sync"
	"template"
	"time"
	"unicode"
	"utf8"
)


// ----------------------------------------------------------------------------
// Support types

// An RWValue wraps a value and permits mutually exclusive
// access to it and records the time the value was last set.
type RWValue struct {
	mutex     sync.RWMutex
	value     interface{}
	timestamp int64 // time of last set(), in seconds since epoch
}


func (v *RWValue) set(value interface{}) {
	v.mutex.Lock()
	v.value = value
	v.timestamp = time.Seconds()
	v.mutex.Unlock()
}


func (v *RWValue) get() (interface{}, int64) {
	v.mutex.RLock()
	defer v.mutex.RUnlock()
	return v.value, v.timestamp
}


// ----------------------------------------------------------------------------
// Globals

type delayTime struct {
	RWValue
}


func (dt *delayTime) backoff(max int) {
	dt.mutex.Lock()
	v := dt.value.(int) * 2
	if v > max {
		v = max
	}
	dt.value = v
	dt.mutex.Unlock()
}


var (
	verbose = flag.Bool("v", false, "verbose mode")

	// "fixed" file system roots
	goroot   string
	cmdroot  string
	pkgroot  string
	tmplroot string

	// additional file system roots to consider
	path = flag.String("path", "", "additional package directories (colon-separated)")

	// layout control
	tabwidth = flag.Int("tabwidth", 4, "tab width")

	// file system mapping
	fsMap  Mapping // user-defined mapping
	fsTree RWValue // *Directory tree of packages, updated with each sync

	// http handlers
	fileServer http.Handler // default file server
	cmdHandler httpHandler
	pkgHandler httpHandler
)


func init() {
	goroot = os.Getenv("GOROOT")
	if goroot == "" {
		goroot = pathutil.Join(os.Getenv("HOME"), "go")
	}
	flag.StringVar(&goroot, "goroot", goroot, "Go root directory")

	// other flags/variables that depend on goroot
	flag.StringVar(&cmdroot, "cmdroot", pathutil.Join(goroot, "src/cmd"), "command source directory")
	flag.StringVar(&pkgroot, "pkgroot", pathutil.Join(goroot, "src/pkg"), "package source directory")
	flag.StringVar(&tmplroot, "tmplroot", pathutil.Join(goroot, "lib/godoc"), "template directory")
}


func initHandlers() {
	fsMap.Init(*path)
	fileServer = http.FileServer(goroot, "")
	cmdHandler = httpHandler{"/cmd/", cmdroot, false}
	pkgHandler = httpHandler{"/pkg/", pkgroot, true}
}


func registerPublicHandlers(mux *http.ServeMux) {
	mux.Handle(cmdHandler.pattern, &cmdHandler)
	mux.Handle(pkgHandler.pattern, &pkgHandler)
	mux.Handle("/search", http.HandlerFunc(search))
	mux.Handle("/", http.HandlerFunc(serveFile))
}


// ----------------------------------------------------------------------------
// Predicates and small utility functions

func isGoFile(dir *os.Dir) bool {
	return dir.IsRegular() &&
		!strings.HasPrefix(dir.Name, ".") && // ignore .files
		pathutil.Ext(dir.Name) == ".go"
}


func isPkgFile(dir *os.Dir) bool {
	return isGoFile(dir) &&
		!strings.HasSuffix(dir.Name, "_test.go") // ignore test files
}


func isPkgDir(dir *os.Dir) bool {
	return dir.IsDirectory() && len(dir.Name) > 0 && dir.Name[0] != '_'
}


func pkgName(filename string) string {
	file, err := parser.ParseFile(filename, nil, nil, parser.PackageClauseOnly)
	if err != nil || file == nil {
		return ""
	}
	return file.Name.Name()
}


func htmlEscape(s string) string {
	var buf bytes.Buffer
	template.HTMLEscape(&buf, strings.Bytes(s))
	return buf.String()
}


func firstSentence(s string) string {
	i := -1 // index+1 of first period
	j := -1 // index+1 of first period that is followed by white space
	prev := 'A'
	for k, ch := range s {
		k1 := k + 1
		if ch == '.' {
			if i < 0 {
				i = k1 // first period
			}
			if k1 < len(s) && s[k1] <= ' ' {
				if j < 0 {
					j = k1 // first period followed by white space
				}
				if !unicode.IsUpper(prev) {
					j = k1
					break
				}
			}
		}
		prev = ch
	}

	if j < 0 {
		// use the next best period
		j = i
		if j < 0 {
			// no period at all, use the entire string
			j = len(s)
		}
	}

	return s[0:j]
}


func absolutePath(path, defaultRoot string) string {
	abspath := fsMap.ToAbsolute(path)
	if abspath == "" {
		// no user-defined mapping found; use default mapping
		abspath = pathutil.Join(defaultRoot, path)
	}
	return abspath
}


func relativePath(path string) string {
	relpath := fsMap.ToRelative(path)
	if relpath == "" && strings.HasPrefix(path, goroot+"/") {
		// no user-defined mapping found; use default mapping
		relpath = path[len(goroot)+1:]
	}
	// Only if path is an invalid absolute path is relpath == ""
	// at this point. This should never happen since absolute paths
	// are only created via godoc for files that do exist. However,
	// it is ok to return ""; it will simply provide a link to the
	// top of the pkg or src directories.
	return relpath
}


// ----------------------------------------------------------------------------
// Package directories

type Directory struct {
	Depth int
	Path  string // includes Name
	Name  string
	Text  string       // package documentation, if any
	Dirs  []*Directory // subdirectories
}


func newDirTree(path, name string, depth, maxDepth int) *Directory {
	if depth >= maxDepth {
		// return a dummy directory so that the parent directory
		// doesn't get discarded just because we reached the max
		// directory depth
		return &Directory{depth, path, name, "", nil}
	}

	list, _ := ioutil.ReadDir(path) // ignore errors

	// determine number of subdirectories and package files
	ndirs := 0
	nfiles := 0
	text := ""
	for _, d := range list {
		switch {
		case isPkgDir(d):
			ndirs++
		case isPkgFile(d):
			nfiles++
			if text == "" {
				// no package documentation yet; take the first found
				file, err := parser.ParseFile(pathutil.Join(path, d.Name), nil, nil,
					parser.ParseComments|parser.PackageClauseOnly)
				if err == nil &&
					// Also accept fakePkgName, so we get synopses for commmands.
					// Note: This may lead to incorrect results if there is a
					// (left-over) "documentation" package somewhere in a package
					// directory of different name, but this is very unlikely and
					// against current conventions.
					(file.Name.Name() == name || file.Name.Name() == fakePkgName) &&
					file.Doc != nil {
					// found documentation; extract a synopsys
					text = firstSentence(doc.CommentText(file.Doc))
				}
			}
		}
	}

	// create subdirectory tree
	var dirs []*Directory
	if ndirs > 0 {
		dirs = make([]*Directory, ndirs)
		i := 0
		for _, d := range list {
			if isPkgDir(d) {
				dd := newDirTree(pathutil.Join(path, d.Name), d.Name, depth+1, maxDepth)
				if dd != nil {
					dirs[i] = dd
					i++
				}
			}
		}
		dirs = dirs[0:i]
	}

	// if there are no package files and no subdirectories
	// (with package files), ignore the directory
	if nfiles == 0 && len(dirs) == 0 {
		return nil
	}

	return &Directory{depth, path, name, text, dirs}
}


// newDirectory creates a new package directory tree with at most maxDepth
// levels, anchored at root which is relative to goroot. The result tree
// only contains directories that contain package files or that contain
// subdirectories containing package files (transitively).
//
func newDirectory(root string, maxDepth int) *Directory {
	d, err := os.Lstat(root)
	if err != nil || !isPkgDir(d) {
		return nil
	}
	return newDirTree(root, d.Name, 0, maxDepth)
}


func (dir *Directory) walk(c chan<- *Directory, skipRoot bool) {
	if dir != nil {
		if !skipRoot {
			c <- dir
		}
		for _, d := range dir.Dirs {
			d.walk(c, false)
		}
	}
}


func (dir *Directory) iter(skipRoot bool) <-chan *Directory {
	c := make(chan *Directory)
	go func() {
		dir.walk(c, skipRoot)
		close(c)
	}()
	return c
}


func (dir *Directory) lookupLocal(name string) *Directory {
	for _, d := range dir.Dirs {
		if d.Name == name {
			return d
		}
	}
	return nil
}


// lookup looks for the *Directory for a given path, relative to dir.
func (dir *Directory) lookup(path string) *Directory {
	d := strings.Split(dir.Path, "/", 0)
	p := strings.Split(path, "/", 0)
	i := 0
	for i < len(d) {
		if i >= len(p) || d[i] != p[i] {
			return nil
		}
		i++
	}
	for dir != nil && i < len(p) {
		dir = dir.lookupLocal(p[i])
		i++
	}
	return dir
}


// DirEntry describes a directory entry. The Depth and Height values
// are useful for presenting an entry in an indented fashion.
//
type DirEntry struct {
	Depth    int    // >= 0
	Height   int    // = DirList.MaxHeight - Depth, > 0
	Path     string // includes Name, relative to DirList root
	Name     string
	Synopsis string
}


type DirList struct {
	MaxHeight int // directory tree height, > 0
	List      []DirEntry
}


// listing creates a (linear) directory listing from a directory tree.
// If skipRoot is set, the root directory itself is excluded from the list.
//
func (root *Directory) listing(skipRoot bool) *DirList {
	if root == nil {
		return nil
	}

	// determine number of entries n and maximum height
	n := 0
	minDepth := 1 << 30 // infinity
	maxDepth := 0
	for d := range root.iter(skipRoot) {
		n++
		if minDepth > d.Depth {
			minDepth = d.Depth
		}
		if maxDepth < d.Depth {
			maxDepth = d.Depth
		}
	}
	maxHeight := maxDepth - minDepth + 1

	if n == 0 {
		return nil
	}

	// create list
	list := make([]DirEntry, n)
	i := 0
	for d := range root.iter(skipRoot) {
		p := &list[i]
		p.Depth = d.Depth - minDepth
		p.Height = maxHeight - p.Depth
		// the path is relative to root.Path - remove the root.Path
		// prefix (the prefix should always be present but avoid
		// crashes and check)
		path := d.Path
		if strings.HasPrefix(d.Path, root.Path) {
			path = d.Path[len(root.Path):]
		}
		// remove trailing '/' if any - path must be relative
		if len(path) > 0 && path[0] == '/' {
			path = path[1:]
		}
		p.Path = path
		p.Name = d.Name
		p.Synopsis = d.Text
		i++
	}

	return &DirList{maxHeight, list}
}


// ----------------------------------------------------------------------------
// HTML formatting support

// Styler implements a printer.Styler.
type Styler struct {
	linetags  bool
	highlight string
}


// Use the defaultStyler when there is no specific styler.
// The defaultStyler does not emit line tags since they may
// interfere with tags emitted by templates.
// TODO(gri): Should emit line tags at the beginning of a line;
//            never in the middle of code.
var defaultStyler Styler


func (s *Styler) LineTag(line int) (text []byte, tag printer.HTMLTag) {
	if s.linetags {
		tag = printer.HTMLTag{fmt.Sprintf(`<a id="L%d">`, line), "</a>"}
	}
	return
}


func (s *Styler) Comment(c *ast.Comment, line []byte) (text []byte, tag printer.HTMLTag) {
	text = line
	// minimal syntax-coloring of comments for now - people will want more
	// (don't do anything more until there's a button to turn it on/off)
	tag = printer.HTMLTag{`<span class="comment">`, "</span>"}
	return
}


func (s *Styler) BasicLit(x *ast.BasicLit) (text []byte, tag printer.HTMLTag) {
	text = x.Value
	return
}


func (s *Styler) Ident(id *ast.Ident) (text []byte, tag printer.HTMLTag) {
	text = strings.Bytes(id.Name())
	if s.highlight == id.Name() {
		tag = printer.HTMLTag{"<span class=highlight>", "</span>"}
	}
	return
}


func (s *Styler) Token(tok token.Token) (text []byte, tag printer.HTMLTag) {
	text = strings.Bytes(tok.String())
	return
}


// ----------------------------------------------------------------------------
// Tab conversion

var spaces = strings.Bytes("                ") // 16 spaces seems like a good number

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


func (p *tconv) writeIndent() (err os.Error) {
	i := p.indent
	for i > len(spaces) {
		i -= len(spaces)
		if _, err = p.output.Write(spaces); err != nil {
			return
		}
	}
	_, err = p.output.Write(spaces[0:i])
	return
}


func (p *tconv) Write(data []byte) (n int, err os.Error) {
	pos := 0 // valid if p.state == collecting
	var b byte
	for n, b = range data {
		switch p.state {
		case indenting:
			switch b {
			case '\t', '\v':
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
	if p.state == collecting {
		_, err = p.output.Write(data[pos:])
	}
	return
}


// ----------------------------------------------------------------------------
// Templates

// Write an AST-node to w; optionally html-escaped.
func writeNode(w io.Writer, node interface{}, html bool, styler printer.Styler) {
	mode := printer.TabIndent | printer.UseSpaces
	if html {
		mode |= printer.GenHTML
	}
	// convert trailing tabs into spaces using a tconv filter
	// to ensure a good outcome in most browsers (there may still
	// be tabs in comments and strings, but converting those into
	// the right number of spaces is much harder)
	(&printer.Config{mode, *tabwidth, styler}).Fprint(&tconv{output: w}, node)
}


// Write text to w; optionally html-escaped.
func writeText(w io.Writer, text []byte, html bool) {
	if html {
		template.HTMLEscape(w, text)
		return
	}
	w.Write(text)
}


// Write anything to w; optionally html-escaped.
func writeAny(w io.Writer, x interface{}, html bool) {
	switch v := x.(type) {
	case []byte:
		writeText(w, v, html)
	case string:
		writeText(w, strings.Bytes(v), html)
	case ast.Decl, ast.Expr, ast.Stmt, *ast.File:
		writeNode(w, x, html, &defaultStyler)
	default:
		if html {
			var buf bytes.Buffer
			fmt.Fprint(&buf, x)
			writeText(w, buf.Bytes(), true)
		} else {
			fmt.Fprint(w, x)
		}
	}
}


// Template formatter for "html" format.
func htmlFmt(w io.Writer, x interface{}, format string) {
	writeAny(w, x, true)
}


// Template formatter for "html-esc" format.
func htmlEscFmt(w io.Writer, x interface{}, format string) {
	var buf bytes.Buffer
	writeAny(&buf, x, false)
	template.HTMLEscape(w, buf.Bytes())
}


// Template formatter for "html-comment" format.
func htmlCommentFmt(w io.Writer, x interface{}, format string) {
	var buf bytes.Buffer
	writeAny(&buf, x, false)
	doc.ToHTML(w, buf.Bytes()) // does html-escaping
}


// Template formatter for "" (default) format.
func textFmt(w io.Writer, x interface{}, format string) {
	writeAny(w, x, false)
}


// Template formatter for the various "url-xxx" formats.
func urlFmt(w io.Writer, x interface{}, format string) {
	var path string
	var line int

	// determine path and position info, if any
	type positioner interface {
		Pos() token.Position
	}
	switch t := x.(type) {
	case string:
		path = t
	case positioner:
		pos := t.Pos()
		if pos.IsValid() {
			path = pos.Filename
			line = pos.Line
		}
	}

	// map path
	relpath := relativePath(path)

	// convert to URL
	switch format {
	default:
		// we should never reach here, but be resilient
		// and assume the url-pkg format instead
		log.Stderrf("INTERNAL ERROR: urlFmt(%s)", format)
		fallthrough
	case "url-pkg":
		// because of the irregular mapping under goroot
		// we need to correct certain relative paths
		if strings.HasPrefix(relpath, "src/pkg/") {
			relpath = relpath[len("src/pkg/"):]
		}
		template.HTMLEscape(w, strings.Bytes(pkgHandler.pattern+relpath))
	case "url-src":
		template.HTMLEscape(w, strings.Bytes("/"+relpath))
	case "url-pos":
		// line id's in html-printed source are of the
		// form "L%d" where %d stands for the line number
		template.HTMLEscape(w, strings.Bytes("/"+relpath))
		fmt.Fprintf(w, "#L%d", line)
	}
}


// The strings in infoKinds must be properly html-escaped.
var infoKinds = [nKinds]string{
	PackageClause: "package&nbsp;clause",
	ImportDecl: "import&nbsp;decl",
	ConstDecl: "const&nbsp;decl",
	TypeDecl: "type&nbsp;decl",
	VarDecl: "var&nbsp;decl",
	FuncDecl: "func&nbsp;decl",
	MethodDecl: "method&nbsp;decl",
	Use: "use",
}


// Template formatter for "infoKind" format.
func infoKindFmt(w io.Writer, x interface{}, format string) {
	fmt.Fprintf(w, infoKinds[x.(SpotKind)]) // infoKind entries are html-escaped
}


// Template formatter for "infoLine" format.
func infoLineFmt(w io.Writer, x interface{}, format string) {
	info := x.(SpotInfo)
	line := info.Lori()
	if info.IsIndex() {
		index, _ := searchIndex.get()
		line = index.(*Index).Snippet(line).Line
	}
	fmt.Fprintf(w, "%d", line)
}


// Template formatter for "infoSnippet" format.
func infoSnippetFmt(w io.Writer, x interface{}, format string) {
	info := x.(SpotInfo)
	text := `<span class="alert">no snippet text available</span>`
	if info.IsIndex() {
		index, _ := searchIndex.get()
		// no escaping of snippet text needed;
		// snippet text is escaped when generated
		text = index.(*Index).Snippet(info.Lori()).Text
	}
	fmt.Fprint(w, text)
}


// Template formatter for "padding" format.
func paddingFmt(w io.Writer, x interface{}, format string) {
	for i := x.(int); i > 0; i-- {
		fmt.Fprint(w, `<td width="25"></td>`)
	}
}


// Template formatter for "time" format.
func timeFmt(w io.Writer, x interface{}, format string) {
	// note: os.Dir.Mtime_ns is in uint64 in ns!
	template.HTMLEscape(w, strings.Bytes(time.SecondsToLocalTime(int64(x.(uint64)/1e9)).String()))
}


// Template formatter for "dir/" format.
func dirslashFmt(w io.Writer, x interface{}, format string) {
	if x.(*os.Dir).IsDirectory() {
		w.Write([]byte{'/'})
	}
}


// Template formatter for "localname" format.
func localnameFmt(w io.Writer, x interface{}, format string) {
	_, localname := pathutil.Split(x.(string))
	template.HTMLEscape(w, strings.Bytes(localname))
}


var fmap = template.FormatterMap{
	"": textFmt,
	"html": htmlFmt,
	"html-esc": htmlEscFmt,
	"html-comment": htmlCommentFmt,
	"url-pkg": urlFmt,
	"url-src": urlFmt,
	"url-pos": urlFmt,
	"infoKind": infoKindFmt,
	"infoLine": infoLineFmt,
	"infoSnippet": infoSnippetFmt,
	"padding": paddingFmt,
	"time": timeFmt,
	"dir/": dirslashFmt,
	"localname": localnameFmt,
}


func readTemplate(name string) *template.Template {
	path := pathutil.Join(tmplroot, name)
	data, err := ioutil.ReadFile(path)
	if err != nil {
		log.Exitf("ReadFile %s: %v", path, err)
	}
	t, err := template.Parse(string(data), fmap)
	if err != nil {
		log.Exitf("%s: %v", name, err)
	}
	return t
}


var (
	dirlistHTML,
		errorHTML,
		godocHTML,
		packageHTML,
		packageText,
		searchHTML *template.Template
)

func readTemplates() {
	// have to delay until after flags processing, so that tmplroot is known
	dirlistHTML = readTemplate("dirlist.html")
	errorHTML = readTemplate("error.html")
	godocHTML = readTemplate("godoc.html")
	packageHTML = readTemplate("package.html")
	packageText = readTemplate("package.txt")
	searchHTML = readTemplate("search.html")
}


// ----------------------------------------------------------------------------
// Generic HTML wrapper

func servePage(c *http.Conn, title, query string, content []byte) {
	type Data struct {
		Title     string
		PkgRoots  []string
		Timestamp uint64 // int64 to be compatible with os.Dir.Mtime_ns
		Query     string
		Content   []byte
	}

	_, ts := fsTree.get()
	d := Data{
		Title: title,
		PkgRoots: fsMap.PrefixList(),
		Timestamp: uint64(ts) * 1e9, // timestamp in ns
		Query: query,
		Content: content,
	}

	if err := godocHTML.Execute(&d, c); err != nil {
		log.Stderrf("godocHTML.Execute: %s", err)
	}
}


func serveText(c *http.Conn, text []byte) {
	c.SetHeader("Content-Type", "text/plain; charset=utf-8")
	c.Write(text)
}


// ----------------------------------------------------------------------------
// Files

var (
	tagBegin = strings.Bytes("<!--")
	tagEnd   = strings.Bytes("-->")
)

// commentText returns the text of the first HTML comment in src.
func commentText(src []byte) (text string) {
	i := bytes.Index(src, tagBegin)
	j := bytes.Index(src, tagEnd)
	if i >= 0 && j >= i+len(tagBegin) {
		text = string(bytes.TrimSpace(src[i+len(tagBegin) : j]))
	}
	return
}


func serveError(c *http.Conn, r *http.Request, relpath string, err os.Error) {
	contents := applyTemplate(errorHTML, "errorHTML", err)
	servePage(c, "File "+relpath, "", contents)
}


func serveHTMLDoc(c *http.Conn, r *http.Request, abspath, relpath string) {
	// get HTML body contents
	src, err := ioutil.ReadFile(abspath)
	if err != nil {
		log.Stderrf("ioutil.ReadFile: %s", err)
		serveError(c, r, relpath, err)
		return
	}

	// if it begins with "<!DOCTYPE " assume it is standalone
	// html that doesn't need the template wrapping.
	if bytes.HasPrefix(src, strings.Bytes("<!DOCTYPE ")) {
		c.Write(src)
		return
	}

	// if it's the language spec, add tags to EBNF productions
	if strings.HasSuffix(abspath, "go_spec.html") {
		var buf bytes.Buffer
		linkify(&buf, src)
		src = buf.Bytes()
	}

	title := commentText(src)
	servePage(c, title, "", src)
}


func applyTemplate(t *template.Template, name string, data interface{}) []byte {
	var buf bytes.Buffer
	if err := t.Execute(data, &buf); err != nil {
		log.Stderrf("%s.Execute: %s", name, err)
	}
	return buf.Bytes()
}


func serveGoSource(c *http.Conn, r *http.Request, abspath, relpath string) {
	file, err := parser.ParseFile(abspath, nil, nil, parser.ParseComments)
	if err != nil {
		log.Stderrf("parser.ParseFile: %s", err)
		serveError(c, r, relpath, err)
		return
	}

	var buf bytes.Buffer
	fmt.Fprintln(&buf, "<pre>")
	writeNode(&buf, file, true, &Styler{linetags: true, highlight: r.FormValue("h")})
	fmt.Fprintln(&buf, "</pre>")

	servePage(c, "Source file "+relpath, "", buf.Bytes())
}


func redirect(c *http.Conn, r *http.Request) (redirected bool) {
	if canonical := pathutil.Clean(r.URL.Path) + "/"; r.URL.Path != canonical {
		http.Redirect(c, canonical, http.StatusMovedPermanently)
		redirected = true
	}
	return
}


// TODO(gri): Should have a mapping from extension to handler, eventually.

// textExt[x] is true if the extension x indicates a text file, and false otherwise.
var textExt = map[string]bool{
	".css": false, // must be served raw
	".js": false, // must be served raw
}


func isTextFile(path string) bool {
	// if the extension is known, use it for decision making
	if isText, found := textExt[pathutil.Ext(path)]; found {
		return isText
	}

	// the extension is not known; read an initial chunk of
	// file and check if it looks like correct UTF-8; if it
	// does, it's probably a text file
	f, err := os.Open(path, os.O_RDONLY, 0)
	if err != nil {
		return false
	}
	defer f.Close()

	var buf [1024]byte
	n, err := f.Read(&buf)
	if err != nil {
		return false
	}

	s := string(buf[0:n])
	n -= utf8.UTFMax // make sure there's enough bytes for a complete unicode char
	for i, c := range s {
		if i > n {
			break
		}
		if c == 0xFFFD || c < ' ' && c != '\n' && c != '\t' {
			// decoding error or control character - not a text file
			return false
		}
	}

	// likely a text file
	return true
}


func serveTextFile(c *http.Conn, r *http.Request, abspath, relpath string) {
	src, err := ioutil.ReadFile(abspath)
	if err != nil {
		log.Stderrf("ioutil.ReadFile: %s", err)
		serveError(c, r, relpath, err)
		return
	}

	var buf bytes.Buffer
	fmt.Fprintln(&buf, "<pre>")
	template.HTMLEscape(&buf, src)
	fmt.Fprintln(&buf, "</pre>")

	servePage(c, "Text file "+relpath, "", buf.Bytes())
}


func serveDirectory(c *http.Conn, r *http.Request, abspath, relpath string) {
	if redirect(c, r) {
		return
	}

	list, err := ioutil.ReadDir(abspath)
	if err != nil {
		log.Stderrf("ioutil.ReadDir: %s", err)
		serveError(c, r, relpath, err)
		return
	}

	for _, d := range list {
		if d.IsDirectory() {
			d.Size = 0
		}
	}

	contents := applyTemplate(dirlistHTML, "dirlistHTML", list)
	servePage(c, "Directory "+relpath, "", contents)
}


func serveFile(c *http.Conn, r *http.Request) {
	relpath := r.URL.Path[1:] // serveFile URL paths start with '/'
	abspath := absolutePath(relpath, goroot)

	// pick off special cases and hand the rest to the standard file server
	switch r.URL.Path {
	case "/":
		serveHTMLDoc(c, r, pathutil.Join(goroot, "doc/root.html"), "doc/root.html")
		return

	case "/doc/root.html":
		// hide landing page from its real name
		http.Redirect(c, "/", http.StatusMovedPermanently)
		return
	}

	switch pathutil.Ext(abspath) {
	case ".html":
		if strings.HasSuffix(abspath, "/index.html") {
			// We'll show index.html for the directory.
			// Use the dir/ version as canonical instead of dir/index.html.
			http.Redirect(c, r.URL.Path[0:len(r.URL.Path)-len("index.html")], http.StatusMovedPermanently)
			return
		}
		serveHTMLDoc(c, r, abspath, relpath)
		return

	case ".go":
		serveGoSource(c, r, abspath, relpath)
		return
	}

	dir, err := os.Lstat(abspath)
	if err != nil {
		log.Stderr(err)
		serveError(c, r, abspath, err)
		return
	}

	if dir != nil && dir.IsDirectory() {
		if redirect(c, r) {
			return
		}
		if index := abspath + "/index.html"; isTextFile(index) {
			serveHTMLDoc(c, r, index, relativePath(index))
			return
		}
		serveDirectory(c, r, abspath, relpath)
		return
	}

	if isTextFile(abspath) {
		serveTextFile(c, r, abspath, relpath)
		return
	}

	fileServer.ServeHTTP(c, r)
}


// ----------------------------------------------------------------------------
// Packages

// Fake package file and name for commands. Contains the command documentation.
const fakePkgFile = "doc.go"
const fakePkgName = "documentation"


type PageInfo struct {
	Dirname string          // directory containing the package
	PDoc    *doc.PackageDoc // nil if no package found
	Dirs    *DirList        // nil if no directory information found
	IsPkg   bool            // false if this is not documenting a real package
}


type httpHandler struct {
	pattern string // url pattern; e.g. "/pkg/"
	fsRoot  string // file system root to which the pattern is mapped
	isPkg   bool   // true if this handler serves real package documentation (as opposed to command documentation)
}


// getPageInfo returns the PageInfo for a package directory path. If
// the parameter try is true, no errors are logged if getPageInfo fails.
// If there is no corresponding package in the directory, PageInfo.PDoc
// is nil. If there are no subdirectories, PageInfo.Dirs is nil.
//
func (h *httpHandler) getPageInfo(relpath string, try bool) PageInfo {
	dirname := absolutePath(relpath, h.fsRoot)

	// filter function to select the desired .go files
	filter := func(d *os.Dir) bool {
		// If we are looking at cmd documentation, only accept
		// the special fakePkgFile containing the documentation.
		return isPkgFile(d) && (h.isPkg || d.Name == fakePkgFile)
	}

	// get package ASTs
	pkgs, err := parser.ParseDir(dirname, filter, parser.ParseComments)
	if err != nil && !try {
		// TODO: errors should be shown instead of an empty directory
		log.Stderrf("parser.parseDir: %s", err)
	}
	if len(pkgs) != 1 && !try {
		// TODO: should handle multiple packages
		log.Stderrf("parser.parseDir: found %d packages", len(pkgs))
	}

	// Get the best matching package: either the first one, or the
	// first one whose package name matches the directory name.
	// The package name is the directory name within its parent
	// (use dirname instead of path because dirname is clean; i.e.
	// has no trailing '/').
	_, pkgname := pathutil.Split(dirname)
	var pkg *ast.Package
	for _, p := range pkgs {
		switch {
		case pkg == nil:
			pkg = p
		case p.Name == pkgname:
			pkg = p
			break
		}
	}

	// compute package documentation
	var pdoc *doc.PackageDoc
	if pkg != nil {
		ast.PackageExports(pkg)
		pdoc = doc.NewPackageDoc(pkg, pathutil.Clean(relpath)) // no trailing '/' in importpath
	}

	// get directory information
	var dir *Directory
	if tree, _ := fsTree.get(); tree != nil {
		// directory tree is present; lookup respective directory
		// (may still fail if the file system was updated and the
		// new directory tree has not yet been computed)
		// TODO(gri) Need to build directory tree for fsMap entries
		dir = tree.(*Directory).lookup(dirname)
	}
	if dir == nil {
		// no directory tree present (either early after startup
		// or command-line mode, or we don't build a tree for the
		// directory; e.g. google3); compute one level for this page
		dir = newDirectory(dirname, 1)
	}

	return PageInfo{dirname, pdoc, dir.listing(true), h.isPkg}
}


func (h *httpHandler) ServeHTTP(c *http.Conn, r *http.Request) {
	if redirect(c, r) {
		return
	}

	relpath := r.URL.Path[len(h.pattern):]
	info := h.getPageInfo(relpath, false)

	if r.FormValue("f") == "text" {
		contents := applyTemplate(packageText, "packageText", info)
		serveText(c, contents)
		return
	}

	var title string
	if info.PDoc != nil {
		switch {
		case h.isPkg:
			title = "Package " + info.PDoc.PackageName
		case info.PDoc.PackageName == fakePkgName:
			// assume that the directory name is the command name
			_, pkgname := pathutil.Split(pathutil.Clean(relpath))
			title = "Command " + pkgname
		default:
			title = "Command " + info.PDoc.PackageName
		}
	} else {
		title = "Directory " + relativePath(info.Dirname)
	}

	contents := applyTemplate(packageHTML, "packageHTML", info)
	servePage(c, title, "", contents)
}


// ----------------------------------------------------------------------------
// Search

var searchIndex RWValue

type SearchResult struct {
	Query    string
	Hit      *LookupResult
	Alt      *AltWords
	Illegal  bool
	Accurate bool
}

func search(c *http.Conn, r *http.Request) {
	query := strings.TrimSpace(r.FormValue("q"))
	var result SearchResult

	if index, timestamp := searchIndex.get(); index != nil {
		result.Query = query
		result.Hit, result.Alt, result.Illegal = index.(*Index).Lookup(query)
		_, ts := fsTree.get()
		result.Accurate = timestamp >= ts
	}

	var title string
	if result.Hit != nil {
		title = fmt.Sprintf(`Results for query %q`, query)
	} else {
		title = fmt.Sprintf(`No results found for query %q`, query)
	}

	contents := applyTemplate(searchHTML, "searchHTML", result)
	servePage(c, title, query, contents)
}


// ----------------------------------------------------------------------------
// Indexer

func indexer() {
	for {
		_, ts := fsTree.get()
		if _, timestamp := searchIndex.get(); timestamp < ts {
			// index possibly out of date - make a new one
			// (could use a channel to send an explicit signal
			// from the sync goroutine, but this solution is
			// more decoupled, trivial, and works well enough)
			start := time.Nanoseconds()
			index := NewIndex(goroot)
			stop := time.Nanoseconds()
			searchIndex.set(index)
			if *verbose {
				secs := float64((stop-start)/1e6) / 1e3
				nwords, nspots := index.Size()
				log.Stderrf("index updated (%gs, %d unique words, %d spots)", secs, nwords, nspots)
			}
		}
		time.Sleep(1 * 60e9) // try once a minute
	}
}
