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
	"regexp"
	"runtime"
	"sort"
	"strings"
	"template"
	"time"
)


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
	// don't change dt.timestamp - calling backoff indicates an error condition
	dt.mutex.Unlock()
}


var (
	verbose = flag.Bool("v", false, "verbose mode")

	// file system roots
	// TODO(gri) consider the invariant that goroot always end in '/'
	goroot      = flag.String("goroot", runtime.GOROOT(), "Go root directory")
	testDir     = flag.String("testdir", "", "Go root subdirectory - for testing only (faster startups)")
	pkgPath     = flag.String("path", "", "additional package directories (colon-separated)")
	filter      = flag.String("filter", "", "filter file containing permitted package directory paths")
	filterMin   = flag.Int("filter_minutes", 0, "filter file update interval in minutes; disabled if <= 0")
	filterDelay delayTime // actual filter update interval in minutes; usually filterDelay == filterMin, but filterDelay may back off exponentially

	// layout control
	tabwidth       = flag.Int("tabwidth", 4, "tab width")
	showTimestamps = flag.Bool("timestamps", true, "show timestamps with directory listings")
	maxResults     = flag.Int("maxresults", 10000, "maximum number of full text search results shown")

	// file system mapping
	fsMap      Mapping // user-defined mapping
	fsTree     RWValue // *Directory tree of packages, updated with each sync
	pathFilter RWValue // filter used when building fsMap directory trees
	fsModified RWValue // timestamp of last call to invalidateIndex

	// http handlers
	fileServer http.Handler // default file server
	cmdHandler httpHandler
	pkgHandler httpHandler
)


func initHandlers() {
	fsMap.Init(*pkgPath)
	fileServer = http.FileServer(*goroot, "")
	cmdHandler = httpHandler{"/cmd/", pathutil.Join(*goroot, "src/cmd"), false}
	pkgHandler = httpHandler{"/pkg/", pathutil.Join(*goroot, "src/pkg"), true}
}


func registerPublicHandlers(mux *http.ServeMux) {
	mux.Handle(cmdHandler.pattern, &cmdHandler)
	mux.Handle(pkgHandler.pattern, &pkgHandler)
	mux.HandleFunc("/doc/codewalk/", codewalk)
	mux.HandleFunc("/search", search)
	mux.Handle("/robots.txt", fileServer)
	mux.HandleFunc("/", serveFile)
}


func initFSTree() {
	fsTree.set(newDirectory(pathutil.Join(*goroot, *testDir), nil, -1))
	invalidateIndex()
}


// ----------------------------------------------------------------------------
// Directory filters

// isParentOf returns true if p is a parent of (or the same as) q
// where p and q are directory paths.
func isParentOf(p, q string) bool {
	n := len(p)
	return strings.HasPrefix(q, p) && (len(q) <= n || q[n] == '/')
}


func setPathFilter(list []string) {
	if len(list) == 0 {
		pathFilter.set(nil)
		return
	}

	// len(list) > 0
	pathFilter.set(func(path string) bool {
		// list is sorted in increasing order and for each path all its children are removed
		i := sort.Search(len(list), func(i int) bool { return list[i] > path })
		// Now we have list[i-1] <= path < list[i].
		// Path may be a child of list[i-1] or a parent of list[i].
		return i > 0 && isParentOf(list[i-1], path) || i < len(list) && isParentOf(path, list[i])
	})
}


func getPathFilter() func(string) bool {
	f, _ := pathFilter.get()
	if f != nil {
		return f.(func(string) bool)
	}
	return nil
}


// readDirList reads a file containing a newline-separated list
// of directory paths and returns the list of paths.
func readDirList(filename string) ([]string, os.Error) {
	contents, err := ioutil.ReadFile(filename)
	if err != nil {
		return nil, err
	}
	// create a sorted list of valid directory names
	filter := func(path string) bool {
		d, err := os.Lstat(path)
		return err == nil && isPkgDir(d)
	}
	list := canonicalizePaths(strings.Split(string(contents), "\n", -1), filter)
	// for each parent path, remove all it's children q
	// (requirement for binary search to work when filtering)
	i := 0
	for _, q := range list {
		if i == 0 || !isParentOf(list[i-1], q) {
			list[i] = q
			i++
		}
	}
	return list[0:i], nil
}


// updateMappedDirs computes the directory tree for
// each user-defined file system mapping. If a filter
// is provided, it is used to filter directories.
//
func updateMappedDirs(filter func(string) bool) {
	if !fsMap.IsEmpty() {
		fsMap.Iterate(func(path string, value *RWValue) bool {
			value.set(newDirectory(path, filter, -1))
			return true
		})
		invalidateIndex()
	}
}


func updateFilterFile() {
	updateMappedDirs(nil) // no filter for accuracy

	// collect directory tree leaf node paths
	var buf bytes.Buffer
	fsMap.Iterate(func(_ string, value *RWValue) bool {
		v, _ := value.get()
		if v != nil && v.(*Directory) != nil {
			v.(*Directory).writeLeafs(&buf)
		}
		return true
	})

	// update filter file
	if err := writeFileAtomically(*filter, buf.Bytes()); err != nil {
		log.Printf("writeFileAtomically(%s): %s", *filter, err)
		filterDelay.backoff(24 * 60) // back off exponentially, but try at least once a day
	} else {
		filterDelay.set(*filterMin) // revert to regular filter update schedule
	}
}


func initDirTrees() {
	// setup initial path filter
	if *filter != "" {
		list, err := readDirList(*filter)
		if err != nil {
			log.Printf("%s", err)
		} else if len(list) == 0 {
			log.Printf("no directory paths in file %s", *filter)
		}
		setPathFilter(list)
	}

	go updateMappedDirs(getPathFilter()) // use filter for speed

	// start filter update goroutine, if enabled.
	if *filter != "" && *filterMin > 0 {
		filterDelay.set(*filterMin) // initial filter update delay
		go func() {
			for {
				if *verbose {
					log.Printf("start update of %s", *filter)
				}
				updateFilterFile()
				delay, _ := filterDelay.get()
				if *verbose {
					log.Printf("next filter update in %dmin", delay.(int))
				}
				time.Sleep(int64(delay.(int)) * 60e9)
			}
		}()
	}
}


// ----------------------------------------------------------------------------
// Path mapping

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
	if relpath == "" {
		// prefix must end in '/'
		prefix := *goroot
		if len(prefix) > 0 && prefix[len(prefix)-1] != '/' {
			prefix += "/"
		}
		if strings.HasPrefix(path, prefix) {
			// no user-defined mapping found; use default mapping
			relpath = path[len(prefix):]
		}
	}
	// Only if path is an invalid absolute path is relpath == ""
	// at this point. This should never happen since absolute paths
	// are only created via godoc for files that do exist. However,
	// it is ok to return ""; it will simply provide a link to the
	// top of the pkg or src directories.
	return relpath
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


func (p *tconv) writeIndent() (err os.Error) {
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


func (p *tconv) Write(data []byte) (n int, err os.Error) {
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
	(&printer.Config{mode, *tabwidth}).Fprint(&tconv{output: w}, fset, x)
}


// Write anything to w.
func writeAny(w io.Writer, fset *token.FileSet, x interface{}) {
	switch v := x.(type) {
	case []byte:
		w.Write(v)
	case string:
		w.Write([]byte(v))
	case ast.Decl, ast.Expr, ast.Stmt, *ast.File:
		writeNode(w, fset, x)
	default:
		fmt.Fprint(w, x)
	}
}


// Write anything html-escaped to w.
func writeAnyHTML(w io.Writer, fset *token.FileSet, x interface{}) {
	switch v := x.(type) {
	case []byte:
		template.HTMLEscape(w, v)
	case string:
		template.HTMLEscape(w, []byte(v))
	case ast.Decl, ast.Expr, ast.Stmt, *ast.File:
		var buf bytes.Buffer
		writeNode(&buf, fset, x)
		FormatText(w, buf.Bytes(), -1, true, "", nil)
	default:
		var buf bytes.Buffer
		fmt.Fprint(&buf, x)
		template.HTMLEscape(w, buf.Bytes())
	}
}


func fileset(x []interface{}) *token.FileSet {
	if len(x) > 1 {
		if fset, ok := x[1].(*token.FileSet); ok {
			return fset
		}
	}
	return nil
}


// Template formatter for "html-esc" format.
func htmlEscFmt(w io.Writer, format string, x ...interface{}) {
	writeAnyHTML(w, fileset(x), x[0])
}


// Template formatter for "html-comment" format.
func htmlCommentFmt(w io.Writer, format string, x ...interface{}) {
	var buf bytes.Buffer
	writeAny(&buf, fileset(x), x[0])
	// TODO(gri) Provide list of words (e.g. function parameters)
	//           to be emphasized by ToHTML.
	doc.ToHTML(w, buf.Bytes(), nil) // does html-escaping
}


// Template formatter for "" (default) format.
func textFmt(w io.Writer, format string, x ...interface{}) {
	writeAny(w, fileset(x), x[0])
}


// Template formatter for "urlquery-esc" format.
func urlQueryEscFmt(w io.Writer, format string, x ...interface{}) {
	var buf bytes.Buffer
	writeAny(&buf, fileset(x), x[0])
	template.HTMLEscape(w, []byte(http.URLEscape(string(buf.Bytes()))))
}


// Template formatter for the various "url-xxx" formats excluding url-esc.
func urlFmt(w io.Writer, format string, x ...interface{}) {
	var path string
	var line int
	var low, high int // selection

	// determine path and position info, if any
	type positioner interface {
		Pos() token.Pos
		End() token.Pos
	}
	switch t := x[0].(type) {
	case string:
		path = t
	case positioner:
		fset := fileset(x)
		if p := t.Pos(); p.IsValid() {
			pos := fset.Position(p)
			path = pos.Filename
			line = pos.Line
			low = pos.Offset
		}
		if p := t.End(); p.IsValid() {
			high = fset.Position(p).Offset
		}
	default:
		// we should never reach here, but be resilient
		// and assume the position is invalid (empty path,
		// and line 0)
		log.Printf("INTERNAL ERROR: urlFmt(%s) without a string or positioner", format)
	}

	// map path
	relpath := relativePath(path)

	// convert to relative URLs so that they can also
	// be used as relative file names in .txt templates
	switch format {
	default:
		// we should never reach here, but be resilient
		// and assume the url-pkg format instead
		log.Printf("INTERNAL ERROR: urlFmt(%s)", format)
		fallthrough
	case "url-pkg":
		// because of the irregular mapping under goroot
		// we need to correct certain relative paths
		if strings.HasPrefix(relpath, "src/pkg/") {
			relpath = relpath[len("src/pkg/"):]
		}
		template.HTMLEscape(w, []byte(pkgHandler.pattern[1:]+relpath)) // remove trailing '/' for relative URL
	case "url-src":
		template.HTMLEscape(w, []byte(relpath))
	case "url-pos":
		template.HTMLEscape(w, []byte(relpath))
		// selection ranges are of form "s=low:high"
		if low < high {
			fmt.Fprintf(w, "?s=%d:%d", low, high)
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
			fmt.Fprintf(w, "#L%d", line)
		}
	}
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


// Template formatter for "infoKind" format.
func infoKindFmt(w io.Writer, format string, x ...interface{}) {
	fmt.Fprintf(w, infoKinds[x[0].(SpotKind)]) // infoKind entries are html-escaped
}


// Template formatter for "infoLine" format.
func infoLineFmt(w io.Writer, format string, x ...interface{}) {
	info := x[0].(SpotInfo)
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
	fmt.Fprintf(w, "%d", line)
}


// Template formatter for "infoSnippet" format.
func infoSnippetFmt(w io.Writer, format string, x ...interface{}) {
	info := x[0].(SpotInfo)
	text := []byte(`<span class="alert">no snippet text available</span>`)
	if info.IsIndex() {
		index, _ := searchIndex.get()
		// no escaping of snippet text needed;
		// snippet text is escaped when generated
		text = index.(*Index).Snippet(info.Lori()).Text
	}
	w.Write(text)
}


// Template formatter for "padding" format.
func paddingFmt(w io.Writer, format string, x ...interface{}) {
	for i := x[0].(int); i > 0; i-- {
		fmt.Fprint(w, `<td width="25"></td>`)
	}
}


// Template formatter for "time" format.
func timeFmt(w io.Writer, format string, x ...interface{}) {
	template.HTMLEscape(w, []byte(time.SecondsToLocalTime(x[0].(int64)/1e9).String()))
}


// Template formatter for "dir/" format.
func dirslashFmt(w io.Writer, format string, x ...interface{}) {
	if x[0].(*os.FileInfo).IsDirectory() {
		w.Write([]byte{'/'})
	}
}


// Template formatter for "localname" format.
func localnameFmt(w io.Writer, format string, x ...interface{}) {
	_, localname := pathutil.Split(x[0].(string))
	template.HTMLEscape(w, []byte(localname))
}


// Template formatter for "numlines" format.
func numlinesFmt(w io.Writer, format string, x ...interface{}) {
	list := x[0].([]int)
	fmt.Fprintf(w, "%d", len(list))
}


var fmap = template.FormatterMap{
	"":             textFmt,
	"html-esc":     htmlEscFmt,
	"html-comment": htmlCommentFmt,
	"urlquery-esc": urlQueryEscFmt,
	"url-pkg":      urlFmt,
	"url-src":      urlFmt,
	"url-pos":      urlFmt,
	"infoKind":     infoKindFmt,
	"infoLine":     infoLineFmt,
	"infoSnippet":  infoSnippetFmt,
	"padding":      paddingFmt,
	"time":         timeFmt,
	"dir/":         dirslashFmt,
	"localname":    localnameFmt,
	"numlines":     numlinesFmt,
}


func readTemplate(name string) *template.Template {
	path := pathutil.Join(*goroot, "lib/godoc/"+name)
	data, err := ioutil.ReadFile(path)
	if err != nil {
		log.Fatalf("ReadFile %s: %v", path, err)
	}
	t, err := template.Parse(string(data), fmap)
	if err != nil {
		log.Fatalf("%s: %v", name, err)
	}
	return t
}


var (
	codewalkHTML,
	codewalkdirHTML,
	dirlistHTML,
	errorHTML,
	godocHTML,
	packageHTML,
	packageText,
	searchHTML,
	searchText *template.Template
)

func readTemplates() {
	// have to delay until after flags processing since paths depend on goroot
	codewalkHTML = readTemplate("codewalk.html")
	codewalkdirHTML = readTemplate("codewalkdir.html")
	dirlistHTML = readTemplate("dirlist.html")
	errorHTML = readTemplate("error.html")
	godocHTML = readTemplate("godoc.html")
	packageHTML = readTemplate("package.html")
	packageText = readTemplate("package.txt")
	searchHTML = readTemplate("search.html")
	searchText = readTemplate("search.txt")
}


// ----------------------------------------------------------------------------
// Generic HTML wrapper

func servePage(w http.ResponseWriter, title, subtitle, query string, content []byte) {
	d := struct {
		Title    string
		Subtitle string
		PkgRoots []string
		Query    string
		Version  string
		Menu     []byte
		Content  []byte
	}{
		title,
		subtitle,
		fsMap.PrefixList(),
		query,
		runtime.Version(),
		nil,
		content,
	}

	if err := godocHTML.Execute(w, &d); err != nil {
		log.Printf("godocHTML.Execute: %s", err)
	}
}


func serveText(w http.ResponseWriter, text []byte) {
	w.SetHeader("Content-Type", "text/plain; charset=utf-8")
	w.Write(text)
}


// ----------------------------------------------------------------------------
// Files

var (
	titleRx        = regexp.MustCompile(`<!-- title ([^\-]*)-->`)
	subtitleRx     = regexp.MustCompile(`<!-- subtitle ([^\-]*)-->`)
	firstCommentRx = regexp.MustCompile(`<!--([^\-]*)-->`)
)


func extractString(src []byte, rx *regexp.Regexp) (s string) {
	m := rx.FindSubmatch(src)
	if m != nil {
		s = strings.TrimSpace(string(m[1]))
	}
	return
}


func serveHTMLDoc(w http.ResponseWriter, r *http.Request, abspath, relpath string) {
	// get HTML body contents
	src, err := ioutil.ReadFile(abspath)
	if err != nil {
		log.Printf("ioutil.ReadFile: %s", err)
		serveError(w, r, relpath, err)
		return
	}

	// if it begins with "<!DOCTYPE " assume it is standalone
	// html that doesn't need the template wrapping.
	if bytes.HasPrefix(src, []byte("<!DOCTYPE ")) {
		w.Write(src)
		return
	}

	// if it's the language spec, add tags to EBNF productions
	if strings.HasSuffix(abspath, "go_spec.html") {
		var buf bytes.Buffer
		linkify(&buf, src)
		src = buf.Bytes()
	}

	// get title and subtitle, if any
	title := extractString(src, titleRx)
	if title == "" {
		// no title found; try first comment for backward-compatibility
		title = extractString(src, firstCommentRx)
	}
	subtitle := extractString(src, subtitleRx)

	servePage(w, title, subtitle, "", src)
}


func applyTemplate(t *template.Template, name string, data interface{}) []byte {
	var buf bytes.Buffer
	if err := t.Execute(&buf, data); err != nil {
		log.Printf("%s.Execute: %s", name, err)
	}
	return buf.Bytes()
}


func redirect(w http.ResponseWriter, r *http.Request) (redirected bool) {
	if canonical := pathutil.Clean(r.URL.Path) + "/"; r.URL.Path != canonical {
		http.Redirect(w, r, canonical, http.StatusMovedPermanently)
		redirected = true
	}
	return
}


func serveTextFile(w http.ResponseWriter, r *http.Request, abspath, relpath, title string) {
	src, err := ioutil.ReadFile(abspath)
	if err != nil {
		log.Printf("ioutil.ReadFile: %s", err)
		serveError(w, r, relpath, err)
		return
	}

	var buf bytes.Buffer
	buf.WriteString("<pre>")
	FormatText(&buf, src, 1, pathutil.Ext(abspath) == ".go", r.FormValue("h"), rangeSelection(r.FormValue("s")))
	buf.WriteString("</pre>")

	servePage(w, title+" "+relpath, "", "", buf.Bytes())
}


func serveDirectory(w http.ResponseWriter, r *http.Request, abspath, relpath string) {
	if redirect(w, r) {
		return
	}

	list, err := ioutil.ReadDir(abspath)
	if err != nil {
		log.Printf("ioutil.ReadDir: %s", err)
		serveError(w, r, relpath, err)
		return
	}

	for _, d := range list {
		if d.IsDirectory() {
			d.Size = 0
		}
	}

	contents := applyTemplate(dirlistHTML, "dirlistHTML", list)
	servePage(w, "Directory "+relpath, "", "", contents)
}


func serveFile(w http.ResponseWriter, r *http.Request) {
	relpath := r.URL.Path[1:] // serveFile URL paths start with '/'
	abspath := absolutePath(relpath, *goroot)

	// pick off special cases and hand the rest to the standard file server
	switch r.URL.Path {
	case "/":
		serveHTMLDoc(w, r, pathutil.Join(*goroot, "doc/root.html"), "doc/root.html")
		return

	case "/doc/root.html":
		// hide landing page from its real name
		http.Redirect(w, r, "/", http.StatusMovedPermanently)
		return
	}

	switch pathutil.Ext(abspath) {
	case ".html":
		if strings.HasSuffix(abspath, "/index.html") {
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

	dir, err := os.Lstat(abspath)
	if err != nil {
		log.Print(err)
		serveError(w, r, relpath, err)
		return
	}

	if dir != nil && dir.IsDirectory() {
		if redirect(w, r) {
			return
		}
		if index := abspath + "/index.html"; isTextFile(index) {
			serveHTMLDoc(w, r, index, relativePath(index))
			return
		}
		serveDirectory(w, r, abspath, relpath)
		return
	}

	if isTextFile(abspath) {
		serveTextFile(w, r, abspath, relpath, "Text file")
		return
	}

	fileServer.ServeHTTP(w, r)
}


// ----------------------------------------------------------------------------
// Packages

// Fake package file and name for commands. Contains the command documentation.
const fakePkgFile = "doc.go"
const fakePkgName = "documentation"

type PageInfoMode uint

const (
	exportsOnly PageInfoMode = 1 << iota // only keep exported stuff
	genDoc                               // generate documentation
)


type PageInfo struct {
	Dirname string          // directory containing the package
	PList   []string        // list of package names found
	FSet    *token.FileSet  // corresponding file set
	PAst    *ast.File       // nil if no single AST with package exports
	PDoc    *doc.PackageDoc // nil if no single package documentation
	Dirs    *DirList        // nil if no directory information
	DirTime int64           // directory time stamp in seconds since epoch
	IsPkg   bool            // false if this is not documenting a real package
	Err     os.Error        // directory read error or nil
}


func (info *PageInfo) IsEmpty() bool {
	return info.Err != nil || info.PAst == nil && info.PDoc == nil && info.Dirs == nil
}


type httpHandler struct {
	pattern string // url pattern; e.g. "/pkg/"
	fsRoot  string // file system root to which the pattern is mapped
	isPkg   bool   // true if this handler serves real package documentation (as opposed to command documentation)
}


// getPageInfo returns the PageInfo for a package directory abspath. If the
// parameter genAST is set, an AST containing only the package exports is
// computed (PageInfo.PAst), otherwise package documentation (PageInfo.Doc)
// is extracted from the AST. If there is no corresponding package in the
// directory, PageInfo.PAst and PageInfo.PDoc are nil. If there are no sub-
// directories, PageInfo.Dirs is nil. If a directory read error occurred,
// PageInfo.Err is set to the respective error but the error is not logged.
//
func (h *httpHandler) getPageInfo(abspath, relpath, pkgname string, mode PageInfoMode) PageInfo {
	// filter function to select the desired .go files
	filter := func(d *os.FileInfo) bool {
		// If we are looking at cmd documentation, only accept
		// the special fakePkgFile containing the documentation.
		return isPkgFile(d) && (h.isPkg || d.Name == fakePkgFile)
	}

	// get package ASTs
	fset := token.NewFileSet()
	pkgs, err := parser.ParseDir(fset, abspath, filter, parser.ParseComments)
	if err != nil && pkgs == nil {
		// only report directory read errors, ignore parse errors
		// (may be able to extract partial package information)
		return PageInfo{Dirname: abspath, Err: err}
	}

	// select package
	var pkg *ast.Package // selected package
	var plist []string   // list of other package (names), if any
	if len(pkgs) == 1 {
		// Exactly one package - select it.
		for _, p := range pkgs {
			pkg = p
		}

	} else if len(pkgs) > 1 {
		// Multiple packages - select the best matching package: The
		// 1st choice is the package with pkgname, the 2nd choice is
		// the package with dirname, and the 3rd choice is a package
		// that is not called "main" if there is exactly one such
		// package. Otherwise, don't select a package.
		dirpath, dirname := pathutil.Split(abspath)

		// If the dirname is "go" we might be in a sub-directory for
		// .go files - use the outer directory name instead for better
		// results.
		if dirname == "go" {
			_, dirname = pathutil.Split(pathutil.Clean(dirpath))
		}

		var choice3 *ast.Package
	loop:
		for _, p := range pkgs {
			switch {
			case p.Name == pkgname:
				pkg = p
				break loop // 1st choice; we are done
			case p.Name == dirname:
				pkg = p // 2nd choice
			case p.Name != "main":
				choice3 = p
			}
		}
		if pkg == nil && len(pkgs) == 2 {
			pkg = choice3
		}

		// Compute the list of other packages
		// (excluding the selected package, if any).
		plist = make([]string, len(pkgs))
		i := 0
		for name := range pkgs {
			if pkg == nil || name != pkg.Name {
				plist[i] = name
				i++
			}
		}
		plist = plist[0:i]
	}

	// compute package documentation
	var past *ast.File
	var pdoc *doc.PackageDoc
	if pkg != nil {
		if mode&exportsOnly != 0 {
			ast.PackageExports(pkg)
		}
		if mode&genDoc != 0 {
			pdoc = doc.NewPackageDoc(pkg, pathutil.Clean(relpath)) // no trailing '/' in importpath
		} else {
			past = ast.MergePackageFiles(pkg, ast.FilterUnassociatedComments)
		}
	}

	// get directory information
	var dir *Directory
	var timestamp int64
	if tree, ts := fsTree.get(); tree != nil && tree.(*Directory) != nil {
		// directory tree is present; lookup respective directory
		// (may still fail if the file system was updated and the
		// new directory tree has not yet been computed)
		dir = tree.(*Directory).lookup(abspath)
		timestamp = ts
	}
	if dir == nil {
		// the path may refer to a user-specified file system mapped
		// via fsMap; lookup that mapping and corresponding RWValue
		// if any
		var v *RWValue
		fsMap.Iterate(func(path string, value *RWValue) bool {
			if isParentOf(path, abspath) {
				// mapping found
				v = value
				return false
			}
			return true
		})
		if v != nil {
			// found a RWValue associated with a user-specified file
			// system; a non-nil RWValue stores a (possibly out-of-date)
			// directory tree for that file system
			if tree, ts := v.get(); tree != nil && tree.(*Directory) != nil {
				dir = tree.(*Directory).lookup(abspath)
				timestamp = ts
			}
		}
	}
	if dir == nil {
		// no directory tree present (too early after startup or
		// command-line mode); compute one level for this page
		// note: cannot use path filter here because in general
		//       it doesn't contain the fsTree path
		dir = newDirectory(abspath, nil, 1)
		timestamp = time.Seconds()
	}

	return PageInfo{abspath, plist, fset, past, pdoc, dir.listing(true), timestamp, h.isPkg, nil}
}


func (h *httpHandler) ServeHTTP(w http.ResponseWriter, r *http.Request) {
	if redirect(w, r) {
		return
	}

	relpath := r.URL.Path[len(h.pattern):]
	abspath := absolutePath(relpath, h.fsRoot)
	mode := exportsOnly
	if r.FormValue("m") != "src" {
		mode |= genDoc
	}
	info := h.getPageInfo(abspath, relpath, r.FormValue("p"), mode)
	if info.Err != nil {
		log.Print(info.Err)
		serveError(w, r, relpath, info.Err)
		return
	}

	if r.FormValue("f") == "text" {
		contents := applyTemplate(packageText, "packageText", info)
		serveText(w, contents)
		return
	}

	var title, subtitle string
	switch {
	case info.PAst != nil:
		title = "Package " + info.PAst.Name.Name
	case info.PDoc != nil:
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
	default:
		title = "Directory " + relativePath(info.Dirname)
		if *showTimestamps {
			subtitle = "Last update: " + time.SecondsToLocalTime(info.DirTime).String()
		}
	}

	contents := applyTemplate(packageHTML, "packageHTML", info)
	servePage(w, title, subtitle, "", contents)
}


// ----------------------------------------------------------------------------
// Search

var searchIndex RWValue

type SearchResult struct {
	Query string
	Alert string // error or warning message

	// identifier matches
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
		var err os.Error
		result.Hit, result.Alt, err = index.Lookup(query)
		if err != nil && *maxResults <= 0 {
			// ignore the error if full text search is enabled
			// since the query may be a valid regular expression
			result.Alert = "Error in query string: " + err.String()
			return
		}

		// full text search
		if *maxResults > 0 && query != "" {
			rx, err := regexp.Compile(query)
			if err != nil {
				result.Alert = "Error in query regular expression: " + err.String()
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
	if _, ts := fsModified.get(); timestamp < ts {
		// The index is older than the latest file system change
		// under godoc's observation. Indexing may be in progress
		// or start shortly (see indexer()).
		result.Alert = "Indexing in progress: result may be inaccurate"
	}

	return
}


func search(w http.ResponseWriter, r *http.Request) {
	query := strings.TrimSpace(r.FormValue("q"))
	result := lookup(query)

	if r.FormValue("f") == "text" {
		contents := applyTemplate(searchText, "searchText", result)
		serveText(w, contents)
		return
	}

	var title string
	if result.Hit != nil || len(result.Textual) > 0 {
		title = fmt.Sprintf(`Results for query %q`, query)
	} else {
		title = fmt.Sprintf(`No results found for query %q`, query)
	}

	contents := applyTemplate(searchHTML, "searchHTML", result)
	servePage(w, title, "", query, contents)
}


// ----------------------------------------------------------------------------
// Indexer

// invalidateIndex should be called whenever any of the file systems
// under godoc's observation change so that the indexer is kicked on.
//
func invalidateIndex() {
	fsModified.set(nil)
}


// indexUpToDate() returns true if the search index is not older
// than any of the file systems under godoc's observation.
//
func indexUpToDate() bool {
	_, fsTime := fsModified.get()
	_, siTime := searchIndex.get()
	return fsTime <= siTime
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
	c := make(chan string, 256) // asynchronous for fewer context switches
	go func() {
		feedDirnames(&fsTree, c)
		fsMap.Iterate(func(_ string, root *RWValue) bool {
			feedDirnames(root, c)
			return true
		})
		close(c)
	}()
	return c
}


func indexer() {
	for {
		if !indexUpToDate() {
			// index possibly out of date - make a new one
			if *verbose {
				log.Printf("updating index...")
			}
			start := time.Nanoseconds()
			index := NewIndex(fsDirnames(), *maxResults > 0)
			stop := time.Nanoseconds()
			searchIndex.set(index)
			if *verbose {
				secs := float64((stop-start)/1e6) / 1e3
				stats := index.Stats()
				log.Printf("index updated (%gs, %d bytes of source, %d files, %d lines, %d unique words, %d spots)",
					secs, stats.Bytes, stats.Files, stats.Lines, stats.Words, stats.Spots)
			}
			log.Printf("before GC: bytes = %d footprint = %d", runtime.MemStats.HeapAlloc, runtime.MemStats.Sys)
			runtime.GC()
			log.Printf("after  GC: bytes = %d footprint = %d", runtime.MemStats.HeapAlloc, runtime.MemStats.Sys)
		}
		var delay int64 = 60 * 1e9 // by default, try every 60s
		if *testDir != "" {
			// in test mode, try once a second for fast startup
			delay = 1 * 1e9
		}
		time.Sleep(delay)
	}
}
