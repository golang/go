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
	"go/printer"
	"go/token"
	"io"
	"io/ioutil"
	"log"
	"net/http"
	"net/url"
	"os"
	"path"
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
	goroot      = flag.String("goroot", runtime.GOROOT(), "Go root directory")
	testDir     = flag.String("testdir", "", "Go root subdirectory - for testing only (faster startups)")
	pkgPath     = flag.String("path", "", "additional package directories (colon-separated)")
	filter      = flag.String("filter", "", "filter file containing permitted package directory paths")
	filterMin   = flag.Int("filter_minutes", 0, "filter file update interval in minutes; disabled if <= 0")
	filterDelay delayTime // actual filter update interval in minutes; usually filterDelay == filterMin, but filterDelay may back off exponentially

	// layout control
	tabwidth       = flag.Int("tabwidth", 4, "tab width")
	showTimestamps = flag.Bool("timestamps", false, "show timestamps with directory listings")
	templateDir    = flag.String("templates", "", "directory containing alternate template files")

	// search index
	indexEnabled = flag.Bool("index", false, "enable search index")
	indexFiles   = flag.String("index_files", "", "glob pattern specifying index files;"+
			"if not empty, the index is read from these files in sorted order")
	maxResults    = flag.Int("maxresults", 10000, "maximum number of full text search results shown")
	indexThrottle = flag.Float64("index_throttle", 0.75, "index throttle value; 0.0 = no time allocated, 1.0 = full throttle")

	// file system mapping
	fs          FileSystem      // the underlying file system for godoc
	fsHttp      http.FileSystem // the underlying file system for http
	fsMap       Mapping         // user-defined mapping
	fsTree      RWValue         // *Directory tree of packages, updated with each sync
	pathFilter  RWValue         // filter used when building fsMap directory trees
	fsModified  RWValue         // timestamp of last call to invalidateIndex
	docMetadata RWValue         // mapping from paths to *Metadata

	// http handlers
	fileServer http.Handler // default file server
	cmdHandler httpHandler
	pkgHandler httpHandler
)

func initHandlers() {
	paths := filepath.SplitList(*pkgPath)
	gorootSrc := filepath.Join(build.Default.GOROOT, "src", "pkg")
	for _, p := range build.Default.SrcDirs() {
		if p != gorootSrc {
			paths = append(paths, p)
		}
	}
	fsMap.Init(paths)

	fileServer = http.FileServer(fsHttp)
	cmdHandler = httpHandler{"/cmd/", filepath.Join(*goroot, "src", "cmd"), false}
	pkgHandler = httpHandler{"/pkg/", filepath.Join(*goroot, "src", "pkg"), true}
}

func registerPublicHandlers(mux *http.ServeMux) {
	mux.Handle(cmdHandler.pattern, &cmdHandler)
	mux.Handle(pkgHandler.pattern, &pkgHandler)
	mux.HandleFunc("/doc/codewalk/", codewalk)
	mux.HandleFunc("/search", search)
	mux.Handle("/robots.txt", fileServer)
	mux.HandleFunc("/opensearch.xml", serveSearchDesc)
	mux.HandleFunc("/", serveFile)
}

func initFSTree() {
	dir := newDirectory(filepath.Join(*goroot, *testDir), nil, -1)
	if dir == nil {
		log.Println("Warning: FSTree is nil")
		return
	}
	fsTree.set(dir)
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
func readDirList(filename string) ([]string, error) {
	contents, err := ReadFile(fs, filename)
	if err != nil {
		return nil, err
	}
	// create a sorted list of valid directory names
	filter := func(path string) bool {
		d, e := fs.Lstat(path)
		if e != nil && err == nil {
			// remember first error and return it from readDirList
			// so we have at least some information if things go bad
			err = e
		}
		return e == nil && isPkgDir(d)
	}
	list := canonicalizePaths(strings.Split(string(contents), "\n"), filter)
	// for each parent path, remove all its children q
	// (requirement for binary search to work when filtering)
	i := 0
	for _, q := range list {
		if i == 0 || !isParentOf(list[i-1], q) {
			list[i] = q
			i++
		}
	}
	return list[0:i], err
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
		filterDelay.backoff(24 * time.Hour) // back off exponentially, but try at least once a day
	} else {
		filterDelay.set(*filterMin) // revert to regular filter update schedule
	}
}

func initDirTrees() {
	// setup initial path filter
	if *filter != "" {
		list, err := readDirList(*filter)
		if err != nil {
			log.Printf("readDirList(%s): %s", *filter, err)
		}
		if *verbose || len(list) == 0 {
			log.Printf("found %d directory paths in file %s", len(list), *filter)
		}
		setPathFilter(list)
	}

	go updateMappedDirs(getPathFilter()) // use filter for speed

	// start filter update goroutine, if enabled.
	if *filter != "" && *filterMin > 0 {
		filterDelay.set(time.Duration(*filterMin) * time.Minute) // initial filter update delay
		go func() {
			for {
				if *verbose {
					log.Printf("start update of %s", *filter)
				}
				updateFilterFile()
				delay, _ := filterDelay.get()
				dt := delay.(time.Duration)
				if *verbose {
					log.Printf("next filter update in %s", dt)
				}
				time.Sleep(dt)
			}
		}()
	}
}

// ----------------------------------------------------------------------------
// Path mapping

// Absolute paths are file system paths (backslash-separated on Windows),
// but relative paths are always slash-separated.

func absolutePath(relpath, defaultRoot string) string {
	abspath := fsMap.ToAbsolute(relpath)
	if abspath == "" {
		// no user-defined mapping found; use default mapping
		abspath = filepath.Join(defaultRoot, filepath.FromSlash(relpath))
	}
	return abspath
}

func relativeURL(abspath string) string {
	relpath := fsMap.ToRelative(abspath)
	if relpath == "" {
		// prefix must end in a path separator
		prefix := *goroot
		if len(prefix) > 0 && prefix[len(prefix)-1] != filepath.Separator {
			prefix += string(filepath.Separator)
		}
		if strings.HasPrefix(abspath, prefix) {
			// no user-defined mapping found; use default mapping
			relpath = filepath.ToSlash(abspath[len(prefix):])
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
	_, localname := filepath.Split(path)
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

func nodeFunc(node interface{}, fset *token.FileSet) string {
	var buf bytes.Buffer
	writeNode(&buf, fset, node)
	return buf.String()
}

func node_htmlFunc(node interface{}, fset *token.FileSet) string {
	var buf1 bytes.Buffer
	writeNode(&buf1, fset, node)
	var buf2 bytes.Buffer
	FormatText(&buf2, buf1.Bytes(), -1, true, "", nil)
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

func example_htmlFunc(funcName string, examples []*doc.Example, fset *token.FileSet) string {
	var buf bytes.Buffer
	for _, eg := range examples {
		name := eg.Name

		// strip lowercase braz in Foo_braz or Foo_Bar_braz from name
		// while keeping uppercase Braz in Foo_Braz
		if i := strings.LastIndex(name, "_"); i != -1 {
			if i < len(name)-1 && !startsWithUppercase(name[i+1:]) {
				name = name[:i]
			}
		}

		if name != funcName {
			continue
		}

		// print code
		cnode := &printer.CommentedNode{Node: eg.Code, Comments: eg.Comments}
		code := node_htmlFunc(cnode, fset)
		out := eg.Output

		// additional formatting if this is a function body
		if n := len(code); n >= 2 && code[0] == '{' && code[n-1] == '}' {
			// remove surrounding braces
			code = code[1 : n-1]
			// unindent
			code = strings.Replace(code, "\n    ", "\n", -1)
			// remove output comment
			if loc := exampleOutputRx.FindStringIndex(code); loc != nil {
				code = strings.TrimSpace(code[:loc[0]])
			}
		} else {
			// drop output, as the output comment will appear in the code
			out = ""
		}

		err := exampleHTML.Execute(&buf, struct {
			Name, Doc, Code, Output string
		}{eg.Name, eg.Doc, code, out})
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
	relpath := relativeURL(path)
	// because of the irregular mapping under goroot
	// we need to correct certain relative paths
	if strings.HasPrefix(relpath, "src/pkg/") {
		relpath = relpath[len("src/pkg/"):]
	}
	return pkgHandler.pattern[1:] + relpath // remove trailing '/' for relative URL
}

func posLink_urlFunc(node ast.Node, fset *token.FileSet) string {
	var relpath string
	var line int
	var low, high int // selection

	if p := node.Pos(); p.IsValid() {
		pos := fset.Position(p)
		relpath = relativeURL(pos.Filename)
		line = pos.Line
		low = pos.Offset
	}
	if p := node.End(); p.IsValid() {
		high = fset.Position(p).Offset
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

// fmap describes the template functions installed with all godoc templates.
// Convention: template function names ending in "_html" or "_url" produce
//             HTML- or URL-escaped strings; all other function results may
//             require explicit escaping in the template.
var fmap = template.FuncMap{
	// various helpers
	"filename": filenameFunc,
	"repeat":   strings.Repeat,

	// accss to FileInfos (directory listings)
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
	"srcLink":     relativeURL,
	"posLink_url": posLink_urlFunc,

	// formatting of Examples
	"example_html":   example_htmlFunc,
	"example_name":   example_nameFunc,
	"example_suffix": example_suffixFunc,
}

func readTemplate(name string) *template.Template {
	path := filepath.Join(*goroot, "lib", "godoc", name)
	if *templateDir != "" {
		defaultpath := path
		path = filepath.Join(*templateDir, name)
		if _, err := fs.Stat(path); err != nil {
			log.Print("readTemplate:", err)
			path = defaultpath
		}
	}

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

func servePage(w http.ResponseWriter, title, subtitle, query string, content []byte) {
	d := struct {
		Title     string
		Subtitle  string
		PkgRoots  []string
		SearchBox bool
		Query     string
		Version   string
		Menu      []byte
		Content   []byte
	}{
		title,
		subtitle,
		fsMap.PrefixList(),
		*indexEnabled,
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

	// if it's the language spec, add tags to EBNF productions
	if strings.HasSuffix(abspath, "go_spec.html") {
		var buf bytes.Buffer
		Linkify(&buf, src)
		src = buf.Bytes()
	}

	servePage(w, meta.Title, meta.Subtitle, "", src)
}

func applyTemplate(t *template.Template, name string, data interface{}) []byte {
	var buf bytes.Buffer
	if err := t.Execute(&buf, data); err != nil {
		log.Printf("%s.Execute: %s", name, err)
	}
	return buf.Bytes()
}

func redirect(w http.ResponseWriter, r *http.Request) (redirected bool) {
	canonical := path.Clean(r.URL.Path)
	if !strings.HasSuffix("/", canonical) {
		canonical += "/"
	}
	if r.URL.Path != canonical {
		http.Redirect(w, r, canonical, http.StatusMovedPermanently)
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

	var buf bytes.Buffer
	buf.WriteString("<pre>")
	FormatText(&buf, src, 1, filepath.Ext(abspath) == ".go", r.FormValue("h"), rangeSelection(r.FormValue("s")))
	buf.WriteString("</pre>")

	servePage(w, title+" "+relpath, "", "", buf.Bytes())
}

func serveDirectory(w http.ResponseWriter, r *http.Request, abspath, relpath string) {
	if redirect(w, r) {
		return
	}

	list, err := fs.ReadDir(abspath)
	if err != nil {
		log.Printf("ReadDir: %s", err)
		serveError(w, r, relpath, err)
		return
	}

	contents := applyTemplate(dirlistHTML, "dirlistHTML", list)
	servePage(w, "Directory "+relpath, "", "", contents)
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

	relpath = relpath[1:] // strip leading slash
	abspath := absolutePath(relpath, *goroot)

	switch path.Ext(relpath) {
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
		if index := filepath.Join(abspath, "index.html"); isTextFile(index) {
			serveHTMLDoc(w, r, index, relativeURL(index))
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

// Fake package file and name for commands. Contains the command documentation.
const fakePkgFile = "doc.go"
const fakePkgName = "documentation"

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
	Dirname  string         // directory containing the package
	PList    []string       // list of package names found
	FSet     *token.FileSet // corresponding file set
	PAst     *ast.File      // nil if no single AST with package exports
	PDoc     *doc.Package   // nil if no single package documentation
	Examples []*doc.Example // nil if no example code
	Dirs     *DirList       // nil if no directory information
	DirTime  time.Time      // directory time stamp
	DirFlat  bool           // if set, show directory in a flat (non-indented) manner
	IsPkg    bool           // false if this is not documenting a real package
	Err      error          // I/O error or nil
}

func (info *PageInfo) IsEmpty() bool {
	return info.Err != nil || info.PAst == nil && info.PDoc == nil && info.Dirs == nil
}

type httpHandler struct {
	pattern string // url pattern; e.g. "/pkg/"
	fsRoot  string // file system root to which the pattern is mapped
	isPkg   bool   // true if this handler serves real package documentation (as opposed to command documentation)
}

// fsReadDir implements ReadDir for the go/build package.
func fsReadDir(dir string) ([]os.FileInfo, error) {
	return fs.ReadDir(dir)
}

// fsOpenFile implements OpenFile for the go/build package.
func fsOpenFile(name string) (r io.ReadCloser, err error) {
	data, err := ReadFile(fs, name)
	if err != nil {
		return nil, err
	}
	return ioutil.NopCloser(bytes.NewReader(data)), nil
}

func inList(name string, list []string) bool {
	for _, l := range list {
		if name == l {
			return true
		}
	}
	return false
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
	var pkgFiles []string

	// If we're showing the default package, restrict to the ones
	// that would be used when building the package on this
	// system.  This makes sure that if there are separate
	// implementations for, say, Windows vs Unix, we don't
	// jumble them all together.
	if pkgname == "" {
		// Note: Uses current binary's GOOS/GOARCH.
		// To use different pair, such as if we allowed the user
		// to choose, set ctxt.GOOS and ctxt.GOARCH before
		// calling ctxt.ScanDir.
		ctxt := build.Default
		ctxt.IsAbsPath = path.IsAbs
		ctxt.ReadDir = fsReadDir
		ctxt.OpenFile = fsOpenFile
		dir, err := ctxt.ImportDir(abspath, 0)
		if err == nil {
			pkgFiles = append(dir.GoFiles, dir.CgoFiles...)
		}
	}

	// filter function to select the desired .go files
	filter := func(d os.FileInfo) bool {
		// Only Go files.
		if !isPkgFile(d) {
			return false
		}
		// If we are looking at cmd documentation, only accept
		// the special fakePkgFile containing the documentation.
		if !h.isPkg {
			return d.Name() == fakePkgFile
		}
		// Also restrict file list to pkgFiles.
		return pkgFiles == nil || inList(d.Name(), pkgFiles)
	}

	// get package ASTs
	fset := token.NewFileSet()
	pkgs, err := parseDir(fset, abspath, filter)
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
		dirpath, dirname := filepath.Split(abspath)

		// If the dirname is "go" we might be in a sub-directory for
		// .go files - use the outer directory name instead for better
		// results.
		if dirname == "go" {
			_, dirname = filepath.Split(filepath.Clean(dirpath))
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
		sort.Strings(plist)
	}

	// get examples from *_test.go files
	var examples []*doc.Example
	filter = func(d os.FileInfo) bool {
		return isGoFile(d) && strings.HasSuffix(d.Name(), "_test.go")
	}
	if testpkgs, err := parseDir(fset, abspath, filter); err != nil {
		log.Println("parsing test files:", err)
	} else {
		for _, testpkg := range testpkgs {
			var files []*ast.File
			for _, f := range testpkg.Files {
				files = append(files, f)
			}
			examples = append(examples, doc.Examples(files...)...)
		}
	}

	// compute package documentation
	var past *ast.File
	var pdoc *doc.Package
	if pkg != nil {
		if mode&showSource == 0 {
			// show extracted documentation
			var m doc.Mode
			if mode&noFiltering != 0 {
				m = doc.AllDecls
			}
			if mode&allMethods != 0 {
				m |= doc.AllMethods
			}
			pdoc = doc.New(pkg, path.Clean(relpath), m) // no trailing '/' in importpath
		} else {
			// show source code
			// TODO(gri) Consider eliminating export filtering in this mode,
			//           or perhaps eliminating the mode altogether.
			if mode&noFiltering == 0 {
				ast.PackageExports(pkg)
			}
			past = ast.MergePackageFiles(pkg, ast.FilterUnassociatedComments)
		}
	}

	// get directory information
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
		timestamp = time.Now()
	}

	return PageInfo{
		Dirname:  abspath,
		PList:    plist,
		FSet:     fset,
		PAst:     past,
		PDoc:     pdoc,
		Examples: examples,
		Dirs:     dir.listing(true),
		DirTime:  timestamp,
		DirFlat:  mode&flatDir != 0,
		IsPkg:    h.isPkg,
		Err:      nil,
	}
}

func (h *httpHandler) ServeHTTP(w http.ResponseWriter, r *http.Request) {
	if redirect(w, r) {
		return
	}

	relpath := path.Clean(r.URL.Path[len(h.pattern):])
	abspath := absolutePath(relpath, h.fsRoot)
	mode := getPageInfoMode(r)
	if relpath == builtinPkgPath {
		mode = noFiltering
	}
	info := h.getPageInfo(abspath, relpath, r.FormValue("p"), mode)
	if info.Err != nil {
		log.Print(info.Err)
		serveError(w, r, relpath, info.Err)
		return
	}

	if mode&noHtml != 0 {
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
		case info.IsPkg:
			title = "Package " + info.PDoc.Name
		case info.PDoc.Name == fakePkgName:
			// assume that the directory name is the command name
			_, pkgname := path.Split(relpath)
			title = "Command " + pkgname
		default:
			title = "Command " + info.PDoc.Name
		}
	default:
		title = "Directory " + relativeURL(info.Dirname)
		if *showTimestamps {
			subtitle = "Last update: " + info.DirTime.String()
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
// Documentation Metadata

type Metadata struct {
	Title    string
	Subtitle string
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
			name := filepath.Join(dir, fi.Name())
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
			meta.filePath = filepath.Join("/", name[len(*goroot):])
			if meta.Path == "" {
				// If no Path, canonical path is actual path.
				meta.Path = meta.filePath
			}
			// Store under both paths.
			metadata[meta.Path] = &meta
			metadata[meta.filePath] = &meta
		}
	}
	scan(filepath.Join(*goroot, "doc"))
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
		return m.(map[string]*Metadata)[relpath]
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

func readIndex(filenames string) error {
	matches, err := filepath.Glob(filenames)
	if err != nil {
		return err
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
