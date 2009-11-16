// Copyright 2009 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

package main

import (
	"bytes";
	"flag";
	"fmt";
	"go/ast";
	"go/doc";
	"go/parser";
	"go/printer";
	"go/scanner";
	"go/token";
	"http";
	"io";
	"log";
	"os";
	pathutil "path";
	"strings";
	"sync";
	"template";
	"time";
	"unicode";
	"utf8";
)


// ----------------------------------------------------------------------------
// Support types

// An RWValue wraps a value and permits mutually exclusive
// access to it and records the time the value was last set.
type RWValue struct {
	mutex		sync.RWMutex;
	value		interface{};
	timestamp	int64;	// time of last set(), in seconds since epoch
}


func (v *RWValue) set(value interface{}) {
	v.mutex.Lock();
	v.value = value;
	v.timestamp = time.Seconds();
	v.mutex.Unlock();
}


func (v *RWValue) get() (interface{}, int64) {
	v.mutex.RLock();
	defer v.mutex.RUnlock();
	return v.value, v.timestamp;
}


// ----------------------------------------------------------------------------
// Globals

type delayTime struct {
	RWValue;
}


func (dt *delayTime) backoff(max int) {
	dt.mutex.Lock();
	v := dt.value.(int) * 2;
	if v > max {
		v = max
	}
	dt.value = v;
	dt.mutex.Unlock();
}


var (
	verbose	= flag.Bool("v", false, "verbose mode");

	// file system roots
	goroot		string;
	cmdroot		= flag.String("cmdroot", "src/cmd", "root command source directory (if unrooted, relative to goroot)");
	pkgroot		= flag.String("pkgroot", "src/pkg", "root package source directory (if unrooted, relative to goroot)");
	tmplroot	= flag.String("tmplroot", "lib/godoc", "root template directory (if unrooted, relative to goroot)");

	// layout control
	tabwidth	= flag.Int("tabwidth", 4, "tab width");
)


var fsTree RWValue	// *Directory tree of packages, updated with each sync


func init() {
	goroot = os.Getenv("GOROOT");
	if goroot == "" {
		goroot = pathutil.Join(os.Getenv("HOME"), "go")
	}
	flag.StringVar(&goroot, "goroot", goroot, "Go root directory");
}


// ----------------------------------------------------------------------------
// Predicates and small utility functions

func isGoFile(dir *os.Dir) bool {
	return dir.IsRegular() &&
		!strings.HasPrefix(dir.Name, ".") &&	// ignore .files
		pathutil.Ext(dir.Name) == ".go"
}


func isPkgFile(dir *os.Dir) bool {
	return isGoFile(dir) &&
		!strings.HasSuffix(dir.Name, "_test.go")	// ignore test files
}


func isPkgDir(dir *os.Dir) bool {
	return dir.IsDirectory() && len(dir.Name) > 0 && dir.Name[0] != '_'
}


func pkgName(filename string) string {
	file, err := parse(filename, parser.PackageClauseOnly);
	if err != nil || file == nil {
		return ""
	}
	return file.Name.Value;
}


func htmlEscape(s string) string {
	var buf bytes.Buffer;
	template.HTMLEscape(&buf, strings.Bytes(s));
	return buf.String();
}


func firstSentence(s string) string {
	i := -1;	// index+1 of first period
	j := -1;	// index+1 of first period that is followed by white space
	prev := 'A';
	for k, ch := range s {
		k1 := k + 1;
		if ch == '.' {
			if i < 0 {
				i = k1	// first period
			}
			if k1 < len(s) && s[k1] <= ' ' {
				if j < 0 {
					j = k1	// first period followed by white space
				}
				if !unicode.IsUpper(prev) {
					j = k1;
					break;
				}
			}
		}
		prev = ch;
	}

	if j < 0 {
		// use the next best period
		j = i;
		if j < 0 {
			// no period at all, use the entire string
			j = len(s)
		}
	}

	return s[0:j];
}


// ----------------------------------------------------------------------------
// Package directories

type Directory struct {
	Depth	int;
	Path	string;	// includes Name
	Name	string;
	Text	string;		// package documentation, if any
	Dirs	[]*Directory;	// subdirectories
}


func newDirTree(path, name string, depth, maxDepth int) *Directory {
	if depth >= maxDepth {
		// return a dummy directory so that the parent directory
		// doesn't get discarded just because we reached the max
		// directory depth
		return &Directory{depth, path, name, "", nil}
	}

	list, _ := io.ReadDir(path);	// ignore errors

	// determine number of subdirectories and package files
	ndirs := 0;
	nfiles := 0;
	text := "";
	for _, d := range list {
		switch {
		case isPkgDir(d):
			ndirs++
		case isPkgFile(d):
			nfiles++;
			if text == "" {
				// no package documentation yet; take the first found
				file, err := parser.ParseFile(pathutil.Join(path, d.Name), nil,
					parser.ParseComments|parser.PackageClauseOnly);
				if err == nil &&
					// Also accept fakePkgName, so we get synopses for commmands.
					// Note: This may lead to incorrect results if there is a
					// (left-over) "documentation" package somewhere in a package
					// directory of different name, but this is very unlikely and
					// against current conventions.
					(file.Name.Value == name || file.Name.Value == fakePkgName) &&
					file.Doc != nil {
					// found documentation; extract a synopsys
					text = firstSentence(doc.CommentText(file.Doc))
				}
			}
		}
	}

	// create subdirectory tree
	var dirs []*Directory;
	if ndirs > 0 {
		dirs = make([]*Directory, ndirs);
		i := 0;
		for _, d := range list {
			if isPkgDir(d) {
				dd := newDirTree(pathutil.Join(path, d.Name), d.Name, depth+1, maxDepth);
				if dd != nil {
					dirs[i] = dd;
					i++;
				}
			}
		}
		dirs = dirs[0:i];
	}

	// if there are no package files and no subdirectories
	// (with package files), ignore the directory
	if nfiles == 0 && len(dirs) == 0 {
		return nil
	}

	return &Directory{depth, path, name, text, dirs};
}


// newDirectory creates a new package directory tree with at most maxDepth
// levels, anchored at root which is relative to goroot. The result tree
// only contains directories that contain package files or that contain
// subdirectories containing package files (transitively).
//
func newDirectory(root string, maxDepth int) *Directory {
	d, err := os.Lstat(root);
	if err != nil || !isPkgDir(d) {
		return nil
	}
	return newDirTree(root, d.Name, 0, maxDepth);
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
	c := make(chan *Directory);
	go func() {
		dir.walk(c, skipRoot);
		close(c);
	}();
	return c;
}


// lookup looks for the *Directory for a given path, relative to dir.
func (dir *Directory) lookup(path string) *Directory {
	path = pathutil.Clean(path);	// no trailing '/'

	if dir == nil || path == "" || path == "." {
		return dir
	}

	dpath, dname := pathutil.Split(path);
	if dpath == "" {
		// directory-local name
		for _, d := range dir.Dirs {
			if dname == d.Name {
				return d
			}
		}
		return nil;
	}

	return dir.lookup(dpath).lookup(dname);
}


// DirEntry describes a directory entry. The Depth and Height values
// are useful for presenting an entry in an indented fashion.
//
type DirEntry struct {
	Depth		int;	// >= 0
	Height		int;	// = DirList.MaxHeight - Depth, > 0
	Path		string;	// includes Name, relative to DirList root
	Name		string;
	Synopsis	string;
}


type DirList struct {
	MaxHeight	int;	// directory tree height, > 0
	List		[]DirEntry;
}


// listing creates a (linear) directory listing from a directory tree.
// If skipRoot is set, the root directory itself is excluded from the list.
//
func (root *Directory) listing(skipRoot bool) *DirList {
	if root == nil {
		return nil
	}

	// determine number of entries n and maximum height
	n := 0;
	minDepth := 1 << 30;	// infinity
	maxDepth := 0;
	for d := range root.iter(skipRoot) {
		n++;
		if minDepth > d.Depth {
			minDepth = d.Depth
		}
		if maxDepth < d.Depth {
			maxDepth = d.Depth
		}
	}
	maxHeight := maxDepth - minDepth + 1;

	if n == 0 {
		return nil
	}

	// create list
	list := make([]DirEntry, n);
	i := 0;
	for d := range root.iter(skipRoot) {
		p := &list[i];
		p.Depth = d.Depth - minDepth;
		p.Height = maxHeight - p.Depth;
		// the path is relative to root.Path - remove the root.Path
		// prefix (the prefix should always be present but avoid
		// crashes and check)
		path := d.Path;
		if strings.HasPrefix(d.Path, root.Path) {
			path = d.Path[len(root.Path):len(d.Path)]
		}
		// remove trailing '/' if any - path must be relative
		if len(path) > 0 && path[0] == '/' {
			path = path[1:len(path)]
		}
		p.Path = path;
		p.Name = d.Name;
		p.Synopsis = d.Text;
		i++;
	}

	return &DirList{maxHeight, list};
}


func listing(dirs []*os.Dir) *DirList {
	list := make([]DirEntry, len(dirs)+1);
	list[0] = DirEntry{0, 1, "..", "..", ""};
	for i, d := range dirs {
		p := &list[i+1];
		p.Depth = 0;
		p.Height = 1;
		p.Path = d.Name;
		p.Name = d.Name;
	}
	return &DirList{1, list};
}


// ----------------------------------------------------------------------------
// Parsing

// A single error in the parsed file.
type parseError struct {
	src	[]byte;	// source before error
	line	int;	// line number of error
	msg	string;	// error message
}


// All the errors in the parsed file, plus surrounding source code.
// Each error has a slice giving the source text preceding it
// (starting where the last error occurred).  The final element in list[]
// has msg = "", to give the remainder of the source code.
// This data structure is handed to the templates parseerror.txt and parseerror.html.
//
type parseErrors struct {
	filename	string;		// path to file
	list		[]parseError;	// the errors
	src		[]byte;		// the file's entire source code
}


// Parses a file (path) and returns the corresponding AST and
// a sorted list (by file position) of errors, if any.
//
func parse(path string, mode uint) (*ast.File, *parseErrors) {
	src, err := io.ReadFile(path);
	if err != nil {
		log.Stderrf("%v", err);
		errs := []parseError{parseError{nil, 0, err.String()}};
		return nil, &parseErrors{path, errs, nil};
	}

	prog, err := parser.ParseFile(path, src, mode);
	if err != nil {
		var errs []parseError;
		if errors, ok := err.(scanner.ErrorList); ok {
			// convert error list (already sorted)
			// TODO(gri) If the file contains //line comments, the errors
			//           may not be sorted in increasing file offset value
			//           which will lead to incorrect output.
			errs = make([]parseError, len(errors)+1);	// +1 for final fragment of source
			offs := 0;
			for i, r := range errors {
				// Should always be true, but check for robustness.
				if 0 <= r.Pos.Offset && r.Pos.Offset <= len(src) {
					errs[i].src = src[offs:r.Pos.Offset];
					offs = r.Pos.Offset;
				}
				errs[i].line = r.Pos.Line;
				errs[i].msg = r.Msg;
			}
			errs[len(errors)].src = src[offs:len(src)];
		} else {
			// single error of unspecified type
			errs = make([]parseError, 2);
			errs[0] = parseError{[]byte{}, 0, err.String()};
			errs[1].src = src;
		}
		return nil, &parseErrors{path, errs, src};
	}

	return prog, nil;
}


// ----------------------------------------------------------------------------
// HTML formatting support

// Styler implements a printer.Styler.
type Styler struct {
	linetags	bool;
	highlight	string;
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
	return;
}


func (s *Styler) Comment(c *ast.Comment, line []byte) (text []byte, tag printer.HTMLTag) {
	text = line;
	// minimal syntax-coloring of comments for now - people will want more
	// (don't do anything more until there's a button to turn it on/off)
	tag = printer.HTMLTag{`<span class="comment">`, "</span>"};
	return;
}


func (s *Styler) BasicLit(x *ast.BasicLit) (text []byte, tag printer.HTMLTag) {
	text = x.Value;
	return;
}


func (s *Styler) Ident(id *ast.Ident) (text []byte, tag printer.HTMLTag) {
	text = strings.Bytes(id.Value);
	if s.highlight == id.Value {
		tag = printer.HTMLTag{"<span class=highlight>", "</span>"}
	}
	return;
}


func (s *Styler) Token(tok token.Token) (text []byte, tag printer.HTMLTag) {
	text = strings.Bytes(tok.String());
	return;
}


// ----------------------------------------------------------------------------
// Templates

// Write an AST-node to w; optionally html-escaped.
func writeNode(w io.Writer, node interface{}, html bool, styler printer.Styler) {
	mode := printer.UseSpaces;
	if html {
		mode |= printer.GenHTML
	}
	(&printer.Config{mode, *tabwidth, styler}).Fprint(w, node);
}


// Write text to w; optionally html-escaped.
func writeText(w io.Writer, text []byte, html bool) {
	if html {
		template.HTMLEscape(w, text);
		return;
	}
	w.Write(text);
}


// Write anything to w; optionally html-escaped.
func writeAny(w io.Writer, x interface{}, html bool) {
	switch v := x.(type) {
	case []byte:
		writeText(w, v, html)
	case string:
		writeText(w, strings.Bytes(v), html)
	case ast.Decl:
		writeNode(w, v, html, &defaultStyler)
	case ast.Expr:
		writeNode(w, v, html, &defaultStyler)
	default:
		if html {
			var buf bytes.Buffer;
			fmt.Fprint(&buf, x);
			writeText(w, buf.Bytes(), true);
		} else {
			fmt.Fprint(w, x)
		}
	}
}


// Template formatter for "html" format.
func htmlFmt(w io.Writer, x interface{}, format string) {
	writeAny(w, x, true)
}


// Template formatter for "html-comment" format.
func htmlCommentFmt(w io.Writer, x interface{}, format string) {
	var buf bytes.Buffer;
	writeAny(&buf, x, false);
	doc.ToHTML(w, buf.Bytes());	// does html-escaping
}


// Template formatter for "" (default) format.
func textFmt(w io.Writer, x interface{}, format string) {
	writeAny(w, x, false)
}


func removePrefix(s, prefix string) string {
	if strings.HasPrefix(s, prefix) {
		return s[len(prefix):len(s)]
	}
	return s;
}


// Template formatter for "path" format.
func pathFmt(w io.Writer, x interface{}, format string) {
	// TODO(gri): Need to find a better solution for this.
	//            This will not work correctly if *cmdroot
	//            or *pkgroot change.
	writeAny(w, removePrefix(x.(string), "src"), true)
}


// Template formatter for "link" format.
func linkFmt(w io.Writer, x interface{}, format string) {
	type Positioner interface {
		Pos() token.Position;
	}
	if node, ok := x.(Positioner); ok {
		pos := node.Pos();
		if pos.IsValid() {
			// line id's in html-printed source are of the
			// form "L%d" where %d stands for the line number
			fmt.Fprintf(w, "/%s#L%d", htmlEscape(pos.Filename), pos.Line)
		}
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
	fmt.Fprintf(w, infoKinds[x.(SpotKind)])	// infoKind entries are html-escaped
}


// Template formatter for "infoLine" format.
func infoLineFmt(w io.Writer, x interface{}, format string) {
	info := x.(SpotInfo);
	line := info.Lori();
	if info.IsIndex() {
		index, _ := searchIndex.get();
		line = index.(*Index).Snippet(line).Line;
	}
	fmt.Fprintf(w, "%d", line);
}


// Template formatter for "infoSnippet" format.
func infoSnippetFmt(w io.Writer, x interface{}, format string) {
	info := x.(SpotInfo);
	text := `<span class="alert">no snippet text available</span>`;
	if info.IsIndex() {
		index, _ := searchIndex.get();
		// no escaping of snippet text needed;
		// snippet text is escaped when generated
		text = index.(*Index).Snippet(info.Lori()).Text;
	}
	fmt.Fprint(w, text);
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


var fmap = template.FormatterMap{
	"": textFmt,
	"html": htmlFmt,
	"html-comment": htmlCommentFmt,
	"path": pathFmt,
	"link": linkFmt,
	"infoKind": infoKindFmt,
	"infoLine": infoLineFmt,
	"infoSnippet": infoSnippetFmt,
	"padding": paddingFmt,
	"time": timeFmt,
}


func readTemplate(name string) *template.Template {
	path := pathutil.Join(*tmplroot, name);
	data, err := io.ReadFile(path);
	if err != nil {
		log.Exitf("ReadFile %s: %v", path, err)
	}
	t, err := template.Parse(string(data), fmap);
	if err != nil {
		log.Exitf("%s: %v", name, err)
	}
	return t;
}


var (
	dirlistHTML,
		godocHTML,
		packageHTML,
		packageText,
		parseerrorHTML,
		parseerrorText,
		searchHTML *template.Template;
)

func readTemplates() {
	// have to delay until after flags processing,
	// so that main has chdir'ed to goroot.
	dirlistHTML = readTemplate("dirlist.html");
	godocHTML = readTemplate("godoc.html");
	packageHTML = readTemplate("package.html");
	packageText = readTemplate("package.txt");
	parseerrorHTML = readTemplate("parseerror.html");
	parseerrorText = readTemplate("parseerror.txt");
	searchHTML = readTemplate("search.html");
}


// ----------------------------------------------------------------------------
// Generic HTML wrapper

func servePage(c *http.Conn, title, query string, content []byte) {
	type Data struct {
		Title		string;
		Timestamp	uint64;	// int64 to be compatible with os.Dir.Mtime_ns
		Query		string;
		Content		[]byte;
	}

	_, ts := fsTree.get();
	d := Data{
		Title: title,
		Timestamp: uint64(ts) * 1e9,	// timestamp in ns
		Query: query,
		Content: content,
	};

	if err := godocHTML.Execute(&d, c); err != nil {
		log.Stderrf("godocHTML.Execute: %s", err)
	}
}


func serveText(c *http.Conn, text []byte) {
	c.SetHeader("content-type", "text/plain; charset=utf-8");
	c.Write(text);
}


// ----------------------------------------------------------------------------
// Files

var (
	tagBegin	= strings.Bytes("<!--");
	tagEnd		= strings.Bytes("-->");
)

// commentText returns the text of the first HTML comment in src.
func commentText(src []byte) (text string) {
	i := bytes.Index(src, tagBegin);
	j := bytes.Index(src, tagEnd);
	if i >= 0 && j >= i+len(tagBegin) {
		text = string(bytes.TrimSpace(src[i+len(tagBegin) : j]))
	}
	return;
}


func serveHTMLDoc(c *http.Conn, r *http.Request, path string) {
	// get HTML body contents
	src, err := io.ReadFile(path);
	if err != nil {
		log.Stderrf("%v", err);
		http.NotFound(c, r);
		return;
	}

	// if it's the language spec, add tags to EBNF productions
	if strings.HasSuffix(path, "go_spec.html") {
		var buf bytes.Buffer;
		linkify(&buf, src);
		src = buf.Bytes();
	}

	title := commentText(src);
	servePage(c, title, "", src);
}


func serveParseErrors(c *http.Conn, errors *parseErrors) {
	// format errors
	var buf bytes.Buffer;
	if err := parseerrorHTML.Execute(errors, &buf); err != nil {
		log.Stderrf("parseerrorHTML.Execute: %s", err)
	}
	servePage(c, "Parse errors in source file "+errors.filename, "", buf.Bytes());
}


func serveGoSource(c *http.Conn, r *http.Request, path string, styler printer.Styler) {
	prog, errors := parse(path, parser.ParseComments);
	if errors != nil {
		serveParseErrors(c, errors);
		return;
	}

	var buf bytes.Buffer;
	fmt.Fprintln(&buf, "<pre>");
	writeNode(&buf, prog, true, styler);
	fmt.Fprintln(&buf, "</pre>");

	servePage(c, "Source file "+r.URL.Path, "", buf.Bytes());
}


func redirect(c *http.Conn, r *http.Request) (redirected bool) {
	if canonical := pathutil.Clean(r.URL.Path) + "/"; r.URL.Path != canonical {
		http.Redirect(c, canonical, http.StatusMovedPermanently);
		redirected = true;
	}
	return;
}


// TODO(gri): Should have a mapping from extension to handler, eventually.

// textExt[x] is true if the extension x indicates a text file, and false otherwise.
var textExt = map[string]bool{
	".css": false,	// must be served raw
	".js": false,	// must be served raw
}


func isTextFile(path string) bool {
	// if the extension is known, use it for decision making
	if isText, found := textExt[pathutil.Ext(path)]; found {
		return isText
	}

	// the extension is not known; read an initial chunk of
	// file and check if it looks like correct UTF-8; if it
	// does, it's probably a text file
	f, err := os.Open(path, os.O_RDONLY, 0);
	if err != nil {
		return false
	}

	var buf [1024]byte;
	n, err := f.Read(&buf);
	if err != nil {
		return false
	}

	s := string(buf[0:n]);
	n -= utf8.UTFMax;	// make sure there's enough bytes for a complete unicode char
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
	return true;
}


func serveTextFile(c *http.Conn, r *http.Request, path string) {
	src, err := io.ReadFile(path);
	if err != nil {
		log.Stderrf("serveTextFile: %s", err)
	}

	var buf bytes.Buffer;
	fmt.Fprintln(&buf, "<pre>");
	template.HTMLEscape(&buf, src);
	fmt.Fprintln(&buf, "</pre>");

	servePage(c, "Text file "+path, "", buf.Bytes());
}


func serveDirectory(c *http.Conn, r *http.Request, path string) {
	if redirect(c, r) {
		return
	}

	list, err := io.ReadDir(path);
	if err != nil {
		http.NotFound(c, r);
		return;
	}

	var buf bytes.Buffer;
	if err := dirlistHTML.Execute(list, &buf); err != nil {
		log.Stderrf("dirlistHTML.Execute: %s", err)
	}

	servePage(c, "Directory "+path, "", buf.Bytes());
}


var fileServer = http.FileServer(".", "")

func serveFile(c *http.Conn, r *http.Request) {
	path := pathutil.Join(".", r.URL.Path);

	// pick off special cases and hand the rest to the standard file server
	switch ext := pathutil.Ext(path); {
	case r.URL.Path == "/":
		serveHTMLDoc(c, r, "doc/root.html");
		return;

	case r.URL.Path == "/doc/root.html":
		// hide landing page from its real name
		http.NotFound(c, r);
		return;

	case ext == ".html":
		serveHTMLDoc(c, r, path);
		return;

	case ext == ".go":
		serveGoSource(c, r, path, &Styler{linetags: true, highlight: r.FormValue("h")});
		return;
	}

	dir, err := os.Lstat(path);
	if err != nil {
		http.NotFound(c, r);
		return;
	}

	if dir != nil && dir.IsDirectory() {
		serveDirectory(c, r, path);
		return;
	}

	if isTextFile(path) {
		serveTextFile(c, r, path);
		return;
	}

	fileServer.ServeHTTP(c, r);
}


// ----------------------------------------------------------------------------
// Packages

// Package name used for commands that have non-identifier names.
const fakePkgName = "documentation"


type PageInfo struct {
	PDoc	*doc.PackageDoc;	// nil if no package found
	Dirs	*DirList;		// nil if no directory information found
	IsPkg	bool;			// false if this is not documenting a real package
}


type httpHandler struct {
	pattern	string;	// url pattern; e.g. "/pkg/"
	fsRoot	string;	// file system root to which the pattern is mapped
	isPkg	bool;	// true if this handler serves real package documentation (as opposed to command documentation)
}


// getPageInfo returns the PageInfo for a given package directory.
// If there is no corresponding package in the directory,
// PageInfo.PDoc is nil. If there are no subdirectories,
// PageInfo.Dirs is nil.
//
func (h *httpHandler) getPageInfo(path string) PageInfo {
	// the path is relative to h.fsroot
	dirname := pathutil.Join(h.fsRoot, path);

	// the package name is the directory name within its parent
	// (use dirname instead of path because dirname is clean; i.e. has no trailing '/')
	_, pkgname := pathutil.Split(dirname);

	// filter function to select the desired .go files
	filter := func(d *os.Dir) bool {
		if isPkgFile(d) {
			// Some directories contain main packages: Only accept
			// files that belong to the expected package so that
			// parser.ParsePackage doesn't return "multiple packages
			// found" errors.
			// Additionally, accept the special package name
			// fakePkgName if we are looking at cmd documentation.
			name := pkgName(dirname + "/" + d.Name);
			return name == pkgname || h.fsRoot == *cmdroot && name == fakePkgName;
		}
		return false;
	};

	// get package AST
	pkg, err := parser.ParsePackage(dirname, filter, parser.ParseComments);
	if err != nil {
		// TODO: parse errors should be shown instead of an empty directory
		log.Stderrf("parser.parsePackage: %s", err)
	}

	// compute package documentation
	var pdoc *doc.PackageDoc;
	if pkg != nil {
		ast.PackageExports(pkg);
		pdoc = doc.NewPackageDoc(pkg, pathutil.Clean(path));	// no trailing '/' in importpath
	}

	// get directory information
	var dir *Directory;
	if tree, _ := fsTree.get(); tree != nil {
		// directory tree is present; lookup respective directory
		// (may still fail if the file system was updated and the
		// new directory tree has not yet beet computed)
		dir = tree.(*Directory).lookup(dirname)
	} else {
		// no directory tree present (either early after startup
		// or command-line mode); compute one level for this page
		dir = newDirectory(dirname, 1)
	}

	return PageInfo{pdoc, dir.listing(true), h.isPkg};
}


func (h *httpHandler) ServeHTTP(c *http.Conn, r *http.Request) {
	if redirect(c, r) {
		return
	}

	path := r.URL.Path;
	path = path[len(h.pattern):len(path)];
	info := h.getPageInfo(path);

	var buf bytes.Buffer;
	if r.FormValue("f") == "text" {
		if err := packageText.Execute(info, &buf); err != nil {
			log.Stderrf("packageText.Execute: %s", err)
		}
		serveText(c, buf.Bytes());
		return;
	}

	if err := packageHTML.Execute(info, &buf); err != nil {
		log.Stderrf("packageHTML.Execute: %s", err)
	}

	if path == "" {
		path = "."	// don't display an empty path
	}
	title := "Directory " + path;
	if info.PDoc != nil {
		switch {
		case h.isPkg:
			title = "Package " + info.PDoc.PackageName
		case info.PDoc.PackageName == fakePkgName:
			// assume that the directory name is the command name
			_, pkgname := pathutil.Split(pathutil.Clean(path));
			title = "Command " + pkgname;
		default:
			title = "Command " + info.PDoc.PackageName
		}
	}

	servePage(c, title, "", buf.Bytes());
}


// ----------------------------------------------------------------------------
// Search

var searchIndex RWValue

type SearchResult struct {
	Query		string;
	Hit		*LookupResult;
	Alt		*AltWords;
	Illegal		bool;
	Accurate	bool;
}

func search(c *http.Conn, r *http.Request) {
	query := r.FormValue("q");
	var result SearchResult;

	if index, timestamp := searchIndex.get(); index != nil {
		result.Query = query;
		result.Hit, result.Alt, result.Illegal = index.(*Index).Lookup(query);
		_, ts := fsTree.get();
		result.Accurate = timestamp >= ts;
	}

	var buf bytes.Buffer;
	if err := searchHTML.Execute(result, &buf); err != nil {
		log.Stderrf("searchHTML.Execute: %s", err)
	}

	var title string;
	if result.Hit != nil {
		title = fmt.Sprintf(`Results for query %q`, query)
	} else {
		title = fmt.Sprintf(`No results found for query %q`, query)
	}

	servePage(c, title, query, buf.Bytes());
}


// ----------------------------------------------------------------------------
// Server

var (
	cmdHandler	= httpHandler{"/cmd/", *cmdroot, false};
	pkgHandler	= httpHandler{"/pkg/", *pkgroot, true};
)


func registerPublicHandlers(mux *http.ServeMux) {
	mux.Handle(cmdHandler.pattern, &cmdHandler);
	mux.Handle(pkgHandler.pattern, &pkgHandler);
	mux.Handle("/search", http.HandlerFunc(search));
	mux.Handle("/", http.HandlerFunc(serveFile));
}


// Indexing goroutine.
func indexer() {
	for {
		_, ts := fsTree.get();
		if _, timestamp := searchIndex.get(); timestamp < ts {
			// index possibly out of date - make a new one
			// (could use a channel to send an explicit signal
			// from the sync goroutine, but this solution is
			// more decoupled, trivial, and works well enough)
			start := time.Nanoseconds();
			index := NewIndex(".");
			stop := time.Nanoseconds();
			searchIndex.set(index);
			if *verbose {
				secs := float64((stop-start)/1e6) / 1e3;
				nwords, nspots := index.Size();
				log.Stderrf("index updated (%gs, %d unique words, %d spots)", secs, nwords, nspots);
			}
		}
		time.Sleep(1 * 60e9);	// try once a minute
	}
}
