// Copyright 2009 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

package main

import (
	"bytes";
	"container/vector";
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
	"sort";
	"strings";
	"sync";
	"template";
	"time";
)


const Pkg = "/pkg/"	// name for auto-generated package documentation tree


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
		v = max;
	}
	dt.value = v;
	dt.mutex.Unlock();
}


var (
	verbose	= flag.Bool("v", false, "verbose mode");

	// file system roots
	goroot		string;
	pkgroot		= flag.String("pkgroot", "src/pkg", "root package source directory (if unrooted, relative to goroot)");
	tmplroot	= flag.String("tmplroot", "lib/godoc", "root template directory (if unrooted, relative to goroot)");

	// layout control
	tabwidth	= flag.Int("tabwidth", 4, "tab width");
)


func init() {
	goroot = os.Getenv("GOROOT");
	if goroot == "" {
		goroot = "/home/r/go-release/go";
	}
	flag.StringVar(&goroot, "goroot", goroot, "Go root directory");
}


// ----------------------------------------------------------------------------
// Predicates and small utility functions

func isGoFile(dir *os.Dir) bool {
	return dir.IsRegular() &&
		!strings.HasPrefix(dir.Name, ".") &&	// ignore .files
		pathutil.Ext(dir.Name) == ".go";
}


func isPkgFile(dir *os.Dir) bool {
	return isGoFile(dir) &&
		!strings.HasSuffix(dir.Name, "_test.go");	// ignore test files
}


func isPkgDir(dir *os.Dir) bool {
	return dir.IsDirectory() && len(dir.Name) > 0 && dir.Name[0] != '_';
}


func htmlEscape(s string) string {
	var buf bytes.Buffer;
	template.HtmlEscape(&buf, strings.Bytes(s));
	return buf.String();
}


// ----------------------------------------------------------------------------
// Directory trees

type Directory struct {
	Path string;  // including Name
	Name string;
	Subdirs []*Directory
}


func newDirTree0(path, name string) *Directory {
	list, _ := io.ReadDir(path);  // ignore errors
	// determine number of subdirectories n
	n := 0;
	for _, d := range list {
		if isPkgDir(d) {
			n++;
		}
	}
	// create Directory node
	var subdirs []*Directory;
	if n > 0 {
		subdirs = make([]*Directory, n);
		i := 0;
		for _, d := range list {
			if isPkgDir(d) {
				subdirs[i] = newDirTree0(pathutil.Join(path, d.Name), d.Name);
				i++;
			}
		}
	}
	if strings.HasPrefix(path, "src/") {
		path = path[len("src/") : len(path)];
	}
	return &Directory{path, name, subdirs};
}


func newDirTree(root string) *Directory {
	d, err := os.Lstat(root);
	if err != nil {
		log.Stderrf("%v", err);
		return nil;
	}
	if !isPkgDir(d) {
		log.Stderrf("not a package directory: %s", d.Name);
		return nil;
	}
	return newDirTree0(root, d.Name);
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
					errs[i].src = src[offs : r.Pos.Offset];
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
	highlight string;
}


func (s *Styler) LineTag(line int) (text []byte, tag printer.HtmlTag) {
	tag = printer.HtmlTag{fmt.Sprintf(`<a id="L%d">`, line), "</a>"};
	return;
}


func (s *Styler) Comment(c *ast.Comment, line []byte) (text []byte, tag printer.HtmlTag) {
	text = line;
	// minimal syntax-coloring of comments for now - people will want more
	// (don't do anything more until there's a button to turn it on/off)
	tag = printer.HtmlTag{`<span class="comment">`, "</span>"};
	return;
}


func (s *Styler) BasicLit(x *ast.BasicLit) (text []byte, tag printer.HtmlTag) {
	text = x.Value;
	return;
}


func (s *Styler) Ident(id *ast.Ident) (text []byte, tag printer.HtmlTag) {
	text = strings.Bytes(id.Value);
	if s.highlight == id.Value {
		tag = printer.HtmlTag{"<span class=highlight>", "</span>"};
	}
	return;
}


func (s *Styler) Token(tok token.Token) (text []byte, tag printer.HtmlTag) {
	text = strings.Bytes(tok.String());
	return;
}


// ----------------------------------------------------------------------------
// Templates

// Write an AST-node to w; optionally html-escaped.
func writeNode(w io.Writer, node interface{}, html bool, styler printer.Styler) {
	mode := printer.UseSpaces;
	if html {
		mode |= printer.GenHTML;
	}
	(&printer.Config{mode, *tabwidth, styler}).Fprint(w, node);
}


// Write text to w; optionally html-escaped.
func writeText(w io.Writer, text []byte, html bool) {
	if html {
		template.HtmlEscape(w, text);
		return;
	}
	w.Write(text);
}


// Write anything to w; optionally html-escaped.
func writeAny(w io.Writer, x interface{}, html bool) {
	switch v := x.(type) {
	case []byte:
		writeText(w, v, html);
	case string:
		writeText(w, strings.Bytes(v), html);
	case ast.Decl:
		writeNode(w, v, html, nil);
	case ast.Expr:
		writeNode(w, v, html, nil);
	default:
		if html {
			var buf bytes.Buffer;
			fmt.Fprint(&buf, x);
			writeText(w, buf.Bytes(), true);
		} else {
			fmt.Fprint(w, x);
		}
	}
}


// Template formatter for "html" format.
func htmlFmt(w io.Writer, x interface{}, format string) {
	writeAny(w, x, true);
}


// Template formatter for "html-comment" format.
func htmlCommentFmt(w io.Writer, x interface{}, format string) {
	var buf bytes.Buffer;
	writeAny(&buf, x, false);
	doc.ToHtml(w, buf.Bytes());  // does html-escaping
}


// Template formatter for "" (default) format.
func textFmt(w io.Writer, x interface{}, format string) {
	writeAny(w, x, false);
}


// Template formatter for "dir" format.
func dirFmt(w io.Writer, x interface{}, format string) {
	_ = x.(*Directory);  // die quickly if x has the wrong type
	if err := dirsHtml.Execute(x, w); err != nil {
		log.Stderrf("dirsHtml.Execute: %s", err);
	}
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
			fmt.Fprintf(w, "/%s#L%d", htmlEscape(pos.Filename), pos.Line);
		}
	}
}


// The strings in infoClasses must be properly html-escaped.
var infoClasses = [nKinds]string{
	"package",	// PackageClause
	"import",	// ImportDecl
	"const",	// ConstDecl
	"type",	// TypeDecl
	"var",	// VarDecl
	"func",	// FuncDecl
	"method",	// MethodDecl
	"use",	// Use
}


// Template formatter for "infoClass" format.
func infoClassFmt(w io.Writer, x interface{}, format string) {
	fmt.Fprintf(w, infoClasses[x.(SpotInfo).Kind()]);  // no html escaping needed
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


var fmap = template.FormatterMap{
	"": textFmt,
	"html": htmlFmt,
	"html-comment": htmlCommentFmt,
	"dir": dirFmt,
	"link": linkFmt,
	"infoClass": infoClassFmt,
	"infoLine": infoLineFmt,
	"infoSnippet": infoSnippetFmt,
}


func readTemplate(name string) *template.Template {
	path := pathutil.Join(*tmplroot, name);
	data, err := io.ReadFile(path);
	if err != nil {
		log.Exitf("ReadFile %s: %v", path, err);
	}
	t, err := template.Parse(string(data), fmap);
	if err != nil {
		log.Exitf("%s: %v", name, err);
	}
	return t;
}


var (
	dirsHtml,
	godocHtml,
	packageHtml,
	packageText,
	parseerrorHtml,
	parseerrorText,
	searchHtml *template.Template;
)

func readTemplates() {
	// have to delay until after flags processing,
	// so that main has chdir'ed to goroot.
	dirsHtml = readTemplate("dirs.html");
	godocHtml = readTemplate("godoc.html");
	packageHtml = readTemplate("package.html");
	packageText = readTemplate("package.txt");
	parseerrorHtml = readTemplate("parseerror.html");
	parseerrorText = readTemplate("parseerror.txt");
	searchHtml = readTemplate("search.html");
}


// ----------------------------------------------------------------------------
// Generic HTML wrapper

var pkgTree RWValue;  // *Directory tree of packages, updated with each sync

func servePage(c *http.Conn, title, query string, content []byte) {
	type Data struct {
		Title		string;
		Timestamp	string;
		Query		string;
		Content		[]byte;
	}

	_, ts := pkgTree.get();
	d := Data{
		Title: title,
		Timestamp: time.SecondsToLocalTime(ts).String(),
		Query: query,
		Content: content,
	};

	if err := godocHtml.Execute(&d, c); err != nil {
		log.Stderrf("godocHtml.Execute: %s", err);
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
		text = string(bytes.TrimSpace(src[i+len(tagBegin) : j]));
	}
	return;
}


func serveHtmlDoc(c *http.Conn, r *http.Request, filename string) {
	// get HTML body contents
	path := pathutil.Join(goroot, filename);
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
	if err := parseerrorHtml.Execute(errors, &buf); err != nil {
		log.Stderrf("parseerrorHtml.Execute: %s", err);
	}
	servePage(c, "Parse errors in source file " + errors.filename, "", buf.Bytes());
}


func serveGoSource(c *http.Conn, filename string, styler printer.Styler) {
	path := pathutil.Join(goroot, filename);
	prog, errors := parse(path, parser.ParseComments);
	if errors != nil {
		serveParseErrors(c, errors);
		return;
	}

	var buf bytes.Buffer;
	fmt.Fprintln(&buf, "<pre>");
	writeNode(&buf, prog, true, styler);
	fmt.Fprintln(&buf, "</pre>");

	servePage(c, "Source file " + filename, "", buf.Bytes());
}


var fileServer = http.FileServer(".", "")

func serveFile(c *http.Conn, r *http.Request) {
	path := r.Url.Path;

	// pick off special cases and hand the rest to the standard file server
	switch ext := pathutil.Ext(path); {
	case path == "/":
		serveHtmlDoc(c, r, "doc/root.html");

	case r.Url.Path == "/doc/root.html":
		// hide landing page from its real name
		http.NotFound(c, r);

	case ext == ".html":
		serveHtmlDoc(c, r, path);

	case ext == ".go":
		serveGoSource(c, path, &Styler{highlight: r.FormValue("h")});

	default:
		// TODO:
		// - need to decide what to serve and what not to serve
		// - don't want to download files, want to see them
		fileServer.ServeHTTP(c, r);
	}
}


// ----------------------------------------------------------------------------
// Packages

// TODO if we don't plan to use the directory information, simplify to []string
type dirList []*os.Dir

func (d dirList) Len() int {
	return len(d);
}
func (d dirList) Less(i, j int) bool {
	return d[i].Name < d[j].Name;
}
func (d dirList) Swap(i, j int) {
	d[i], d[j] = d[j], d[i];
}


func pkgName(filename string) string {
	file, err := parse(filename, parser.PackageClauseOnly);
	if err != nil || file == nil {
		return "";
	}
	return file.Name.Value;
}


type PageInfo struct {
	PDoc	*doc.PackageDoc;	// nil if no package found
	Dirs	dirList;		// nil if no subdirectories found
}


// getPageInfo returns the PageInfo for a given package directory.
// If there is no corresponding package in the directory,
// PageInfo.PDoc is nil. If there are no subdirectories,
// PageInfo.Dirs is nil.
//
func getPageInfo(path string) PageInfo {
	// the path is relative to *pkgroot
	dirname := pathutil.Join(*pkgroot, path);

	// the package name is the directory name within its parent
	_, pkgname := pathutil.Split(dirname);

	// filter function to select the desired .go files and
	// collect subdirectories
	var subdirlist vector.Vector;
	subdirlist.Init(0);
	filter := func(d *os.Dir) bool {
		if isPkgFile(d) {
			// Some directories contain main packages: Only accept
			// files that belong to the expected package so that
			// parser.ParsePackage doesn't return "multiple packages
			// found" errors.
			return pkgName(dirname + "/" + d.Name) == pkgname;
		}
		if isPkgDir(d) {
			subdirlist.Push(d);
		}
		return false;
	};

	// get package AST
	pkg, err := parser.ParsePackage(dirname, filter, parser.ParseComments);
	if err != nil {
		// TODO: parse errors should be shown instead of an empty directory
		log.Stderr(err);
	}

	// convert and sort subdirectory list, if any
	var subdirs dirList;
	if subdirlist.Len() > 0 {
		subdirs = make(dirList, subdirlist.Len());
		for i := 0; i < subdirlist.Len(); i++ {
			subdirs[i] = subdirlist.At(i).(*os.Dir);
		}
		sort.Sort(subdirs);
	}

	// compute package documentation
	var pdoc *doc.PackageDoc;
	if pkg != nil {
		ast.PackageExports(pkg);
		pdoc = doc.NewPackageDoc(pkg, pathutil.Clean(path));	// no trailing '/' in importpath
	}

	return PageInfo{pdoc, subdirs};
}


func servePkg(c *http.Conn, r *http.Request) {
	path := r.Url.Path;
	path = path[len(Pkg):len(path)];

	// canonicalize URL path and redirect if necessary
	if canonical := pathutil.Clean(Pkg+path) + "/"; r.Url.Path != canonical {
		http.Redirect(c, canonical, http.StatusMovedPermanently);
		return;
	}

	info := getPageInfo(path);

	var buf bytes.Buffer;
	if r.FormValue("f") == "text" {
		if err := packageText.Execute(info, &buf); err != nil {
			log.Stderrf("packageText.Execute: %s", err);
		}
		serveText(c, buf.Bytes());
		return;
	}

	if err := packageHtml.Execute(info, &buf); err != nil {
		log.Stderrf("packageHtml.Execute: %s", err);
	}

	if path == "" {
		path = ".";	// don't display an empty path
	}
	title := "Directory " + path;
	if info.PDoc != nil {
		title = "Package " + info.PDoc.PackageName;
	}

	servePage(c, title, "", buf.Bytes());
}


// ----------------------------------------------------------------------------
// Directory tree

// TODO(gri): Temporary - integrate with package serving.

func serveTree(c *http.Conn, r *http.Request) {
	dir, _ := pkgTree.get();

	var buf bytes.Buffer;
	dirFmt(&buf, dir, "");

	servePage(c, "Package tree", "", buf.Bytes());
}


// ----------------------------------------------------------------------------
// Search

var searchIndex RWValue

type SearchResult struct {
	Query		string;
	Hit		*LookupResult;
	Alt		*AltWords;
	Accurate	bool;
	Legend		[]string;
}

func search(c *http.Conn, r *http.Request) {
	query := r.FormValue("q");
	var result SearchResult;

	if index, timestamp := searchIndex.get(); index != nil {
		result.Query = query;
		result.Hit, result.Alt = index.(*Index).Lookup(query);
		_, ts := pkgTree.get();
		result.Accurate = timestamp >= ts;
		result.Legend = &infoClasses;
	}

	var buf bytes.Buffer;
	if err := searchHtml.Execute(result, &buf); err != nil {
		log.Stderrf("searchHtml.Execute: %s", err);
	}

	var title string;
	if result.Hit != nil {
		title = fmt.Sprintf(`Results for query %q`, query);
	} else {
		title = fmt.Sprintf(`No results found for query %q`, query);
	}

	servePage(c, title, query, buf.Bytes());
}


// ----------------------------------------------------------------------------
// Server

func registerPublicHandlers(mux *http.ServeMux) {
	mux.Handle(Pkg, http.HandlerFunc(servePkg));
	mux.Handle("/tree", http.HandlerFunc(serveTree));  // TODO(gri): integrate with package serving
	mux.Handle("/search", http.HandlerFunc(search));
	mux.Handle("/", http.HandlerFunc(serveFile));
}


// Indexing goroutine.
func indexer() {
	for {
		_, ts := pkgTree.get();
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
				secs := float64((stop-start)/1e6)/1e3;
				nwords, nspots := index.Size();
				log.Stderrf("index updated (%gs, %d unique words, %d spots)", secs, nwords, nspots);
			}
		}
		time.Sleep(1*60e9);	// try once a minute
	}
}

