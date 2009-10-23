// Copyright 2009 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

// godoc: Go Documentation Server

// Web server tree:
//
//	http://godoc/		main landing page
//	http://godoc/doc/	serve from $GOROOT/doc - spec, mem, tutorial, etc.
//	http://godoc/src/	serve files from $GOROOT/src; .go gets pretty-printed
//	http://godoc/cmd/	serve documentation about commands (TODO)
//	http://godoc/pkg/	serve documentation about packages
//				(idea is if you say import "compress/zlib", you go to
//				http://godoc/pkg/compress/zlib)
//
// Command-line interface:
//
//	godoc packagepath [name ...]
//
//	godoc compress/zlib
//		- prints doc for package compress/zlib
//	godoc crypto/block Cipher NewCMAC
//		- prints doc for Cipher and NewCMAC in package crypto/block


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
	pathutil	"path";
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
	mutex	sync.RWMutex;
	value	interface{};
	timestamp	int64;  // time of last set(), in seconds since epoch
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

	// periodic sync
	syncCmd				= flag.String("sync", "", "sync command; disabled if empty");
	syncMin				= flag.Int("sync_minutes", 0, "sync interval in minutes; disabled if <= 0");
	syncDelay	delayTime;	// actual sync delay in minutes; usually syncDelay == syncMin, but delay may back off exponentially
	syncTime	RWValue;	// time of last p4 sync

	// layout control
	tabwidth	= flag.Int("tabwidth", 4, "tab width");
	html		= flag.Bool("html", false, "print HTML in command-line mode");

	// server control
	httpaddr	= flag.String("http", "", "HTTP service address (e.g., ':6060')");
)


func init() {
	goroot = os.Getenv("GOROOT");
	if goroot == "" {
		goroot = "/home/r/go-release/go";
	}
	flag.StringVar(&goroot, "goroot", goroot, "Go root directory");
	syncTime.set(nil);	// have a reasonable initial value (time is shown on web page)
}


// ----------------------------------------------------------------------------
// Support

func isGoFile(dir *os.Dir) bool {
	return dir.IsRegular() &&
		!strings.HasPrefix(dir.Name, ".") &&	// ignore .files
		pathutil.Ext(dir.Name) == ".go" &&
		!strings.HasSuffix(dir.Name, "_test.go");	// ignore test files
}


func isPkgDir(dir *os.Dir) bool {
	return dir.IsDirectory() && len(dir.Name) > 0 && dir.Name[0] != '_';
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


func (s *Styler) Comment(c *ast.Comment, line []byte)  (text []byte, tag printer.HtmlTag) {
	text = line;
	// minimal syntax-coloring of comments for now - people will want more
	// (don't do anything more until there's a button to turn it on/off)
	tag = printer.HtmlTag{`<span class="comment">`, "</span>"};
	return;
}


func (s *Styler) BasicLit(x *ast.BasicLit)  (text []byte, tag printer.HtmlTag) {
	text = x.Value;
	return;
}


func (s *Styler) Ident(id *ast.Ident)  (text []byte, tag printer.HtmlTag) {
	text = strings.Bytes(id.Value);
	if s.highlight == id.Value {
		tag = printer.HtmlTag{"<span class=highlight>", "</span>"};
	}
	return;
}


func (s *Styler) Token(tok token.Token)  (text []byte, tag printer.HtmlTag) {
	text = strings.Bytes(tok.String());
	return;
}



// ----------------------------------------------------------------------------
// Templates

// Write an AST-node to w; optionally html-escaped.
func writeNode(w io.Writer, node interface{}, html bool, style printer.Styler) {
	mode := printer.UseSpaces;
	if html {
		mode |= printer.GenHTML;
	}
	(&printer.Config{mode, *tabwidth, style}).Fprint(w, node);
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
	doc.ToHtml(w, buf.Bytes());
}


// Template formatter for "" (default) format.
func textFmt(w io.Writer, x interface{}, format string) {
	writeAny(w, x, false);
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
			fmt.Fprintf(w, "/%s#L%d", pos.Filename, pos.Line);
		}
	}
}


var fmap = template.FormatterMap{
	"": textFmt,
	"html": htmlFmt,
	"html-comment": htmlCommentFmt,
	"link": linkFmt,
}


func readTemplate(name string) *template.Template {
	path := pathutil.Join(*tmplroot, name);
	data, err := io.ReadFile(path);
	if err != nil {
		log.Exitf("ReadFile %s: %v", path, err);
	}
	t, err1 := template.Parse(string(data), fmap);
	if err1 != nil {
		log.Exitf("%s: %v", name, err);
	}
	return t;
}


var (
	godocHtml,
	packageHtml,
	packageText,
	parseerrorHtml,
	parseerrorText *template.Template;
)

func readTemplates() {
	// have to delay until after flags processing,
	// so that main has chdir'ed to goroot.
	godocHtml = readTemplate("godoc.html");
	packageHtml = readTemplate("package.html");
	packageText = readTemplate("package.txt");
	parseerrorHtml = readTemplate("parseerror.html");
	parseerrorText = readTemplate("parseerror.txt");
}


// ----------------------------------------------------------------------------
// Generic HTML wrapper

func servePage(c *http.Conn, title, content interface{}) {
	type Data struct {
		title		interface{};
		timestamp	string;
		content		interface{};
	}

	_, ts := syncTime.get();
	d := Data{
		title: title,
		timestamp: time.SecondsToLocalTime(ts).String(),
		content: content,
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
	servePage(c, title, src);
}


func serveParseErrors(c *http.Conn, errors *parseErrors) {
	// format errors
	var buf bytes.Buffer;
	if err := parseerrorHtml.Execute(errors, &buf); err != nil {
		log.Stderrf("parseerrorHtml.Execute: %s", err);
	}
	servePage(c, "Parse errors in source file " + errors.filename, buf.Bytes());
}


func serveGoSource(c *http.Conn, filename string, style printer.Styler) {
	path := pathutil.Join(goroot, filename);
	prog, errors := parse(path, parser.ParseComments);
	if errors != nil {
		serveParseErrors(c, errors);
		return;
	}

	var buf bytes.Buffer;
	fmt.Fprintln(&buf, "<pre>");
	writeNode(&buf, prog, true, style);
	fmt.Fprintln(&buf, "</pre>");

	servePage(c, "Source file " + filename, buf.Bytes());
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
		if isGoFile(d) {
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
	if false {	// TODO req.Params["format"] == "text"
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

	servePage(c, title, buf.Bytes());
}


// ----------------------------------------------------------------------------
// Server

func loggingHandler(h http.Handler) http.Handler {
	return http.HandlerFunc(func(c *http.Conn, req *http.Request) {
		log.Stderrf("%s\t%s", c.RemoteAddr, req.Url);
		h.ServeHTTP(c, req);
	});
}


func exec(c *http.Conn, args []string) bool {
	r, w, err := os.Pipe();
	if err != nil {
		log.Stderrf("os.Pipe(): %v\n", err);
		return false;
	}

	bin := args[0];
	fds := []*os.File{nil, w, w};
	if *verbose {
		log.Stderrf("executing %v", args);
	}
	pid, err := os.ForkExec(bin, args, os.Environ(), goroot, fds);
	defer r.Close();
	w.Close();
	if err != nil {
		log.Stderrf("os.ForkExec(%q): %v\n", bin, err);
		return false;
	}

	var buf bytes.Buffer;
	io.Copy(r, &buf);
	wait, err := os.Wait(pid, 0);
	if err != nil {
		os.Stderr.Write(buf.Bytes());
		log.Stderrf("os.Wait(%d, 0): %v\n", pid, err);
		return false;
	}
	if !wait.Exited() || wait.ExitStatus() != 0 {
		os.Stderr.Write(buf.Bytes());
		log.Stderrf("executing %v failed (exit status = %d)", args, wait.ExitStatus());
		return false;
	}

	if *verbose {
		os.Stderr.Write(buf.Bytes());
	}
	if c != nil {
		c.SetHeader("content-type", "text/plain; charset=utf-8");
		c.Write(buf.Bytes());
	}

	return true;
}


func dosync(c *http.Conn, r *http.Request) {
	args := []string{"/bin/sh", "-c", *syncCmd};
	if exec(c, args) {
		// sync succeeded
		syncTime.set(nil);
		syncDelay.set(*syncMin);	//  revert to regular sync schedule
	} else {
		// sync failed - back off exponentially, but try at least once a day
		syncDelay.backoff(24*60);
	}
}


func usage() {
	fmt.Fprintf(os.Stderr,
		"usage: godoc package [name ...]\n"
		"	godoc -http=:6060\n");
	flag.PrintDefaults();
	os.Exit(2);
}


func main() {
	flag.Usage = usage;
	flag.Parse();

	// Check usage first; get usage message out early.
	switch {
	case *httpaddr != "":
		if flag.NArg() != 0 {
			usage();
		}
	default:
		if flag.NArg() == 0 {
			usage();
		}
	}

	if err := os.Chdir(goroot); err != nil {
		log.Exitf("chdir %s: %v", goroot, err);
	}

	readTemplates();

	if *httpaddr != "" {
		var handler http.Handler = http.DefaultServeMux;
		if *verbose {
			log.Stderrf("Go Documentation Server\n");
			log.Stderrf("address = %s\n", *httpaddr);
			log.Stderrf("goroot = %s\n", goroot);
			log.Stderrf("pkgroot = %s\n", *pkgroot);
			log.Stderrf("tmplroot = %s\n", *tmplroot);
			handler = loggingHandler(handler);
		}

		http.Handle(Pkg, http.HandlerFunc(servePkg));
		if *syncCmd != "" {
			http.Handle("/debug/sync", http.HandlerFunc(dosync));
		}
		http.Handle("/", http.HandlerFunc(serveFile));

		// The server may have been restarted; always wait 1sec to
		// give the forking server a chance to shut down and release
		// the http port.
		time.Sleep(1e9);

		// Start sync goroutine, if enabled.
		if *syncCmd != "" && *syncMin > 0 {
			syncDelay.set(*syncMin);	// initial sync delay
			go func() {
				for {
					dosync(nil, nil);
					delay, _ := syncDelay.get();
					if *verbose {
						log.Stderrf("next sync in %dmin", delay.(int));
					}
					time.Sleep(int64(delay.(int))*60e9);
				}
			}();
		}

		// Start http server.
		if err := http.ListenAndServe(*httpaddr, handler); err != nil {
			log.Exitf("ListenAndServe %s: %v", *httpaddr, err);
		}
		return;
	}

	if *html {
		packageText = packageHtml;
		parseerrorText = parseerrorHtml;
	}

	info := getPageInfo(flag.Arg(0));

	if info.PDoc != nil && flag.NArg() > 1 {
		args := flag.Args();
		info.PDoc.Filter(args[1:len(args)]);
	}

	if err := packageText.Execute(info, os.Stdout); err != nil {
		log.Stderrf("packageText.Execute: %s", err);
	}
}
