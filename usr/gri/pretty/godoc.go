// Copyright 2009 The Go Authors.  All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

// godoc: Go Documentation Server

// Web server tree:
//
//	http://godoc/	main landing page
//	http://godoc/doc/	serve from $GOROOT/doc - spec, mem, tutorial, etc.
//	http://godoc/src/	serve files from $GOROOT/src; .go gets pretty-printed
//	http://godoc/cmd/	serve documentation about commands (TODO)
//	http://godoc/pkg/	serve documentation about packages
//		(idea is if you say import "compress/zlib", you go to
//		http://godoc/pkg/compress/zlib)
//
// Command-line interface:
//
//	godoc packagepath [name ...]
//
//	godoc compress/zlib
//		- prints doc for package proto
//	godoc compress/zlib Cipher NewCMAC
//		- prints doc for Cipher and NewCMAC in package crypto/block


package main

import (
	"container/vector";
	"flag";
	"fmt";
	"go/ast";
	"go/doc";
	"go/parser";
	"go/token";
	"http";
	"io";
	"log";
	"net";
	"os";
	pathutil "path";
	"sort";
	"strings";
	"tabwriter";
	"template";
	"time";

	"astprinter";
	"comment";
)


const Pkg = "/pkg/"	// name for auto-generated package documentation tree


var (
	verbose = flag.Bool("v", false, "verbose mode");

	// file system roots
	goroot string;
	pkgroot = flag.String("pkgroot", "src/lib", "root package source directory (if unrooted, relative to goroot)");
	tmplroot = flag.String("tmplroot", "usr/gri/pretty", "root template directory (if unrooted, relative to goroot)");

	// layout control
	tabwidth = flag.Int("tabwidth", 4, "tab width");
	usetabs = flag.Bool("tabs", false, "align with tabs instead of spaces");
	html = flag.Bool("html", false, "print HTML in command-line mode");

	// server control
	httpaddr = flag.String("http", "", "HTTP service address (e.g., ':6060')");
)


func init() {
	var err os.Error;
	goroot, err = os.Getenv("GOROOT");
	if err != nil {
		goroot = "/home/r/go-release/go";
	}
	flag.StringVar(&goroot, "goroot", goroot, "Go root directory");
}


// ----------------------------------------------------------------------------
// Support

func isGoFile(dir *os.Dir) bool {
	return dir.IsRegular() && pathutil.Ext(dir.Name) == ".go";
}


func isDir(name string) bool {
	d, err := os.Stat(name);
	return err == nil && d.IsDirectory();
}


func makeTabwriter(writer io.Writer) *tabwriter.Writer {
	padchar := byte(' ');
	if *usetabs {
		padchar = '\t';
	}
	return tabwriter.NewWriter(writer, *tabwidth, 1, padchar, tabwriter.FilterHTML);
}


// TODO(rsc): this belongs in a library somewhere, maybe os
func ReadFile(name string) ([]byte, os.Error) {
	f, err := os.Open(name, os.O_RDONLY, 0);
	if err != nil {
		return nil, err;
	}
	defer f.Close();
	var buf io.ByteBuffer;
	if n, err := io.Copy(f, &buf); err != nil {
		return nil, err;
	}
	return buf.Data(), nil;
}


// ----------------------------------------------------------------------------
// Parsing

// A single error in the parsed file.
type parseError struct {
	src []byte;	// source before error
	line int;	// line number of error
	msg string;	// error message
}


// All the errors in the parsed file, plus surrounding source code.
// Each error has a slice giving the source text preceding it
// (starting where the last error occurred).  The final element in list[]
// has msg = "", to give the remainder of the source code.
// This data structure is handed to the templates parseerror.txt and parseerror.html.
//
type parseErrors struct {
	filename string;	// path to file
	list []parseError;	// the errors
	src []byte;	// the file's entire source code
}


// Parses a file (path) and returns the corresponding AST and
// a sorted list (by file position) of errors, if any.
//
func parse(path string, mode uint) (*ast.Program, *parseErrors) {
	src, err := ReadFile(path);
	if err != nil {
		log.Stderrf("ReadFile %s: %v", path, err);
		errs := []parseError{parseError{nil, 0, err.String()}};
		return nil, &parseErrors{path, errs, nil};
	}

	prog, err := parser.Parse(src, mode);
	if err != nil {
		// sort and convert error list
		if errors, ok := err.(parser.ErrorList); ok {
			sort.Sort(errors);
			errs := make([]parseError, len(errors) + 1);	// +1 for final fragment of source
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
			errs[len(errors)].src = src[offs : len(src)];
			return nil, &parseErrors{path, errs, src};
		} else {
			// TODO should have some default handling here to be more robust
			panic("unreachable");
		}
	}

	return prog, nil;
}


// ----------------------------------------------------------------------------
// Templates

// Return text for decl.
func DeclText(d ast.Decl) []byte {
	var buf io.ByteBuffer;
	var p astPrinter.Printer;
	p.Init(&buf, nil, nil, false);
	d.Visit(&p);
	return buf.Data();
}


// Return text for expr.
func ExprText(d ast.Expr) []byte {
	var buf io.ByteBuffer;
	var p astPrinter.Printer;
	p.Init(&buf, nil, nil, false);
	d.Visit(&p);
	return buf.Data();
}


// Convert x, whatever it is, to text form.
func toText(x interface{}) []byte {
	type String interface { String() string }

	switch v := x.(type) {
	case []byte:
		return v;
	case string:
		return io.StringBytes(v);
	case String:
		return io.StringBytes(v.String());
	case ast.Decl:
		return DeclText(v);
	case ast.Expr:
		return ExprText(v);
	}
	var buf io.ByteBuffer;
	fmt.Fprint(&buf, x);
	return buf.Data();
}


// Template formatter for "html" format.
func htmlFmt(w io.Writer, x interface{}, format string) {
	// Can do better than text in some cases.
	switch v := x.(type) {
	case ast.Decl:
		var p astPrinter.Printer;
		tw := makeTabwriter(w);
		p.Init(tw, nil, nil, true);
		v.Visit(&p);
		tw.Flush();
	case ast.Expr:
		var p astPrinter.Printer;
		tw := makeTabwriter(w);
		p.Init(tw, nil, nil, true);
		v.Visit(&p);
		tw.Flush();
	default:
		template.HtmlEscape(w, toText(x));
	}
}


// Template formatter for "html-comment" format.
func htmlCommentFmt(w io.Writer, x interface{}, format string) {
	comment.ToHtml(w, toText(x));
}


// Template formatter for "" (default) format.
func textFmt(w io.Writer, x interface{}, format string) {
	w.Write(toText(x));
}


var fmap = template.FormatterMap{
	"": textFmt,
	"html": htmlFmt,
	"html-comment": htmlCommentFmt,
}


func readTemplate(name string) *template.Template {
	path := pathutil.Join(*tmplroot, name);
	data, err := ReadFile(path);
	if err != nil {
		log.Exitf("ReadFile %s: %v", path, err);
	}
	t, err1 := template.Parse(string(data), fmap);
	if err1 != nil {
		log.Exitf("%s: %v", name, err);
	}
	return t;
}


var godocHtml *template.Template
var packageHtml *template.Template
var packageText *template.Template
var dirlistHtml *template.Template;
var dirlistText *template.Template;
var parseerrorHtml *template.Template;
var parseerrorText *template.Template;

func readTemplates() {
	// have to delay until after flags processing,
	// so that main has chdir'ed to goroot.
	godocHtml = readTemplate("godoc.html");
	packageHtml = readTemplate("package.html");
	packageText = readTemplate("package.txt");
	dirlistHtml = readTemplate("dirlist.html");
	dirlistText = readTemplate("dirlist.txt");
	parseerrorHtml = readTemplate("parseerror.html");
	parseerrorText = readTemplate("parseerror.txt");
}


// ----------------------------------------------------------------------------
// Generic HTML wrapper

func servePage(c *http.Conn, title, content interface{}) {
	type Data struct {
		title interface{};
		header interface{};
		timestamp string;
		content interface{};
	}

	var d Data;
	d.title = title;
	d.header = title;
	d.timestamp = time.UTC().String();
	d.content = content;
	godocHtml.Execute(&d, c);
}


func serveText(c *http.Conn, text []byte) {
	c.SetHeader("content-type", "text/plain; charset=utf-8");
	c.Write(text);
}


// ----------------------------------------------------------------------------
// Files

func serveParseErrors(c *http.Conn, errors *parseErrors) {
	// format errors
	var buf io.ByteBuffer;
	parseerrorHtml.Execute(errors, &buf);
	servePage(c, errors.filename + " - Parse Errors", buf.Data());
}


func serveGoSource(c *http.Conn, name string) {
	prog, errors := parse(name, parser.ParseComments);
	if errors != nil {
		serveParseErrors(c, errors);
		return;
	}

	var buf io.ByteBuffer;
	fmt.Fprintln(&buf, "<pre>");
	var p astPrinter.Printer;
	writer := makeTabwriter(&buf);  // for nicely formatted output
	p.Init(writer, nil, nil, true);
	p.DoProgram(prog);
	writer.Flush();  // ignore errors
	fmt.Fprintln(&buf, "</pre>");

	servePage(c, name + " - Go source", buf.Data());
}


var fileServer = http.FileServer(".", "");

func serveFile(c *http.Conn, req *http.Request) {
	// pick off special cases and hand the rest to the standard file server
	switch {
	case req.Url.Path == "/":
		// serve landing page.
		// TODO: hide page from ordinary file serving.
		// writing doc/index.html will take care of that.
		http.ServeFile(c, req, "doc/root.html");

	case req.Url.Path == "/doc/root.html":
		// hide landing page from its real name
		// TODO why - there is no reason for this (remove eventually)
		http.NotFound(c, req);

	case pathutil.Ext(req.Url.Path) == ".go":
		serveGoSource(c, req.Url.Path[1 : len(req.Url.Path)]);  // strip leading '/' from name

	default:
		// TODO not good enough - don't want to download files
		// want to see them
		fileServer.ServeHTTP(c, req);
	}
}


// ----------------------------------------------------------------------------
// Packages

type pakDesc struct {
	dirname string;  // relative to goroot
	pakname string;  // same as last component of importpath
	importpath string;	// import "___"
	filenames map[string] bool;  // set of file (names) belonging to this package
}


// TODO if we don't plan to use the directory information, simplify to []string
type dirList []*os.Dir

func (d dirList) Len() int  { return len(d) }
func (d dirList) Less(i, j int) bool  { return d[i].Name < d[j].Name }
func (d dirList) Swap(i, j int)  { d[i], d[j] = d[j], d[i] }


func isPackageFile(dirname, filename, pakname string) bool {
	// ignore test files
	if strings.HasSuffix(filename, "_test.go") {
		return false;
	}

	// determine package name
	prog, errors := parse(dirname + "/" + filename, parser.PackageClauseOnly);
	if prog == nil {
		return false;
	}

	return prog != nil && prog.Name.Value == pakname;
}


// Returns the package denoted by path and the list of
// sub-directories in the corresponding package directory.
// If there is no such package, the first result is nil. If
// there are no sub-directories, that list is nil.
func findPackage(path string) (*pakDesc, dirList) {
	// get directory contents, if possible
	importpath := pathutil.Clean(path);  // no trailing '/'
	dirname := pathutil.Join(*pkgroot, importpath);
	if !isDir(dirname) {
		return nil, nil;
	}

	fd, err1 := os.Open(dirname, os.O_RDONLY, 0);
	if err1 != nil {
		log.Stderrf("open %s: %v", dirname, err1);
		return nil, nil;
	}

	list, err2 := fd.Readdir(-1);
	if err2 != nil {
		log.Stderrf("readdir %s: %v", dirname, err2);
		return nil, nil;
	}

	// the package name is is the directory name within its parent
	_, pakname := pathutil.Split(dirname);

	// collect all files belonging to the package and count the
	// number of sub-directories
	filenames := make(map[string]bool);
	nsub := 0;
	for i, entry := range list {
		switch {
		case isGoFile(&entry) && isPackageFile(dirname, entry.Name, pakname):
			// add file to package desc
			if tmp, found := filenames[entry.Name]; found {
				panic("internal error: same file added more than once: " + entry.Name);
			}
			filenames[entry.Name] = true;
		case entry.IsDirectory():
			nsub++;
		}
	}

	// make the list of sub-directories, if any
	var subdirs dirList;
	if nsub > 0 {
		subdirs = make(dirList, nsub);
		nsub = 0;
		for i, entry := range list {
			if entry.IsDirectory() {
				// make a copy here so sorting (and other code) doesn't
				// have to make one every time an entry is moved
				copy := new(os.Dir);
				*copy = entry;
				subdirs[nsub] = copy;
				nsub++;
			}
		}
		sort.Sort(subdirs);
	}

	// if there are no package files, then there is no package
	if len(filenames) == 0 {
		return nil, subdirs;
	}

	return &pakDesc{dirname, pakname, importpath, filenames}, subdirs;
}


func (p *pakDesc) Doc() (*doc.PackageDoc, *parseErrors) {
	// compute documentation
	var r doc.DocReader;
	i := 0;
	for filename := range p.filenames {
		prog, err := parse(p.dirname + "/" + filename, parser.ParseComments);
		if err != nil {
			return nil, err;
		}
		if i == 0 {
			// first file - initialize doc
			r.Init(prog.Name.Value, p.importpath);
		}
		i++;
		r.AddProgram(prog);
	}
	return r.Doc(), nil;
}


func servePackage(c *http.Conn, desc *pakDesc) {
	doc, errors := desc.Doc();
	if errors != nil {
		serveParseErrors(c, errors);
		return;
	}

	var buf io.ByteBuffer;
	if false {	// TODO req.Params["format"] == "text"
		err := packageText.Execute(doc, &buf);
		if err != nil {
			log.Stderrf("packageText.Execute: %s", err);
		}
		serveText(c, buf.Data());
		return;
	}
	err := packageHtml.Execute(doc, &buf);
	if err != nil {
		log.Stderrf("packageHtml.Execute: %s", err);
	}
	servePage(c, doc.ImportPath + " - Go package documentation", buf.Data());
}


func serveDirList(c *http.Conn, path string, dirs dirList) {
	var buf io.ByteBuffer;
	err := dirlistHtml.Execute(dirs, &buf);
	if err != nil {
		log.Stderrf("dirlist.Execute: %s", err);
	}
	servePage(c, path + " - Directories", buf.Data());
}


func servePkg(c *http.Conn, r *http.Request) {
	path := r.Url.Path;
	path = path[len(Pkg) : len(path)];
	desc, dirs := findPackage(path);
	/*
	// TODO do we still need this?
	if r.Url.Path != Pkg + info.Path {
		http.Redirect(c, info.Path, http.StatusMovedPermanently);
		return;
	}
	*/
	if desc != nil {
		servePackage(c, desc);
		// TODO should also serve sub-directories if there are any
	} else {
		// make sure path is not empty otherwise html links become rooted
		// and won't work correctly
		if path == "" {
			path = ".";
		}
		serveDirList(c, path, dirs);
	}
}


// ----------------------------------------------------------------------------
// Server

func loggingHandler(h http.Handler) http.Handler {
	return http.HandlerFunc(func(c *http.Conn, req *http.Request) {
		log.Stderrf("%s\t%s", req.Host, req.Url.Path);
		h.ServeHTTP(c, req);
	})
}


func restartGodoc(c *http.Conn, r *http.Request) {
	binary := os.Args[0];  // TODO currently requires absolute paths because of chdir in the beginning
	pid, err := os.ForkExec(binary, os.Args, os.Environ(), "", []*os.File{os.Stdin, os.Stdout, os.Stderr});
	if err != nil {
		log.Stderrf("os.ForkExec(%s): %v", binary, err);
		return;  // do not terminate
	}
	log.Stderrf("restarted %s, pid = %d\n", binary, pid);
	os.Exit(0);
}


func usage() {
	fmt.Fprintf(os.Stderr,
		"usage: godoc package [name ...]\n"
		"	godoc -http=:6060\n"
	);
	flag.PrintDefaults();
	os.Exit(1);
}


func main() {
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
		http.Handle("/debug/restart", http.HandlerFunc(restartGodoc));
		http.Handle("/", http.HandlerFunc(serveFile));

		// The server may have been restarted; always wait 1sec to
		// give the forking server a chance to shut down and release
		// the http port. (This is necessary because under OS X Exec
		// won't work if there are more than one thread running.)
		time.Sleep(1e9);

		if err := http.ListenAndServe(*httpaddr, handler); err != nil {
			log.Exitf("ListenAndServe %s: %v", *httpaddr, err)
		}
		return;
	}

	if *html {
		packageText = packageHtml;
		dirlistText = dirlistHtml;
		parseerrorText = parseerrorHtml;
	}

	desc, dirs := findPackage(flag.Arg(0));
	if desc == nil {
		err := dirlistText.Execute(dirs, os.Stdout);
		if err != nil {
			log.Stderrf("dirlistText.Execute: %s", err);
		}
		os.Exit(0);
	}

	doc, errors := desc.Doc();
	if errors != nil {
		err := parseerrorText.Execute(errors, os.Stderr);
		if err != nil {
			log.Stderrf("parseerrorText.Execute: %s", err);
		}
		os.Exit(1);
	}

	if flag.NArg() > 1 {
		args := flag.Args();
		doc.Filter(args[1 : len(args)]);
	}

	packageText.Execute(doc, os.Stdout);
}
