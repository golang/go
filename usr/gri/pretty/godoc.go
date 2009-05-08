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
	return dir.IsRegular() && strings.HasSuffix(dir.Name, ".go");
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
	var b io.ByteBuffer;
	if n, err := io.Copy(f, &b); err != nil {
		return nil, err;
	}
	return b.Data(), nil;
}


// ----------------------------------------------------------------------------
// Parsing

type rawError struct {
	pos token.Position;
	msg string;
}


type rawErrorVector struct {
	vector.Vector;
}


func (v *rawErrorVector) At(i int) rawError { return v.Vector.At(i).(rawError) }
func (v *rawErrorVector) Less(i, j int) bool { return v.At(i).pos.Offset < v.At(j).pos.Offset; }


func (v *rawErrorVector) Error(pos token.Position, msg string) {
	// only collect errors that are on a new line
	// in the hope to avoid most follow-up errors
	lastLine := 0;
	if n := v.Len(); n > 0 {
		lastLine = v.At(n - 1).pos.Line;
	}
	if lastLine != pos.Line {
		v.Push(rawError{pos, msg});
	}
}


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
func parse(filename string, mode uint) (*ast.Program, *parseErrors) {
	src, err := ReadFile(filename);
	if err != nil {
		log.Stderrf("ReadFile %s: %v", filename, err);
		errs := []parseError{parseError{nil, 0, err.String()}};
		return nil, &parseErrors{filename, errs, nil};
	}

	var raw rawErrorVector;
	prog, ok := parser.Parse(src, &raw, mode);
	if !ok {
		// sort and convert error list
		sort.Sort(&raw);
		errs := make([]parseError, raw.Len() + 1);	// +1 for final fragment of source
		offs := 0;
		for i := 0; i < raw.Len(); i++ {
			r := raw.At(i);
			// Should always be true, but check for robustness.
			if 0 <= r.pos.Offset && r.pos.Offset <= len(src) {
				errs[i].src = src[offs : r.pos.Offset];
				offs = r.pos.Offset;
			}
			errs[i].line = r.pos.Line;
			errs[i].msg = r.msg;
		}
		errs[raw.Len()].src = src[offs : len(src)];
		return nil, &parseErrors{filename, errs, src};
	}

	return prog, nil;
}


// ----------------------------------------------------------------------------
// Templates

// Return text for decl.
func DeclText(d ast.Decl) []byte {
	var b io.ByteBuffer;
	var p astPrinter.Printer;
	p.Init(&b, nil, nil, false);
	d.Visit(&p);
	return b.Data();
}


// Return text for expr.
func ExprText(d ast.Expr) []byte {
	var b io.ByteBuffer;
	var p astPrinter.Printer;
	p.Init(&b, nil, nil, false);
	d.Visit(&p);
	return b.Data();
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
	var b io.ByteBuffer;
	fmt.Fprint(&b, x);
	return b.Data();
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


// Template formatter for "dir/" format.
// Writes out "/" if the os.Dir argument is a directory.
var slash = io.StringBytes("/");

func dirSlashFmt(w io.Writer, x interface{}, format string) {
	d := x.(os.Dir);	// TODO(rsc): want *os.Dir
	if d.IsDirectory() {
		w.Write(slash);
	}
}


var fmap = template.FormatterMap{
	"": textFmt,
	"html": htmlFmt,
	"html-comment": htmlCommentFmt,
	"dir/": dirSlashFmt,
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
var packagelistHtml *template.Template;
var packagelistText *template.Template;
var parseerrorHtml *template.Template;
var parseerrorText *template.Template;

func readTemplates() {
	// have to delay until after flags processing,
	// so that main has chdir'ed to goroot.
	godocHtml = readTemplate("godoc.html");
	packageHtml = readTemplate("package.html");
	packageText = readTemplate("package.txt");
	packagelistHtml = readTemplate("packagelist.html");
	packagelistText = readTemplate("packagelist.txt");
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


func serveError(c *http.Conn, err, arg string) {
	servePage(c, "Error", fmt.Sprintf("%v (%s)\n", err, arg));
}


// ----------------------------------------------------------------------------
// Files

func serveParseErrors(c *http.Conn, errors *parseErrors) {
	// format errors
	var b io.ByteBuffer;
	parseerrorHtml.Execute(errors, &b);
	servePage(c, errors.filename + " - Parse Errors", b.Data());
}


func serveGoSource(c *http.Conn, name string) {
	prog, errors := parse(name, parser.ParseComments);
	if errors != nil {
		serveParseErrors(c, errors);
		return;
	}

	var b io.ByteBuffer;
	fmt.Fprintln(&b, "<pre>");
	var p astPrinter.Printer;
	writer := makeTabwriter(&b);  // for nicely formatted output
	p.Init(writer, nil, nil, true);
	p.DoProgram(prog);
	writer.Flush();  // ignore errors
	fmt.Fprintln(&b, "</pre>");

	servePage(c, name + " - Go source", b.Data());
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
	pakname string;  // relative to directory
	importpath string;	// import "___"
	filenames map[string] bool;  // set of file (names) belonging to this package
}


type pakArray []*pakDesc
func (p pakArray) Len() int            { return len(p); }
func (p pakArray) Less(i, j int) bool  { return p[i].pakname < p[j].pakname; }
func (p pakArray) Swap(i, j int)       { p[i], p[j] = p[j], p[i]; }


func addFile(pmap map[string]*pakDesc, dirname, filename, importprefix string) {
	if strings.HasSuffix(filename, "_test.go") {
		// ignore package tests
		return;
	}
	// determine package name
	path := pathutil.Join(dirname, filename);
	prog, errors := parse(path, parser.PackageClauseOnly);
	if prog == nil {
		return;
	}
	if prog.Name.Value == "main" {
		// ignore main packages for now
		return;
	}

	var importpath string;
	dir, name := pathutil.Split(importprefix);
	if name == prog.Name.Value {	// package math in directory "math"
		importpath = importprefix;
	} else {
		importpath = pathutil.Clean(importprefix + "/" + prog.Name.Value);
	}

	// find package descriptor
	pakdesc, found := pmap[importpath];
	if !found {
		// add a new descriptor
		pakdesc = &pakDesc{dirname, prog.Name.Value, importpath, make(map[string]bool)};
		pmap[importpath] = pakdesc;
	}

	//fmt.Printf("pak = %s, file = %s\n", pakname, filename);

	// add file to package desc
	if tmp, found := pakdesc.filenames[filename]; found {
		panic("internal error: same file added more then once: " + filename);
	}
	pakdesc.filenames[filename] = true;
}


func addDirectory(pmap map[string]*pakDesc, dirname, importprefix string, subdirs *[]os.Dir) {
	path := dirname;
	fd, err1 := os.Open(path, os.O_RDONLY, 0);
	if err1 != nil {
		log.Stderrf("open %s: %v", path, err1);
		return;
	}

	list, err2 := fd.Readdir(-1);
	if err2 != nil {
		log.Stderrf("readdir %s: %v", path, err2);
		return;
	}

	nsub := 0;
	for i, entry := range list {
		switch {
		case isGoFile(&entry):
			addFile(pmap, dirname, entry.Name, importprefix);
		case entry.IsDirectory():
			nsub++;
		}
	}

	if subdirs != nil && nsub > 0 {
		*subdirs = make([]os.Dir, nsub);
		nsub = 0;
		for i, entry := range list {
			if entry.IsDirectory() {
				subdirs[nsub] = entry;
				nsub++;
			}
		}
	}
}


func mapValues(pmap map[string]*pakDesc) pakArray {
	// build sorted package list
	plist := make(pakArray, len(pmap));
	i := 0;
	for tmp, pakdesc := range pmap {
		plist[i] = pakdesc;
		i++;
	}
	sort.Sort(plist);
	return plist;
}


func (p *pakDesc) Doc() (*doc.PackageDoc, *parseErrors) {
	// compute documentation
	var r doc.DocReader;
	i := 0;
	for filename := range p.filenames {
		path := p.dirname + "/" + filename;
		prog, err := parse(path, parser.ParseComments);
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


func servePackage(c *http.Conn, p *pakDesc) {
	doc, errors := p.Doc();
	if errors != nil {
		serveParseErrors(c, errors);
		return;
	}

	var b io.ByteBuffer;
	if false {	// TODO req.Params["format"] == "text"
		err := packageText.Execute(doc, &b);
		if err != nil {
			log.Stderrf("packageText.Execute: %s", err);
		}
		serveText(c, b.Data());
		return;
	}
	err := packageHtml.Execute(doc, &b);
	if err != nil {
		log.Stderrf("packageHtml.Execute: %s", err);
	}
	servePage(c, doc.ImportPath + " - Go package documentation", b.Data());
}


type pakInfo struct {
	Path string;
	Package *pakDesc;
	Packages pakArray;
	Subdirs []os.Dir;	// TODO(rsc): []*os.Dir
}


func servePackageList(c *http.Conn, info *pakInfo) {
	var b io.ByteBuffer;
	err := packagelistHtml.Execute(info, &b);
	if err != nil {
		log.Stderrf("packagelistHtml.Execute: %s", err);
	}
	servePage(c, info.Path + " - Go packages", b.Data());
}


// Return package or packages named by name.
// Name is either an import string or a directory,
// like you'd see in $GOROOT/pkg/.
//
// Examples:
//	"math"	- single package made up of directory
//	"container"	- directory listing
//	"container/vector"	- single package in container directory
//
func findPackages(name string) *pakInfo {
	info := new(pakInfo);

	// Build list of packages.
	pmap := make(map[string]*pakDesc);

	// If the path names a directory, scan that directory
	// for a package with the name matching the directory name.
	// Otherwise assume it is a package name inside
	// a directory, so scan the parent.
	cname := pathutil.Clean(name);
	if cname == "" {
		cname = "."
	}
	dir := pathutil.Join(*pkgroot, cname);

	if isDir(dir) {
		addDirectory(pmap, dir, cname, &info.Subdirs);
		paks := mapValues(pmap);
		if len(paks) == 1 {
			p := paks[0];
			_, pak := pathutil.Split(dir);
			if p.dirname == dir && p.pakname == pak {
				info.Package = p;
				info.Path = cname;
				return info;
			}
		}

		info.Packages = paks;
		if cname == "." {
			info.Path = "";
		} else {
			info.Path = cname + "/";
		}
		return info;
	}

	// Otherwise, have parentdir/pak.  Look for package pak in parentdir.
	parentdir, _ := pathutil.Split(dir);
	parentname, _ := pathutil.Split(cname);
	if parentname == "" {
		parentname = "."
	}

	addDirectory(pmap, parentdir, parentname, nil);
	if p, ok := pmap[cname]; ok {
		info.Package = p;
		info.Path = cname;
		return info;
	}

	info.Path = name;	// original, uncleaned name
	return info;
}


func servePkg(c *http.Conn, r *http.Request) {
	path := r.Url.Path;
	path = path[len(Pkg) : len(path)];
	info := findPackages(path);
	if r.Url.Path != Pkg + info.Path {
		http.Redirect(c, info.Path);
		return;
	}

	if info.Package != nil {
		servePackage(c, info.Package);
	} else {
		servePackageList(c, info);
	}
}


// ----------------------------------------------------------------------------
// Server

func LoggingHandler(h http.Handler) http.Handler {
	return http.HandlerFunc(func(c *http.Conn, req *http.Request) {
		log.Stderrf("%s\t%s", req.Host, req.Url.Path);
		h.ServeHTTP(c, req);
	})
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
			handler = LoggingHandler(handler);
		}

		http.Handle(Pkg, http.HandlerFunc(servePkg));
		http.Handle("/", http.HandlerFunc(serveFile));

		if err := http.ListenAndServe(*httpaddr, handler); err != nil {
			log.Exitf("ListenAndServe %s: %v", *httpaddr, err)
		}
		return;
	}

	if *html {
		packageText = packageHtml;
		packagelistText = packagelistHtml;
		parseerrorText = parseerrorHtml;
	}

	info := findPackages(flag.Arg(0));
	if info.Package == nil {
		err := packagelistText.Execute(info, os.Stderr);
		if err != nil {
			log.Stderrf("packagelistText.Execute: %s", err);
		}
		os.Exit(1);
	}

	doc, errors := info.Package.Doc();
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
