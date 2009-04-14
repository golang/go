// Copyright 2009 The Go Authors.  All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

// godoc: Go Documentation Server

// Web server tree:
//
//	http://godoc/	main landing page (TODO)
//	http://godoc/doc/	serve from $GOROOT/doc - spec, mem, tutorial, etc. (TODO)
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
	"ast";
	"bufio";
	"flag";
	"fmt";
	"http";
	"io";
	"log";
	"net";
	"once";
	"os";
	"parser";
	pathutil "path";
	"sort";
	"strings";
	"tabwriter";
	"template";
	"time";
	"token";
	"vector";

	"astprinter";
	"docprinter";
)


// TODO
// - uniform use of path, filename, dirname, pakname, etc.
// - fix weirdness with double-/'s in paths
// - split http service into its own source file


const usageString =
	"usage: godoc package [name ...]\n"
	"	godoc -http=:6060\n"


const (
	docPrefix = "/doc/";
	filePrefix = "/file/";
)


var (
	goroot string;

	verbose = flag.Bool("v", false, "verbose mode");

	// server control
	httpaddr = flag.String("http", "", "HTTP service address (e.g., ':6060')");

	// layout control
	tabwidth = flag.Int("tabwidth", 4, "tab width");
	usetabs = flag.Bool("tabs", false, "align with tabs instead of spaces");
)


func init() {
	var err *os.Error;
	goroot, err = os.Getenv("GOROOT");
	if err != nil {
		goroot = "/home/r/go-build/go";
	}
	flag.StringVar(&goroot, "goroot", goroot, "Go root directory");
}


// ----------------------------------------------------------------------------
// Support

func isGoFile(dir *os.Dir) bool {
	return dir.IsRegular() && strings.HasSuffix(dir.Name, ".go");
}


func isHTMLFile(dir *os.Dir) bool {
	return dir.IsRegular() && strings.HasSuffix(dir.Name, ".html");
}


func isDir(name string) bool {
	d, err := os.Stat(name);
	return err == nil && d.IsDirectory();
}


func isFile(name string) bool {
	d, err := os.Stat(name);
	return err == nil && d.IsRegular();
}


func printLink(c io.Write, dir, name string) {
	fmt.Fprintf(c, "<a href=\"%s\">%s</a><br />\n", pathutil.Clean(filePrefix + dir + "/" + name), name);
}


func makeTabwriter(writer io.Write) *tabwriter.Writer {
	padchar := byte(' ');
	if *usetabs {
		padchar = '\t';
	}
	return tabwriter.NewWriter(writer, *tabwidth, 1, padchar, tabwriter.FilterHTML);
}


// ----------------------------------------------------------------------------
// Parsing

type parseError struct {
	pos token.Position;
	msg string;
}


type errorList []parseError
func (list errorList) Len() int { return len(list); }
func (list errorList) Less(i, j int) bool { return list[i].pos.Offset < list[j].pos.Offset; }
func (list errorList) Swap(i, j int) { list[i], list[j] = list[j], list[i]; }


type errorHandler struct {
	lastLine int;
	errors *vector.Vector;
}


func (h *errorHandler) Error(pos token.Position, msg string) {
	// only collect errors that are on a new line
	// in the hope to avoid most follow-up errors
	if pos.Line != h.lastLine {
		h.lastLine = pos.Line;
		if h.errors == nil {
			// lazy initialize - most of the time there are no errors
			h.errors = vector.New(0);
		}
		h.errors.Push(parseError{pos, msg});
	}
}


// Parses a file (path) and returns the corresponding AST and
// a sorted list (by file position) of errors, if any.
//
func parse(path string, mode uint) (*ast.Program, errorList) {
	src, err := os.Open(path, os.O_RDONLY, 0);
	defer src.Close();
	if err != nil {
		log.Stderrf("open %s: %v", path, err);
		var noPos token.Position;
		return nil, errorList{parseError{noPos, err.String()}};
	}

	var handler errorHandler;
	prog, ok := parser.Parse(src, &handler, mode);
	if !ok {
		// convert error list and sort it
		errors := make(errorList, handler.errors.Len());
		for i := 0; i < handler.errors.Len(); i++ {
			errors[i] = handler.errors.At(i).(parseError);
		}
		sort.Sort(errors);
		return nil, errors;
	}

	return prog, nil;
}


// ----------------------------------------------------------------------------
// Templates

// html template
var godoc_html string

func readTemplate() {
	name := "usr/gri/pretty/godoc.html";
	f, err := os.Open(name, os.O_RDONLY, 0);
	if err != nil {
		log.Exitf("open %s: %v", name, err);
	}
	var b io.ByteBuffer;
	if n, err := io.Copy(f, &b); err != nil {
		log.Exitf("copy %s: %v", name, err);
	}
	f.Close();
	godoc_html = string(b.Data());
}


func servePage(c *http.Conn, title, content interface{}) {
	once.Do(readTemplate);

	c.SetHeader("content-type", "text/html; charset=utf-8");

	type Data struct {
		title string;
		header string;
		timestamp string;
		content string;
	}
	
	// TODO(rsc): Once template system can handle []byte,
	// remove this conversion.
	if x, ok := title.([]byte); ok {
		title = string(x);
	}
	if x, ok := content.([]byte); ok {
		content = string(x);
	}

	var d Data;
	d.title = title.(string);
	d.header = title.(string);
	d.timestamp = time.UTC().String();
	d.content = content.(string);
	templ, err, line := template.Parse(godoc_html, nil);
	if err != nil {
		log.Stderrf("template error %s:%d: %s\n", title, line, err);
	} else {
		templ.Execute(&d, c);
	}
}


func serveError(c *http.Conn, err, arg string) {
	servePage(c, "Error", fmt.Sprintf("%v (%s)\n", err, arg));
}


// ----------------------------------------------------------------------------
// Directories

type dirArray []os.Dir
func (p dirArray) Len() int            { return len(p); }
func (p dirArray) Less(i, j int) bool  { return p[i].Name < p[j].Name; }
func (p dirArray) Swap(i, j int)       { p[i], p[j] = p[j], p[i]; }


func serveDir(c *http.Conn, dirname string) {
	fd, err1 := os.Open(dirname, os.O_RDONLY, 0);
	if err1 != nil {
		c.WriteHeader(http.StatusNotFound);
		fmt.Fprintf(c, "Error: %v (%s)\n", err1, dirname);
		return;
	}

	list, err2 := fd.Readdir(-1);
	if err2 != nil {
		c.WriteHeader(http.StatusNotFound);
		fmt.Fprintf(c, "Error: %v (%s)\n", err2, dirname);
		return;
	}

	sort.Sort(dirArray(list));

	path := dirname + "/";

	// Print contents in 3 sections: directories, go files, everything else
	var b io.ByteBuffer;
	fmt.Fprintln(&b, "<h2>Directories</h2>");
	for i, entry := range list {
		if entry.IsDirectory() {
			printLink(&b, path, entry.Name);
		}
	}

	fmt.Fprintln(&b, "<h2>Go files</h2>");
	for i, entry := range list {
		if isGoFile(&entry) {
			printLink(&b, path, entry.Name);
		}
	}

	fmt.Fprintln(&b, "<h2>Other files</h2>");
	for i, entry := range list {
		if !entry.IsDirectory() && !isGoFile(&entry) {
			fmt.Fprintf(&b, "%s<br />\n", entry.Name);
		}
	}

	servePage(c, dirname + " - Contents", b.Data());
}


// ----------------------------------------------------------------------------
// Files

func serveParseErrors(c *http.Conn, filename string, errors errorList) {
	// open file
	path := filename;
	fd, err1 := os.Open(path, os.O_RDONLY, 0);
	defer fd.Close();
	if err1 != nil {
		serveError(c, err1.String(), path);
		return;
	}

	// read source
	var buf io.ByteBuffer;
	n, err2 := io.Copy(fd, &buf);
	if err2 != nil {
		serveError(c, err2.String(), path);
		return;
	}
	src := buf.Data();

	// generate body
	var b io.ByteBuffer;
	// section title
	fmt.Fprintf(&b, "<h1>Parse errors in %s</h1>\n", filename);

	// handle read errors
	if err1 != nil || err2 != nil {
		fmt.Fprintf(&b, "could not read file %s\n", filename);
		return;
	}

	// write source with error messages interspersed
	fmt.Fprintln(&b, "<pre>");
	offs := 0;
	for i, e := range errors {
		if 0 <= e.pos.Offset && e.pos.Offset <= len(src) {
			// TODO handle Write errors
			b.Write(src[offs : e.pos.Offset]);
			// TODO this should be done using a .css file
			fmt.Fprintf(&b, "<b><font color=red>%s >>></font></b>", e.msg);
			offs = e.pos.Offset;
		} else {
			log.Stderrf("error position %d out of bounds (len = %d)", e.pos.Offset, len(src));
		}
	}
	// TODO handle Write errors
	b.Write(src[offs : len(src)]);
	fmt.Fprintln(&b, "</pre>");

	servePage(c, filename, b.Data());
}


func serveGoSource(c *http.Conn, dirname string, filename string) {
	path := dirname + "/" + filename;
	prog, errors := parse(path, parser.ParseComments);
	if len(errors) > 0 {
		serveParseErrors(c, filename, errors);
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

	servePage(c, path + " - Go source", b.Data());
}


func serveHTMLFile(c *http.Conn, filename string) {
	src, err1 := os.Open(filename, os.O_RDONLY, 0);
	defer src.Close();
	if err1 != nil {
		serveError(c, err1.String(), filename);
		return
	}
	if written, err2 := io.Copy(src, c); err2 != nil {
		serveError(c, err2.String(), filename);
		return
	}
}


func serveFile(c *http.Conn, path string) {
	dir, err := os.Stat(path);
	if err != nil {
		serveError(c, err.String(), path);
		return;
	}

	switch {
	case dir.IsDirectory():
		serveDir(c, path);
	case isGoFile(dir):
		serveGoSource(c, ".", path);
	case isHTMLFile(dir):
		serveHTMLFile(c, path);
	default:
		serveError(c, "Not a directory or .go file", path);
	}
}


// ----------------------------------------------------------------------------
// Packages

type pakDesc struct {
	dirname string;  // relative to goroot
	pakname string;  // relative to directory
	filenames map[string] bool;  // set of file (names) belonging to this package
}


type pakArray []*pakDesc
func (p pakArray) Len() int            { return len(p); }
func (p pakArray) Less(i, j int) bool  { return p[i].pakname < p[j].pakname; }
func (p pakArray) Swap(i, j int)       { p[i], p[j] = p[j], p[i]; }


func addFile(pmap map[string]*pakDesc, dirname string, filename string) {
	if strings.HasSuffix(filename, "_test.go") {
		// ignore package tests
		return;
	}
	// determine package name
	path := dirname + "/" + filename;
	prog, errors := parse(path, parser.PackageClauseOnly);
	if prog == nil {
		return;
	}
	if prog.Name.Value == "main" {
		// ignore main packages for now
		return;
	}
	pakname := pathutil.Clean(dirname + "/" + prog.Name.Value);

	// find package descriptor
	pakdesc, found := pmap[pakname];
	if !found {
		// add a new descriptor
		pakdesc = &pakDesc{dirname, prog.Name.Value, make(map[string]bool)};
		pmap[pakname] = pakdesc;
	}

	//fmt.Printf("pak = %s, file = %s\n", pakname, filename);

	// add file to package desc
	if tmp, found := pakdesc.filenames[filename]; found {
		panic("internal error: same file added more then once: " + filename);
	}
	pakdesc.filenames[filename] = true;
}


func addDirectory(pmap map[string]*pakDesc, dirname string) {
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

	for i, entry := range list {
		switch {
		case isGoFile(&entry):
			//fmt.Printf("found %s/%s\n", dirname, entry.Name);
			addFile(pmap, dirname, entry.Name);
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


func servePackage(c *http.Conn, p *pakDesc) {
	// make a filename list
	filenames := make([]string, len(p.filenames));
	i := 0;
	for filename, tmp := range p.filenames {
		filenames[i] = filename;
		i++;
	}

	// compute documentation
	var doc docPrinter.PackageDoc;
	for i, filename := range filenames {
		path := p.dirname + "/" + filename;
		prog, errors := parse(path, parser.ParseComments);
		if len(errors) > 0 {
			serveParseErrors(c, filename, errors);
			return;
		}

		if i == 0 {
			// first package - initialize docPrinter
			doc.Init(prog.Name.Value);
		}
		doc.AddProgram(prog);
	}

	var b io.ByteBuffer;
	writer := makeTabwriter(&b);  // for nicely formatted output
	doc.Print(writer);
	writer.Flush();	// ignore errors

	servePage(c, doc.PackageName() + " - Go package documentation", b.Data());
}


func servePackageList(c *http.Conn, list pakArray) {
	var b io.ByteBuffer;
	for i := 0; i < len(list); i++ {
		p := list[i];
		link := pathutil.Clean(p.dirname + "/" + p.pakname);
		fmt.Fprintf(&b, "<a href=\"%s\">%s</a> <font color=grey>(%s)</font><br />\n",
			p.pakname, p.pakname, link);
	}

	servePage(c, "Packages", b.Data());

	// TODO: show subdirectories
}


// Return package or packages named by name.
// Name is either an import string or a directory,
// like you'd see in $GOROOT/pkg/ once the 6g
// tools can handle a hierarchy there.
//
// Examples:
//	"math"	- single package made up of directory
//	"container"	- directory listing
//	"container/vector"	- single package in container directory
func findPackages(name string) (*pakDesc, pakArray) {
	// Build list of packages.
	// If the path names a directory, scan that directory
	// for a package with the name matching the directory name.
	// Otherwise assume it is a package name inside
	// a directory, so scan the parent.
	pmap := make(map[string]*pakDesc);
	dir := pathutil.Clean("src/lib/" + name);
	if isDir(dir) {
		parent, pak := pathutil.Split(dir);
		addDirectory(pmap, dir);
		paks := mapValues(pmap);
		if len(paks) == 1 {
			p := paks[0];
			if p.dirname == dir && p.pakname == pak {
				return p, nil;
			}
		}
		return nil, paks;
	}

	// Otherwise, have parentdir/pak.  Look for package pak in dir.
	parentdir, pak := pathutil.Split(dir);
	addDirectory(pmap, parentdir);
	if p, ok := pmap[dir]; ok {
		return p, nil;
	}

	return nil, nil;
}


func servePkg(c *http.Conn, path string) {
	pak, paks := findPackages(path);

	// TODO: canonicalize path and redirect if needed.

	switch {
	case pak != nil:
		servePackage(c, pak);
	case len(paks) > 0:
		servePackageList(c, paks);
	default:
		serveError(c, "No packages found", path);
	}
}


// ----------------------------------------------------------------------------
// Server

func makeFixedFileServer(filename string) (func(c *http.Conn, path string)) {
	return func(c *http.Conn, path string) {
		serveFile(c, filename);
	};
}


func installHandler(prefix string, handler func(c *http.Conn, path string)) {
	// create a handler customized with prefix
	f := func(c *http.Conn, req *http.Request) {
		path := req.Url.Path;
		if *verbose {
			log.Stderrf("%s\t%s", req.Host, path);
		}
		handler(c, path[len(prefix) : len(path)]);
	};

	// install the customized handler
	http.Handle(prefix, http.HandlerFunc(f));
}


func usage() {
	fmt.Fprintf(os.Stderr, usageString);
	sys.Exit(1);
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

	if *httpaddr != "" {
		if *verbose {
			log.Stderrf("Go Documentation Server\n");
			log.Stderrf("address = %s\n", *httpaddr);
			log.Stderrf("goroot = %s\n", goroot);
		}

		installHandler("/mem", makeFixedFileServer("doc/go_mem.html"));
		installHandler("/spec", makeFixedFileServer("doc/go_spec.html"));
		installHandler("/pkg/", servePkg);
		installHandler(filePrefix, serveFile);

		if err := http.ListenAndServe(*httpaddr, nil); err != nil {
			log.Exitf("ListenAndServe %s: %v", *httpaddr, err)
		}
		return;
	}

	log.Exitf("godoc command-line not implemented");
}
