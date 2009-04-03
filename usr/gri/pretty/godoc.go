// Copyright 2009 The Go Authors.  All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

// godoc: Go Documentation Server

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
	"os";
	"parser";
	"sort";
	"tabwriter";
	"template";
	"time";
	"token";
	"regexp";
	"vector";

	"astprinter";
	"compilation";  // TODO removing this causes link errors - why?
	"docprinter";
)


// TODO
// - uniform use of path, filename, dirname, pakname, etc.
// - fix weirdness with double-/'s in paths
// - cleanup uses of *root, GOROOT, etc. (quite a mess at the moment)


const (
	docPrefix = "/doc/";
	filePrefix = "/file/";
)


func getenv(varname string) string {
	value, err := os.Getenv(varname);
	return value;
}


var (
	GOROOT = getenv("GOROOT");

	// server control
	verbose = flag.Bool("v", false, "verbose mode");
	port = flag.String("port", "6060", "server port");
	root = flag.String("root", GOROOT, "root directory");

	// layout control
	tabwidth = flag.Int("tabwidth", 4, "tab width");
	usetabs = flag.Bool("usetabs", false, "align with tabs instead of blanks");
)


// ----------------------------------------------------------------------------
// Support

func cleanPath(s string) string {
	for i := 0; i < len(s); i++ {
		if s[i] == '/' {
			i++;
			j := i;
			for j < len(s) && s[j] == '/' {
				j++;
			}
			if j > i {  // more then one '/'
				return s[0 : i] + cleanPath(s[j : len(s)]);
			}
		}
	}
	return s;
}


// Reduce sequences of multiple '/'s into a single '/' and
// strip any trailing '/' (may result in the empty string).
func sanitizePath(s string) string {
	s = cleanPath(s);
	if len(s) > 0 && s[len(s)-1] == '/' {  // strip trailing '/'
		s = s[0 : len(s)-1];
	}
	return s;
}


func hasPrefix(s, prefix string) bool {
	return len(prefix) <= len(s) && s[0 : len(prefix)] == prefix;
}


func hasSuffix(s, postfix string) bool {
	pos := len(s) - len(postfix);
	return pos >= 0 && s[pos : len(s)] == postfix;
}


func isGoFile(dir *os.Dir) bool {
	return dir.IsRegular() && hasSuffix(dir.Name, ".go");
}


func isHTMLFile(dir *os.Dir) bool {
	return dir.IsRegular() && hasSuffix(dir.Name, ".html");
}


func printLink(c *http.Conn, path, name string) {
	fmt.Fprintf(c, "<a href=\"%s\">%s</a><br />\n", filePrefix + path + name, name);
}


func makeTabwriter(writer io.Write) *tabwriter.Writer {
	padchar := byte(' ');
	if *usetabs {
		padchar = '\t';
	}
	return tabwriter.NewWriter(writer, *tabwidth, 1, padchar, tabwriter.FilterHTML);
}


// ----------------------------------------------------------------------------
// Compilation

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


// Compiles a file (path) and returns the corresponding AST and
// a sorted list (by file position) of errors, if any.
//
func compile(path string, mode uint) (*ast.Program, errorList) {
	src, err := os.Open(path, os.O_RDONLY, 0);
	defer src.Close();
	if err != nil {
		log.Stdoutf("%s: %v", path, err);
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
// TODO initialize only if needed (i.e. if run as a server)
var godoc_html = template.NewTemplateOrDie("godoc.html");

func servePage(c *http.Conn, title string, contents func()) {
	c.SetHeader("content-type", "text/html; charset=utf-8");
	
	// TODO handle Apply errors
	godoc_html.Apply(c, "<!--", template.Substitution {
		"TITLE-->" : func() { fmt.Fprint(c, title); },
		"HEADER-->" : func() { fmt.Fprint(c, title); },
		"TIMESTAMP-->" : func() { fmt.Fprint(c, time.UTC().String()); },
		"CONTENTS-->" : contents
	});
}


func serveError(c *http.Conn, err, arg string) {
	servePage(c, "Error", func () {
		fmt.Fprintf(c, "%v (%s)\n", err, arg);
	});
}


// ----------------------------------------------------------------------------
// Directories

type dirArray []os.Dir
func (p dirArray) Len() int            { return len(p); }
func (p dirArray) Less(i, j int) bool  { return p[i].Name < p[j].Name; }
func (p dirArray) Swap(i, j int)       { p[i], p[j] = p[j], p[i]; }


func serveDir(c *http.Conn, dirname string) {
	fd, err1 := os.Open(*root + dirname, os.O_RDONLY, 0);
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
	servePage(c, dirname + " - Contents", func () {
		fmt.Fprintln(c, "<h2>Directories</h2>");
		for i, entry := range list {
			if entry.IsDirectory() {
				printLink(c, path, entry.Name);
			}
		}

		fmt.Fprintln(c, "<h2>Go files</h2>");
		for i, entry := range list {
			if isGoFile(&entry) {
				printLink(c, path, entry.Name);
			}
		}

		fmt.Fprintln(c, "<h2>Other files</h2>");
		for i, entry := range list {
			if !entry.IsDirectory() && !isGoFile(&entry) {
				fmt.Fprintf(c, "%s<br />\n", entry.Name);
			}
		}
	});
}


// ----------------------------------------------------------------------------
// Files

func serveCompilationErrors(c *http.Conn, filename string, errors errorList) {
	// open file
	path := *root + filename;
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

	// TODO handle Apply errors
	servePage(c, filename, func () {
		// section title
		fmt.Fprintf(c, "<h1>Compilation errors in %s</h1>\n", filename);
		
		// handle read errors
		if err1 != nil || err2 != nil /* 6g bug139 */ {
			fmt.Fprintf(c, "could not read file %s\n", filename);
			return;
		}
		
		// write source with error messages interspersed
		fmt.Fprintln(c, "<pre>");
		offs := 0;
		for i, e := range errors {
			if 0 <= e.pos.Offset && e.pos.Offset <= len(src) {
				// TODO handle Write errors
				c.Write(src[offs : e.pos.Offset]);
				// TODO this should be done using a .css file
				fmt.Fprintf(c, "<b><font color=red>%s >>></font></b>", e.msg);
				offs = e.pos.Offset;
			} else {
				log.Stdoutf("error position %d out of bounds (len = %d)", e.pos.Offset, len(src));
			}
		}
		// TODO handle Write errors
		c.Write(src[offs : len(src)]);
		fmt.Fprintln(c, "</pre>");
	});
}


func serveGoSource(c *http.Conn, dirname string, filename string) {
	path := dirname + "/" + filename;
	prog, errors := compile(*root + "/" + path, parser.ParseComments);
	if len(errors) > 0 {
		serveCompilationErrors(c, filename, errors);
		return;
	}

	servePage(c, path + " - Go source", func () {
		fmt.Fprintln(c, "<pre>");
		var p astPrinter.Printer;
		writer := makeTabwriter(c);  // for nicely formatted output
		p.Init(writer, nil, nil, true);
		p.DoProgram(prog);
		writer.Flush();  // ignore errors
		fmt.Fprintln(c, "</pre>");
	});
}


func serveHTMLFile(c *http.Conn, filename string) {
	src, err1 := os.Open(filename, os.O_RDONLY, 0);
	defer src.Close();
	if err1 != nil {
		serveError(c, err1.String(), filename);
		return
	}
	written, err2 := io.Copy(src, c);
	if err2 != nil {
		serveError(c, err2.String(), filename);
		return
	}
}


func serveFile(c *http.Conn, path string) {
	dir, err := os.Stat(*root + path);
	if err != nil {
		serveError(c, err.String(), path);
		return;
	}

	switch {
	case dir.IsDirectory():
		serveDir(c, path);
	case isGoFile(dir):
		serveGoSource(c, "", path);
	case isHTMLFile(dir):
		serveHTMLFile(c, *root + path);
	default:
		serveError(c, "Not a directory or .go file", path);
	}
}


// ----------------------------------------------------------------------------
// Packages

type pakDesc struct {
	dirname string;  // local to *root
	pakname string;  // local to directory
	filenames map[string] bool;  // set of file (names) belonging to this package
}


type pakArray []*pakDesc
func (p pakArray) Len() int            { return len(p); }
func (p pakArray) Less(i, j int) bool  { return p[i].pakname < p[j].pakname; }
func (p pakArray) Swap(i, j int)       { p[i], p[j] = p[j], p[i]; }


// The global list of packages (sorted)
// TODO should be accessed under a lock
var pakList pakArray;


func addFile(pmap map[string]*pakDesc, dirname string, filename string) {
	if hasSuffix(filename, "_test.go") {
		// ignore package tests
		return;
	}
	// determine package name
	path := *root + "/" + dirname + "/" + filename;
	prog, errors := compile(path, parser.PackageClauseOnly);
	if prog == nil {
		return;
	}
	if prog.Name.Value == "main" {
		// ignore main packages for now
		return;
	}
	pakname := dirname + "/" + prog.Name.Value;

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
	// TODO should properly check device and inode to see if we have
	//      traversed this directory already
	fd, err1 := os.Open(*root + dirname, os.O_RDONLY, 0);
	if err1 != nil {
		log.Stdoutf("%s: %v", *root + dirname, err1);
		return;
	}

	list, err2 := fd.Readdir(-1);
	if err2 != nil {
		log.Stdoutf("%s: %v", *root + dirname, err2);
		return;
	}

	for i, entry := range list {
		switch {
		case entry.IsDirectory():
			if entry.Name != "." && entry.Name != ".." {
				addDirectory(pmap, dirname + "/" + entry.Name);
			}
		case isGoFile(&entry):	
			//fmt.Printf("found %s/%s\n", dirname, entry.Name);
			addFile(pmap, dirname, entry.Name);
		}
	}
}


func makePackageMap() {
	// TODO shold do this under a lock
	// populate package map
	pmap := make(map[string]*pakDesc);
	addDirectory(pmap, "");
	
	// build sorted package list
	plist := make(pakArray, len(pmap));
	i := 0;
	for tmp, pakdesc := range pmap {
		plist[i] = pakdesc;
		i++;
	}
	sort.Sort(plist);

	// install package list (TODO should do this under a lock)
	pakList = plist;

	if *verbose {
		log.Stdoutf("%d packages found under %s", i, *root);
	}
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
		path := *root + "/" + p.dirname + "/" + filename;
		prog, errors := compile(path, parser.ParseComments);
		if len(errors) > 0 {
			serveCompilationErrors(c, filename, errors);
			return;
		}

		if i == 0 {
			// first package - initialize docPrinter
			doc.Init(prog.Name.Value);
		}
		doc.AddProgram(prog);
	}

	servePage(c, doc.PackageName() + " - Go package documentation", func () {
		writer := makeTabwriter(c);  // for nicely formatted output
		doc.Print(writer);
		writer.Flush();  // ignore errors
	});
}


func servePackageList(c *http.Conn, list *vector.Vector) {
	servePage(c, "Packages", func () {
		for i := 0; i < list.Len(); i++ {
			p := list.At(i).(*pakDesc);
			link := p.dirname + "/" + p.pakname;
			fmt.Fprintf(c, "<a href=\"%s\">%s</a> <font color=grey>(%s)</font><br />\n", docPrefix + link, p.pakname, link);
		}
	});
}


func serveDoc(c *http.Conn, path string) {
	// make regexp for package matching
	rex, err := regexp.Compile(path);
	if err != nil {
		serveError(c, err.String(), path);
		return;
	}

	// build list of matching packages
	list := vector.New(0);
	for i, p := range pakList {
		if rex.Match(p.dirname + "/" + p.pakname) {
			list.Push(p);
		}
	}

	switch list.Len() {
	case 0:
		serveError(c, "No packages found", path);
	case 1:
		servePackage(c, list.At(0).(*pakDesc));
	default:
		servePackageList(c, list);
	}
}


// ----------------------------------------------------------------------------
// Server

func makeFixedFileServer(filename string) (func(c *http.Conn, path string)) {
	return func(c *http.Conn, path string) {
		// ignore path and always serve the same file
		// TODO this should be serveFile but there are some issues with *root
		serveHTMLFile(c, filename);
	};
}


func installHandler(prefix string, handler func(c *http.Conn, path string)) {
	// create a handler customized with prefix
	f := func(c *http.Conn, req *http.Request) {
		path := req.Url.Path;
		if *verbose {
			log.Stdoutf("%s\t%s", req.Host, path);
		}
		if hasPrefix(path, prefix) {
			path = sanitizePath(path[len(prefix) : len(path)]);
			//log.Stdoutf("sanitized path %s", path);
			handler(c, path);
		} else {
			log.Stdoutf("illegal path %s", path);
		}
	};

	// install the customized handler
	http.Handle(prefix, http.HandlerFunc(f));
}


func main() {
	flag.Parse();

	*root = sanitizePath(*root);
	{	dir, err := os.Stat(*root);
		if err != nil || !dir.IsDirectory() {
			log.Exitf("root not found or not a directory: %s", *root);
		}
	}

	if *verbose {
		log.Stdoutf("Go Documentation Server\n");
		log.Stdoutf("port = %s\n", *port);
		log.Stdoutf("root = %s\n", *root);
	}

	makePackageMap();
	
	installHandler("/mem", makeFixedFileServer(GOROOT + "/doc/go_mem.html"));
	installHandler("/spec", makeFixedFileServer(GOROOT + "/doc/go_spec.html"));
	installHandler(docPrefix, serveDoc);
	installHandler(filePrefix, serveFile);

	{	err := http.ListenAndServe(":" + *port, nil);
		if err != nil {
			log.Exitf("ListenAndServe: %v", err)
		}
	}
}
