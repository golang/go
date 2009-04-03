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

	"compilation";  // TODO removing this causes link errors - why?
	"docprinter";
)


// TODO
// - uniform use of path, filename, dirname, pakname, etc.
// - fix weirdness with double-/'s in paths


const (
	docPrefix = "/doc/";
	srcPrefix = "/src/";
)


func getenv(varname string) string {
	value, err := os.Getenv(varname);
	return value;
}


var (
	GOROOT string;

	// server control
	verbose = flag.Bool("v", false, "verbose mode");
	port = flag.String("port", "6060", "server port");
	root = flag.String("root", getenv("GOROOT"), "go root directory");

	// layout control
	tabwidth = flag.Int("tabwidth", 4, "tab width");
	usetabs = flag.Bool("usetabs", false, "align with tabs instead of blanks");

	// html template
	godoc_template = template.NewTemplateOrDie("godoc.html");
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


func printLink(c *http.Conn, path, name string) {
	fmt.Fprintf(c, "<a href=\"%s\">%s</a><br />\n", srcPrefix + path + name, name);
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

	c.SetHeader("content-type", "text/html; charset=utf-8");
	path := dirname + "/";

	// Print contents in 3 sections: directories, go files, everything else

	// TODO handle Apply errors
	godoc_template.Apply(c, "<!--", template.Substitution {
		"TITLE-->" : func() {
			fmt.Fprint(c, dirname);
		},

		"HEADER-->" : func() {
			fmt.Fprint(c, dirname);
		},

		"TIMESTAMP-->" : func() {
			fmt.Fprint(c, time.UTC().String());
		},

		"CONTENTS-->" : func () {
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
		}
	});
}


// ----------------------------------------------------------------------------
// Files

func printErrors(c *http.Conn, filename string, errors errorList) {
	// open file
	path := *root + filename;
	fd, err1 := os.Open(path, os.O_RDONLY, 0);
	defer fd.Close();
	if err1 != nil {
		// TODO better error handling
		log.Stdoutf("%s: %v", path, err1);
	}

	// read source
	var buf io.ByteBuffer;
	n, err2 := io.Copy(fd, &buf);
	if err2 != nil {
		// TODO better error handling
		log.Stdoutf("%s: %v", path, err2);
	}
	src := buf.Data();

	// TODO handle Apply errors
	godoc_template.Apply(c, "<!--", template.Substitution {
		"TITLE-->" : func() {
			fmt.Fprint(c, filename);
		},

		"HEADER-->" : func() {
			fmt.Fprint(c, filename);
		},

		"TIMESTAMP-->" : func() {
			fmt.Fprint(c, time.UTC().String());
		},

		"CONTENTS-->" : func () {
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
		}
	});
}


func serveGoFile(c *http.Conn, dirname string, filenames []string) {
	// compute documentation
	var doc docPrinter.PackageDoc;
	for i, filename := range filenames {
		path := *root + "/" + dirname + "/" + filename;
		prog, errors := compile(path, parser.ParseComments);
		if len(errors) > 0 {
			c.SetHeader("content-type", "text/html; charset=utf-8");
			printErrors(c, filename, errors);
			return;
		}

		if i == 0 {
			// first package - initialize docPrinter
			doc.Init(prog.Name.Value);
		}
		doc.AddProgram(prog);
	}

	c.SetHeader("content-type", "text/html; charset=utf-8");
	
	godoc_template.Apply(c, "<!--", template.Substitution {
		"TITLE-->" : func() {
			fmt.Fprintf(c, "%s - Go package documentation", doc.PackageName());
		},

		"HEADER-->" : func() {
			fmt.Fprintf(c, "%s - Go package documentation", doc.PackageName());
		},

		"TIMESTAMP-->" : func() {
			fmt.Fprint(c, time.UTC().String());
		},

		"CONTENTS-->" : func () {
			// write documentation
			writer := makeTabwriter(c);  // for nicely formatted output
			doc.Print(writer);
			writer.Flush();  // ignore errors
		}
	});
}


func serveSrc(c *http.Conn, path string) {
	dir, err := os.Stat(*root + path);
	if err != nil {
		c.WriteHeader(http.StatusNotFound);
		fmt.Fprintf(c, "Error: %v (%s)\n", err, path);
		return;
	}

	switch {
	case dir.IsDirectory():
		serveDir(c, path);
	case isGoFile(dir):
		serveGoFile(c, "", []string{path});
	default:
		c.WriteHeader(http.StatusNotFound);
		fmt.Fprintf(c, "Error: Not a directory or .go file (%s)\n", path);
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


var (
	pakMap map[string]*pakDesc;  // dirname/pakname -> package descriptor
	pakList pakArray;  // sorted list of packages; in sync with pakMap
)


func addFile(dirname string, filename string) {
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
	pakdesc, found := pakMap[pakname];
	if !found {
		// add a new descriptor
		pakdesc = &pakDesc{dirname, prog.Name.Value, make(map[string]bool)};
		pakMap[pakname] = pakdesc;
	}
	
	//fmt.Printf("pak = %s, file = %s\n", pakname, filename);

	// add file to package desc
	if tmp, found := pakdesc.filenames[filename]; found {
		panic("internal error: same file added more then once: " + filename);
	}
	pakdesc.filenames[filename] = true;
}


func addDirectory(dirname string) {
	// TODO should properly check device and inode to see if we have
	//      traversed this directory already
	//fmt.Printf("traversing %s\n", dirname);

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
				addDirectory(dirname + "/" + entry.Name);
			}
		case isGoFile(&entry):	
			//fmt.Printf("found %s/%s\n", dirname, entry.Name);
			addFile(dirname, entry.Name);
		}
	}
}


func makePackageMap() {
	// TODO shold do this under a lock, eventually
	// populate package map
	pakMap = make(map[string]*pakDesc);
	addDirectory("");
	
	// build sorted package list
	pakList = make([]*pakDesc, len(pakMap));
	i := 0;
	for tmp, pakdesc := range pakMap {
		pakList[i] = pakdesc;
		i++;
	}
	sort.Sort(pakList);

	if *verbose {
		log.Stdoutf("%d packages found under %s", i, *root);
	}
}


func serveGoPackage(c *http.Conn, p *pakDesc) {
	// make a filename list
	list := make([]string, len(p.filenames));
	i := 0;
	for filename, tmp := range p.filenames {
		list[i] = filename;
		i++;
	}
	
	serveGoFile(c, p.dirname, list);
}


func servePackageList(c *http.Conn, list *vector.Vector) {
	godoc_template.Apply(c, "<!--", template.Substitution {
		"TITLE-->" : func() {
			fmt.Fprint(c, "Packages");
		},

		"HEADER-->" : func() {
			fmt.Fprint(c, "Packages");
		},

		"TIMESTAMP-->" : func() {
			fmt.Fprint(c, time.UTC().String());
		},

		"CONTENTS-->" : func () {
			// TODO should do this under a lock, eventually
			for i := 0; i < list.Len(); i++ {
				p := list.At(i).(*pakDesc);
				link := p.dirname + "/" + p.pakname;
				fmt.Fprintf(c, "<a href=\"%s\">%s</a> <font color=grey>(%s)</font><br />\n", docPrefix + link, p.pakname, link);
			}
		}
	});
}


func serveDoc(c *http.Conn, path string) {
	// make regexp for package matching
	rex, err := regexp.Compile(path);
	if err != nil {
		// TODO report this via an error page
		log.Stdoutf("failed to compile regexp: %s", path);
	}

	// build list of matching packages
	list := vector.New(0);
	for i, p := range pakList {
		if rex.Match(p.dirname + "/" + p.pakname) {
			list.Push(p);
		}
	}

	if list.Len() == 1 {
		serveGoPackage(c, list.At(0).(*pakDesc));
	} else {
		servePackageList(c, list);
	}
}


// ----------------------------------------------------------------------------
// Server

func installHandler(prefix string, handler func(c *http.Conn, path string)) {
	// customized handler with prefix
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

	installHandler(docPrefix, serveDoc);
	installHandler(srcPrefix, serveSrc);
	{	err := http.ListenAndServe(":" + *port, nil);
		if err != nil {
			log.Exitf("ListenAndServe: %v", err)
		}
	}
}

