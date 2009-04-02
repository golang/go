// Copyright 2009 The Go Authors.  All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

// godoc: Go Documentation Server

package main

import (
	"bufio";
	"flag";
	"fmt";
	"http";
	"io";
	"log";
	"net";
	"os";
	"sort";
	"tabwriter";
	"template";
	"regexp";

	"ast";
	"vector";
	"utils";
	"platform";
	"compilation";
	"parser";
	"docprinter";
)


var (
	verbose = flag.Bool("v", false, "verbose mode");
	port = flag.String("port", "6060", "server port");
	root = flag.String("root", Platform.GOROOT, "go root directory");

	// layout control
	tabwidth = flag.Int("tabwidth", 4, "tab width");
	usetabs = flag.Bool("usetabs", false, "align with tabs instead of blanks");
)


// ----------------------------------------------------------------------------
// Support

type dirArray []os.Dir
func (p dirArray) Len() int            { return len(p); }
func (p dirArray) Less(i, j int) bool  { return p[i].Name < p[j].Name; }
func (p dirArray) Swap(i, j int)       { p[i], p[j] = p[j], p[i]; }


func isGoFile(dir *os.Dir) bool {
	const ext = ".go";
	return dir.IsRegular() && Utils.Contains(dir.Name, ext, len(dir.Name) - len(ext));
}


func printLink(c *http.Conn, path, name string) {
	fmt.Fprintf(c, "<a href=\"%s\">%s</a><br />\n", path + name, name);
}


func makeTabwriter(writer io.Write) *tabwriter.Writer {
	padchar := byte(' ');
	if *usetabs {
		padchar = '\t';
	}
	return tabwriter.NewWriter(writer, *tabwidth, 1, padchar, tabwriter.FilterHTML);
}


// ----------------------------------------------------------------------------
// Directories

var dir_template = template.NewTemplateOrDie("dir_template.html");

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
	dir_template.Apply(c, "<!--", template.Substitution {
		"PATH-->" : func() {
			fmt.Fprintf(c, "%s", path);
		},

		"DIRECTORIES-->" : func() {
			for i, entry := range list {
				if entry.IsDirectory() {
					printLink(c, path, entry.Name);
				}
			}
		},

		"GO FILES-->" : func() {
			for i, entry := range list {
				if isGoFile(&entry) {
					printLink(c, path, entry.Name);
				}
			}
		},

		"OTHER FILES-->" : func() {
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

var error_template = template.NewTemplateOrDie("error_template.html");

func printErrors(c *http.Conn, filename string, errors Compilation.ErrorList) {
	// TODO factor code - shouldn't do this here and in Compilation
	src, ok := Platform.ReadSourceFile(*root + filename);

	// TODO handle Apply errors
	error_template.Apply(c, "<!--", template.Substitution {
		"FILE_NAME-->" : func() {
			fmt.Fprintf(c, "%s", filename);
		},

		"ERRORS-->" : func () {
			if ok == false /* 6g bug139 */ {
				fmt.Fprintf(c, "could not read file %s\n", *root + filename);
				return;
			}
			offs := 0;
			for i, e := range errors {
				if 0 <= e.Pos.Offset && e.Pos.Offset <= len(src) {
					// TODO handle Write errors
					c.Write(src[offs : e.Pos.Offset]);
					// TODO this should be done using a .css file
					fmt.Fprintf(c, "<b><font color=red>%s >>></font></b>", e.Msg);
					offs = e.Pos.Offset;
				} else {
					log.Stdoutf("error position %d out of bounds (len = %d)", e.Pos.Offset, len(src));
				}
			}
			// TODO handle Write errors
			c.Write(src[offs : len(src)]);
		}
	});
}


func serveGoFile(c *http.Conn, dirname string, filenames []string) {
	// compute documentation
	var doc docPrinter.PackageDoc;
	for i, filename := range filenames {
		var flags Compilation.Flags;
		prog, errors := Compilation.Compile(*root + "/" + dirname + "/" + filename, &flags);
		if errors == nil {
			c.WriteHeader(http.StatusNotFound);
			fmt.Fprintf(c, "Error: could not read file (%s)\n", filename);
			return;
		}

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
	
	// write documentation
	writer := makeTabwriter(c);  // for nicely formatted output
	doc.Print(writer);
	writer.Flush();  // ignore errors
}


func serveFile(c *http.Conn, path string) {
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


func getAST(dirname string, filename string, mode uint) *ast.Program {
	// open file
	fullname := *root + "/" + dirname + "/" + filename;
	src, err := os.Open(fullname, os.O_RDONLY, 0);
	defer src.Close();
	if err != nil {
		log.Stdoutf("%s: %v", fullname, err);
		return nil;
	}

	// determine package name
	prog, ok := parser.Parse(src, nil, mode);
	if !ok {
		log.Stdoutf("%s: compilation errors", fullname);
		return nil;
	}
	
	return prog;
}


func addFile(dirname string, filename string) {
	// determine package name
	prog := getAST(dirname, filename, parser.PackageClauseOnly);
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
}


var packages_template = template.NewTemplateOrDie("packages_template.html");

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
	packages_template.Apply(c, "<!--", template.Substitution {
		"PACKAGE_LIST-->" : func() {
			// TODO should do this under a lock, eventually
			for i := 0; i < list.Len(); i++ {
				p := list.At(i).(*pakDesc);
				link := p.dirname + "/" + p.pakname;
				fmt.Fprintf(c, "<a href=\"%s\">%s</a> <font color=grey>(%s)</font><br />\n", link + "?p", p.pakname, link);
			}
		}
	});
}


func servePackage(c *http.Conn, path string) {
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

func serve(c *http.Conn, req *http.Request) {
	if *verbose {
		log.Stdoutf("%s\t%s", req.Host, req.RawUrl);
	}

	path := Utils.SanitizePath(req.Url.Path);

	if len(req.Url.Query) > 0 {  // for now any query will do
		servePackage(c, path);
	} else {
		serveFile(c, path);
	}
}


func main() {
	flag.Parse();

	*root = Utils.SanitizePath(*root);
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

	http.Handle("/", http.HandlerFunc(serve));
	{	err := http.ListenAndServe(":" + *port, nil);
		if err != nil {
			log.Exitf("ListenAndServe: %v", err)
		}
	}
}

