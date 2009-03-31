// Copyright 2009 The Go Authors.  All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

// GDS: Go Documentation Server

package main

import (
	"bufio";
	"flag";
	"fmt";
	"http";
	"io";
	"net";
	"os";
	"sort";
	"log";
	"template";
	"tabwriter";

	"utils";
	"platform";
	"compilation";
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


// Support for directory sorting.
type DirArray []os.Dir
func (p DirArray) Len() int            { return len(p); }
func (p DirArray) Less(i, j int) bool  { return p[i].Name < p[j].Name; }
func (p DirArray) Swap(i, j int)       { p[i], p[j] = p[j], p[i]; }


func isGoFile(dir *os.Dir) bool {
	const ext = ".go";
	return dir.IsRegular() && Utils.Contains(dir.Name, ext, len(dir.Name) - len(ext));
}


func printLink(c *http.Conn, path, name string) {
	fmt.Fprintf(c, "<a href=\"%s\">%s</a><br />\n", path + name, name);
}


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

	sort.Sort(DirArray(list));

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


func serveFile(c *http.Conn, filename string) {
	var flags Compilation.Flags;
	prog, errors := Compilation.Compile(*root + filename, &flags);
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

	c.SetHeader("content-type", "text/html; charset=utf-8");
	
	// initialize tabwriter for nicely aligned output
	padchar := byte(' ');
	if *usetabs {
		padchar = '\t';
	}
	writer := tabwriter.NewWriter(c, *tabwidth, 1, padchar, tabwriter.FilterHTML);

	// write documentation
	var doc docPrinter.PackageDoc;
	doc.Init(string(prog.Name.Lit));
	doc.AddProgram(prog);
	doc.Print(writer);

	// flush any pending output
	err := writer.Flush();
	if err != nil {
		panic("print error - exiting");
	}
}


func serve(c *http.Conn, req *http.Request) {
	if *verbose {
		log.Stdoutf("URL = %s\n", req.RawUrl);
	}

	path := Utils.SanitizePath(req.Url.Path);
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
		serveFile(c, path);
	default:
		c.WriteHeader(http.StatusNotFound);
		fmt.Fprintf(c, "Error: Not a directory or .go file (%s)\n", path);
	}
}


func main() {
	flag.Parse();

	*root = Utils.SanitizePath(*root);
	dir, err1 := os.Stat(*root);
	if err1 != nil || !dir.IsDirectory() {
		log.Exitf("root not found or not a directory: %s", *root);
	}

	if *verbose {
		log.Stdoutf("Go Documentation Server\n");
		log.Stdoutf("port = %s\n", *port);
		log.Stdoutf("root = %s\n", *root);
	}

	http.Handle("/", http.HandlerFunc(serve));
	err2 := http.ListenAndServe(":" + *port, nil);
	if err2 != nil {
		log.Exitf("ListenAndServe: %s", err2.String())
	}
}

