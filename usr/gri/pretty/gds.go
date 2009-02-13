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

	Utils "utils";
	Platform "platform";
	Compilation "compilation";
	Printer "printer";
)


var (
	verbose = flag.Bool("v", false, "verbose mode");
	port = flag.String("port", "6060", "server port");
	root = flag.String("root", Platform.GOROOT, "go root directory");
)


// Support for directory sorting.
type DirArray []os.Dir
func (p DirArray) Len() int            { return len(p); }
func (p DirArray) Less(i, j int) bool  { return p[i].Name < p[j].Name; }
func (p DirArray) Swap(i, j int)       { p[i], p[j] = p[j], p[i]; }


func isGoFile(dir *os.Dir) bool {
	ext := ".go";  // TODO 6g bug - should be const
	return dir.IsRegular() && Utils.Contains(dir.Name, ext, len(dir.Name) - len(ext));
}


func printLink(c *http.Conn, path, name string) {
	fmt.Fprintf(c, "<a href=\"%s\">%s</a><br>\n", path + name, name);
}


func serveDir(c *http.Conn, dirname string) {
	fd, err1 := os.Open(*root + dirname, os.O_RDONLY, 0);
	if err1 != nil {
		c.WriteHeader(http.StatusNotFound);
		fmt.Fprintf(c, "Error: %v (%s)\n", err1, dirname);
		return;
	}

	list, err2 := os.Readdir(fd, -1);
	if err2 != nil {
		c.WriteHeader(http.StatusNotFound);
		fmt.Fprintf(c, "Error: %v (%s)\n", err2, dirname);
		return;
	}
	
	sort.Sort(DirArray(list));

	c.SetHeader("content-type", "text/html; charset=utf-8");
	path := dirname + "/";
	fmt.Fprintf(c, "<b>%s</b>\n", path);

	// Print contents in 3 sections: directories, go files, everything else
	
	// 1) directories
	fmt.Fprintln(c, "<p>");
	for i, entry := range list {
		if entry.IsDirectory() {
			printLink(c, path, entry.Name);
		}
	}

	// 2) .go files
	fmt.Fprintln(c, "<p>");
	for i, entry := range list {
		if isGoFile(&entry) {
			printLink(c, path, entry.Name);
		}
	}

	// 3) everything else
	fmt.Fprintln(c, "<p>");
	for i, entry := range list {
		if !entry.IsDirectory() && !isGoFile(&entry) {
			fmt.Fprintf(c, "<font color=grey>%s</font><br>\n", entry.Name);
		}
	}
}


func serveFile(c *http.Conn, filename string) {
	var flags Compilation.Flags;
	prog, nerrors := Compilation.Compile(*root + filename, &flags);
	if nerrors > 0 {
		c.WriteHeader(http.StatusNotFound);
		fmt.Fprintf(c, "Error: File has compilation errors (%s)\n", filename);
		return;
	}
	
	c.SetHeader("content-type", "text/html; charset=utf-8");
	Printer.Print(c, true, prog);
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
		log.Exitf("root not found or not a directory: ", *root);
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

