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

	Utils "utils";
	Platform "platform";
	Compilation "compilation";
	Printer "printer";
)


var urlPrefix = "/gds"  // 6g BUG should be const


var (
	verbose = flag.Bool("v", false, "verbose mode");
	port = flag.String("port", "6060", "server port");
	//root = flag.String("root", Platform.GOROOT, "go root directory");
	root = &Platform.GOROOT;  // TODO cannot change root w/ passing it to printer
)


// TODO should factor this out - also used by the parser
func getFilename(url string) string {
	// strip URL prefix
	if url[0 : len(urlPrefix)] != urlPrefix {
		panic("server error - illegal URL prefix");
	}
	url = url[len(urlPrefix) : len(url)];
	
	// sanitize source file name
	return *root + Utils.TrimExt(url, ".go") + ".go";
}


func docServer(c *http.Conn, req *http.Request) {
	if *verbose {
		fmt.Printf("URL path = %s\n", req.Url.Path);
	}

	filename := getFilename(req.Url.Path);
	var flags Compilation.Flags;
	prog, nerrors := Compilation.Compile(filename, &flags);
	if nerrors > 0 {
		c.WriteHeader(http.StatusNotFound);
		fmt.Fprintf(c, "compilation errors: %s\n", filename);
		return;
	}
	
	c.SetHeader("content-type", "text/html; charset=utf-8");
	Printer.Print(c, true, prog);
}


func main() {
	flag.Parse();

	if *verbose {
		fmt.Printf("Go Documentation Server\n");
		fmt.Printf("port = %s\n", *port);
		fmt.Printf("root = %s\n", *root);
	}

	http.Handle(urlPrefix + "/", http.HandlerFunc(docServer));
	err := http.ListenAndServe(":" + *port, nil);
	if err != nil {
		panic("ListenAndServe: ", err.String())
	}
}

