// Copyright 2009 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

package main

import (
	"bufio";
	"exvar";
	"flag";
	"fmt";
	"http";
	"io";
	"log";
	"net";
	"os";
)


// hello world, the web server
var helloRequests = exvar.NewInt("hello-requests");
func HelloServer(c *http.Conn, req *http.Request) {
	helloRequests.Add(1);
	io.WriteString(c, "hello, world!\n");
}

// simple counter server
type Counter struct {
	n int;
}

// This makes Counter satisfy the exvar.Var interface, so we can export
// it directly.
func (ctr *Counter) String() string {
	return fmt.Sprintf("%d", ctr.n)
}

func (ctr *Counter) ServeHTTP(c *http.Conn, req *http.Request) {
	fmt.Fprintf(c, "counter = %d\n", ctr.n);
	ctr.n++;
}

// simple file server
var webroot = flag.String("root", "/home/rsc", "web root directory")
var pathVar = exvar.NewMap("file-requests");
func FileServer(c *http.Conn, req *http.Request) {
	c.SetHeader("content-type", "text/plain; charset=utf-8");
	pathVar.Add(req.Url.Path, 1);
	path := *webroot + req.Url.Path;	// TODO: insecure: use os.CleanName
	f, err := os.Open(path, os.O_RDONLY, 0);
	if err != nil {
		c.WriteHeader(http.StatusNotFound);
		fmt.Fprintf(c, "open %s: %v\n", path, err);
		return;
	}
	n, err1 := io.Copy(f, c);
	fmt.Fprintf(c, "[%d bytes]\n", n);
	f.Close();
}

// simple flag server
var booleanflag = flag.Bool("boolean", true, "another flag for testing")
func FlagServer(c *http.Conn, req *http.Request) {
	c.SetHeader("content-type", "text/plain; charset=utf-8");
	fmt.Fprint(c, "Flags:\n");
	flag.VisitAll(func (f *flag.Flag) {
		if f.Value.String() != f.DefValue {
			fmt.Fprintf(c, "%s = %s [default = %s]\n", f.Name, f.Value.String(), f.DefValue);
		} else {
			fmt.Fprintf(c, "%s = %s\n", f.Name, f.Value.String());
		}
	});
}

// simple argument server
func ArgServer(c *http.Conn, req *http.Request) {
	for i, s := range sys.Args {
		fmt.Fprint(c, s, " ");
	}
}

// a channel (just for the fun of it)
type Chan chan int

func ChanCreate() Chan {
	c := make(Chan);
	go func(c Chan) {
		for x := 0;; x++ {
			c <- x
		}
	}(c);
	return c;
}

func (ch Chan) ServeHTTP(c *http.Conn, req *http.Request) {
	io.WriteString(c, fmt.Sprintf("channel send #%d\n", <-ch));
}

func main() {
	flag.Parse();

	// The counter is published as a variable directly.
	ctr := new(Counter);
	http.Handle("/counter", ctr);
	exvar.Publish("counter", ctr);

	http.Handle("/go/", http.HandlerFunc(FileServer));
	http.Handle("/flags", http.HandlerFunc(FlagServer));
	http.Handle("/args", http.HandlerFunc(ArgServer));
	http.Handle("/go/hello", http.HandlerFunc(HelloServer));
	http.Handle("/chan", ChanCreate());
	err := http.ListenAndServe(":12345", nil);
	if err != nil {
		log.Crash("ListenAndServe: ", err)
	}
}

