// Copyright 2009 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

package main

import (
	"bufio";
	"flag";
	"fmt";
	"http";
	"io";
	"net";
	"os";
)


// hello world, the web server
func HelloServer(c *http.Conn, req *http.Request) {
	io.WriteString(c, "hello, world!\n");
}

// simple counter server
type Counter struct {
	n int;
}

func (ctr *Counter) ServeHTTP(c *http.Conn, req *http.Request) {
	fmt.Fprintf(c, "counter = %d\n", ctr.n);
	ctr.n++;
}

// simple file server
var webroot = flag.String("root", "/home/rsc", "web root directory")
func FileServer(c *http.Conn, req *http.Request) {
	c.SetHeader("content-type", "text/plain; charset=utf-8");
	path := *webroot + req.Url.Path;	// TODO: insecure: use os.CleanName
	fd, err := os.Open(path, os.O_RDONLY, 0);
	if err != nil {
		c.WriteHeader(http.StatusNotFound);
		fmt.Fprintf(c, "open %s: %v\n", path, err);
		return;
	}
	n, err1 := io.Copy(fd, c);
	fmt.Fprintf(c, "[%d bytes]\n", n);
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
	http.Handle("/counter", new(Counter));
	http.Handle("/go/", http.HandlerFunc(FileServer));
	http.Handle("/go/hello", http.HandlerFunc(HelloServer));
	http.Handle("/chan", ChanCreate());
	err := http.ListenAndServe(":12345", nil);
	if err != nil {
		panic("ListenAndServe: ", err.String())
	}
}

