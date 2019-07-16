// Copyright 2011 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

package main

import (
	"flag"
	"fmt"
	"io"
	"log"
	"net"
	"net/http"
	"os"
	"strings"
	"time"
)

var (
	post = flag.String("post", "", "urlencoded form data to POST")
	addr = flag.Bool("addr", false, "find open address and print to stdout")
	wait = flag.Duration("wait_for_port", 0, "if non-zero, the amount of time to wait for the address to become available")
)

func main() {
	flag.Parse()
	if *addr {
		l, err := net.Listen("tcp", "127.0.0.1:0")
		if err != nil {
			log.Fatal(err)
		}
		defer l.Close()
		fmt.Print(l.Addr())
		return
	}
	url := flag.Arg(0)
	if url == "" {
		log.Fatal("no url supplied")
	}
	var r *http.Response
	var err error
	loopUntil := time.Now().Add(*wait)
	for {
		if *post != "" {
			b := strings.NewReader(*post)
			r, err = http.Post(url, "application/x-www-form-urlencoded", b)
		} else {
			r, err = http.Get(url)
		}
		if err == nil || *wait == 0 || time.Now().After(loopUntil) {
			break
		}
		time.Sleep(100 * time.Millisecond)
	}
	if err != nil {
		log.Fatal(err)
	}
	defer r.Body.Close()
	_, err = io.Copy(os.Stdout, r.Body)
	if err != nil {
		log.Fatal(err)
	}
}
