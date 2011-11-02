package main

import (
	"http"
	"flag"
	"fmt"
	"io"
	"log"
	"net"
	"os"
	"strings"
)

var (
	post = flag.String("post", "", "urlencoded form data to POST")
	addr = flag.Bool("addr", false, "find open address and print to stdout")
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
	if *post != "" {
		b := strings.NewReader(*post)
		r, err = http.Post(url, "application/x-www-form-urlencoded", b)
	} else {
		r, err = http.Get(url)
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
