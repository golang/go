package main

import (
	"http"
	"flag"
	"io"
	"log"
	"os"
	"strings"
)

var post = flag.String("post", "", "urlencoded form data to POST")

func main() {
	flag.Parse()
	url := flag.Arg(0)
	if url == "" {
		log.Exit("no url supplied")
	}
	var r *http.Response
	var err os.Error
	if *post != "" {
		b := strings.NewReader(*post)
		r, err = http.Post(url, "application/x-www-form-urlencoded", b)
	} else {
		r, _, err = http.Get(url)
	}
	if err != nil {
		log.Exit(err)
	}
	defer r.Body.Close()
	_, err = io.Copy(os.Stdout, r.Body)
	if err != nil {
		log.Exit(err)
	}
}
