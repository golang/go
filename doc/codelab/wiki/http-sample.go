package main

import (
	"fmt"
	"http"
)

func handler(c *http.Conn, r *http.Request) {
	fmt.Fprintf(c, "Hi there, I love %s!", r.URL.Path[1:])
}

func main() {
	http.HandleFunc("/", handler)
	http.ListenAndServe(":8080", nil)
}
