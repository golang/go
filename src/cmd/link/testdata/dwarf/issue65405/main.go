package main

import "net/http"

func main() {
	http.Handle("/", http.StripPrefix("/static/", http.FileServer(http.Dir("./output"))))
	http.ListenAndServe(":8000", nil)
}
