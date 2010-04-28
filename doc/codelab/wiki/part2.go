package main

import (
	"fmt"
	"http"
	"io/ioutil"
	"os"
)

type page struct {
	title string
	body  []byte
}

func (p *page) save() os.Error {
	filename := p.title + ".txt"
	return ioutil.WriteFile(filename, p.body, 0600)
}

func loadPage(title string) (*page, os.Error) {
	filename := title + ".txt"
	body, err := ioutil.ReadFile(filename)
	if err != nil {
		return nil, err
	}
	return &page{title: title, body: body}, nil
}

const lenPath = len("/view/")

func viewHandler(c *http.Conn, r *http.Request) {
	title := r.URL.Path[lenPath:]
	p, _ := loadPage(title)
	fmt.Fprintf(c, "<h1>%s</h1><div>%s</div>", p.title, p.body)
}

func main() {
	http.HandleFunc("/view/", viewHandler)
	http.ListenAndServe(":8080", nil)
}
