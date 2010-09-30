package main

import (
	"http"
	"io/ioutil"
	"os"
	"template"
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

func editHandler(w http.ResponseWriter, r *http.Request) {
	title := r.URL.Path[lenPath:]
	p, err := loadPage(title)
	if err != nil {
		p = &page{title: title}
	}
	t, _ := template.ParseFile("edit.html", nil)
	t.Execute(p, w)
}

func viewHandler(w http.ResponseWriter, r *http.Request) {
	title := r.URL.Path[lenPath:]
	p, _ := loadPage(title)
	t, _ := template.ParseFile("view.html", nil)
	t.Execute(p, w)
}

func main() {
	http.HandleFunc("/view/", viewHandler)
	http.HandleFunc("/edit/", editHandler)
	http.ListenAndServe(":8080", nil)
}
