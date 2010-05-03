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

func editHandler(c *http.Conn, r *http.Request) {
	title := r.URL.Path[lenPath:]
	p, err := loadPage(title)
	if err != nil {
		p = &page{title: title}
	}
	renderTemplate(c, "edit", p)
}

func viewHandler(c *http.Conn, r *http.Request) {
	title := r.URL.Path[lenPath:]
	p, _ := loadPage(title)
	renderTemplate(c, "view", p)
}

func saveHandler(c *http.Conn, r *http.Request) {
	title := r.URL.Path[lenPath:]
	body := r.FormValue("body")
	p := &page{title: title, body: []byte(body)}
	p.save()
	http.Redirect(c, "/view/"+title, http.StatusFound)
}

func renderTemplate(c *http.Conn, tmpl string, p *page) {
	t, _ := template.ParseFile(tmpl+".html", nil)
	t.Execute(p, c)
}

func main() {
	http.HandleFunc("/view/", viewHandler)
	http.HandleFunc("/edit/", editHandler)
	http.HandleFunc("/save/", saveHandler)
	http.ListenAndServe(":8080", nil)
}
