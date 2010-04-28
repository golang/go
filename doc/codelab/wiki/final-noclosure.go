package main

import (
	"http"
	"io/ioutil"
	"os"
	"regexp"
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

func viewHandler(c *http.Conn, r *http.Request) {
	title, err := getTitle(c, r)
	if err != nil {
		return
	}
	p, err := loadPage(title)
	if err != nil {
		http.Redirect(c, "/edit/"+title, http.StatusFound)
		return
	}
	renderTemplate(c, "view", p)
}

func editHandler(c *http.Conn, r *http.Request) {
	title, err := getTitle(c, r)
	if err != nil {
		return
	}
	p, err := loadPage(title)
	if err != nil {
		p = &page{title: title}
	}
	renderTemplate(c, "edit", p)
}

func saveHandler(c *http.Conn, r *http.Request) {
	title, err := getTitle(c, r)
	if err != nil {
		return
	}
	body := r.FormValue("body")
	p := &page{title: title, body: []byte(body)}
	err = p.save()
	if err != nil {
		http.Error(c, err.String(), http.StatusInternalServerError)
		return
	}
	http.Redirect(c, "/view/"+title, http.StatusFound)
}

func renderTemplate(c *http.Conn, tmpl string, p *page) {
	t, err := template.ParseFile(tmpl+".html", nil)
	if err != nil {
		http.Error(c, err.String(), http.StatusInternalServerError)
		return
	}
	err = t.Execute(p, c)
	if err != nil {
		http.Error(c, err.String(), http.StatusInternalServerError)
	}
}

const lenPath = len("/view/")

var titleValidator = regexp.MustCompile("^[a-zA-Z0-9]+$")

func getTitle(c *http.Conn, r *http.Request) (title string, err os.Error) {
	title = r.URL.Path[lenPath:]
	if !titleValidator.MatchString(title) {
		http.NotFound(c, r)
		err = os.NewError("Invalid Page Title")
	}
	return
}

func main() {
	http.HandleFunc("/view/", viewHandler)
	http.HandleFunc("/edit/", editHandler)
	http.HandleFunc("/save/", saveHandler)
	http.ListenAndServe(":8080", nil)
}
