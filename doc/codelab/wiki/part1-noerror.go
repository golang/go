package main

import (
	"fmt"
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

func loadPage(title string) *page {
	filename := title + ".txt"
	body, _ := ioutil.ReadFile(filename)
	return &page{title: title, body: body}
}

func main() {
	p1 := &page{title: "TestPage", body: []byte("This is a sample page.")}
	p1.save()
	p2 := loadPage("TestPage")
	fmt.Println(string(p2.body))
}
