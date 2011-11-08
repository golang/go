package main

import (
	"io/ioutil"
	"os"
	"text/template"
)

func main() {
	b, _ := ioutil.ReadAll(os.Stdin)
	template.HTMLEscape(os.Stdout, b)
}
