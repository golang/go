package main

import (
	"template"
	"os"
	"io/ioutil"
)

func main() {
	b, _ := ioutil.ReadAll(os.Stdin)
	template.HTMLEscape(os.Stdout, b)
}
