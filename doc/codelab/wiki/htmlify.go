package main

import (
	"old/template"
	"os"
	"io/ioutil"
)

func main() {
	b, _ := ioutil.ReadAll(os.Stdin)
	template.HTMLFormatter(os.Stdout, "", b)
}
