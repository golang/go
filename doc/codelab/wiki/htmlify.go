package main

import (
	"os"
	"template"
	"io/ioutil"
)

func main() {
	b, _ := ioutil.ReadAll(os.Stdin)
	template.HTMLFormatter(os.Stdout, b, "")
}
