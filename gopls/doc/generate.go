// Copyright 2020 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

// Command generate updates settings.md from the UserOptions struct.
package main

import (
	"bytes"
	"encoding/json"
	"fmt"
	"io/ioutil"
	"os"
	"regexp"

	"golang.org/x/tools/internal/lsp/source"
)

func main() {
	if err := doMain(); err != nil {
		fmt.Fprintf(os.Stderr, "Generation failed: %v\n", err)
		os.Exit(1)
	}
}

func doMain() error {
	var opts map[string][]option
	if err := json.Unmarshal([]byte(source.OptionsJson), &opts); err != nil {
		return err
	}

	doc, err := ioutil.ReadFile("gopls/doc/settings.md")
	if err != nil {
		return err
	}

	content, err := rewriteDoc(doc, opts)
	if err != nil {
		return err
	}

	if err := ioutil.WriteFile("gopls/doc/settings.md", content, 0); err != nil {
		return err
	}

	return nil
}

type option struct {
	Name       string
	Type       string
	Doc        string
	EnumValues []string
	Default    string
}

func rewriteDoc(doc []byte, categories map[string][]option) ([]byte, error) {
	var err error
	for _, cat := range []string{"User", "Experimental"} {
		doc, err = rewriteSection(doc, categories, cat)
		if err != nil {
			return nil, err
		}
	}
	return doc, nil
}

func rewriteSection(doc []byte, categories map[string][]option, category string) ([]byte, error) {
	section := bytes.NewBuffer(nil)
	for _, opt := range categories[category] {
		var enumValues string
		if len(opt.EnumValues) > 0 {
			enumValues = "Must be one of:\n\n"
			for _, val := range opt.EnumValues {
				enumValues += fmt.Sprintf(" * `%v`\n", val)
			}
		}
		fmt.Fprintf(section, "### **%v** *%v*\n%v%v\n\nDefault: `%v`.\n", opt.Name, opt.Type, opt.Doc, enumValues, opt.Default)
	}
	re := regexp.MustCompile(fmt.Sprintf(`(?s)<!-- BEGIN %v.* -->\n(.*?)<!-- END %v.* -->`, category, category))
	idx := re.FindSubmatchIndex(doc)
	if idx == nil {
		return nil, fmt.Errorf("could not find section %v", category)
	}
	result := append([]byte(nil), doc[:idx[2]]...)
	result = append(result, section.Bytes()...)
	result = append(result, doc[idx[3]:]...)
	return result, nil
}
