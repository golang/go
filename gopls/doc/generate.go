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
	"path/filepath"
	"regexp"
	"strings"

	"golang.org/x/tools/internal/lsp/source"
)

func main() {
	if _, err := doMain(".", true); err != nil {
		fmt.Fprintf(os.Stderr, "Generation failed: %v\n", err)
		os.Exit(1)
	}
}

func doMain(baseDir string, write bool) (bool, error) {
	api := &source.APIJSON{}
	if err := json.Unmarshal([]byte(source.GeneratedAPIJSON), api); err != nil {
		return false, err
	}

	if ok, err := rewriteFile(filepath.Join(baseDir, "gopls/doc/settings.md"), api, write, rewriteSettings); !ok || err != nil {
		return ok, err
	}
	if ok, err := rewriteFile(filepath.Join(baseDir, "gopls/doc/commands.md"), api, write, rewriteCommands); !ok || err != nil {
		return ok, err
	}

	return true, nil
}

func rewriteFile(file string, api *source.APIJSON, write bool, rewrite func([]byte, *source.APIJSON) ([]byte, error)) (bool, error) {
	doc, err := ioutil.ReadFile(file)
	if err != nil {
		return false, err
	}

	content, err := rewrite(doc, api)
	if err != nil {
		return false, fmt.Errorf("rewriting %q: %v", file, err)
	}

	if !bytes.Equal(doc, content) && !write {
		return false, nil
	}

	if err := ioutil.WriteFile(file, content, 0); err != nil {
		return false, err
	}

	return true, nil
}

var parBreakRE = regexp.MustCompile("\n{2,}")

func rewriteSettings(doc []byte, api *source.APIJSON) ([]byte, error) {
	result := doc
	for category, opts := range api.Options {
		section := bytes.NewBuffer(nil)
		for _, opt := range opts {
			var enumValues strings.Builder
			if len(opt.EnumValues) > 0 {
				enumValues.WriteString("Must be one of:\n\n")
				for _, val := range opt.EnumValues {
					if val.Doc != "" {
						// Don't break the list item by starting a new paragraph.
						unbroken := parBreakRE.ReplaceAllString(val.Doc, "\\\n")
						fmt.Fprintf(&enumValues, " * %s\n", unbroken)
					} else {
						fmt.Fprintf(&enumValues, " * `%s`\n", val.Value)
					}
				}
			}
			fmt.Fprintf(section, "### **%v** *%v*\n%v%v\n\nDefault: `%v`.\n", opt.Name, opt.Type, opt.Doc, enumValues.String(), opt.Default)
		}
		var err error
		result, err = replaceSection(result, category, section.Bytes())
		if err != nil {
			return nil, err
		}
	}

	section := bytes.NewBuffer(nil)
	for _, lens := range api.Lenses {
		fmt.Fprintf(section, "### **%v**\nIdentifier: `%v`\n\n%v\n\n", lens.Title, lens.Lens, lens.Doc)
	}
	return replaceSection(result, "Lenses", section.Bytes())
}

func rewriteCommands(doc []byte, api *source.APIJSON) ([]byte, error) {
	section := bytes.NewBuffer(nil)
	for _, command := range api.Commands {
		fmt.Fprintf(section, "### **%v**\nIdentifier: `%v`\n\n%v\n\n", command.Title, command.Command, command.Doc)
	}
	return replaceSection(doc, "Commands", section.Bytes())
}

func replaceSection(doc []byte, sectionName string, replacement []byte) ([]byte, error) {
	re := regexp.MustCompile(fmt.Sprintf(`(?s)<!-- BEGIN %v.* -->\n(.*?)<!-- END %v.* -->`, sectionName, sectionName))
	idx := re.FindSubmatchIndex(doc)
	if idx == nil {
		return nil, fmt.Errorf("could not find section %q", sectionName)
	}
	result := append([]byte(nil), doc[:idx[2]]...)
	result = append(result, replacement...)
	result = append(result, doc[idx[3]:]...)
	return result, nil
}
