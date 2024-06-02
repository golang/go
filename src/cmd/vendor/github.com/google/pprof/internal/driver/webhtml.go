// Copyright 2017 Google Inc. All Rights Reserved.
//
// Licensed under the Apache License, Version 2.0 (the "License");
// you may not use this file except in compliance with the License.
// You may obtain a copy of the License at
//
//     http://www.apache.org/licenses/LICENSE-2.0
//
// Unless required by applicable law or agreed to in writing, software
// distributed under the License is distributed on an "AS IS" BASIS,
// WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
// See the License for the specific language governing permissions and
// limitations under the License.

package driver

import (
	"embed"
	"fmt"
	"html/template"
	"os"
	"sync"

	"github.com/google/pprof/internal/report"
)

var (
	htmlTemplates    *template.Template // Lazily loaded templates
	htmlTemplateInit sync.Once
)

// getHTMLTemplates returns the set of HTML templates used by pprof,
// initializing them if necessary.
func getHTMLTemplates() *template.Template {
	htmlTemplateInit.Do(func() {
		htmlTemplates = template.New("templategroup")
		addTemplates(htmlTemplates)
		report.AddSourceTemplates(htmlTemplates)
	})
	return htmlTemplates
}

//go:embed html
var embeddedFiles embed.FS

// addTemplates adds a set of template definitions to templates.
func addTemplates(templates *template.Template) {
	// Load specified file.
	loadFile := func(fname string) string {
		data, err := embeddedFiles.ReadFile(fname)
		if err != nil {
			fmt.Fprintf(os.Stderr, "internal/driver: embedded file %q not found\n",
				fname)
			os.Exit(1)
		}
		return string(data)
	}
	loadCSS := func(fname string) string {
		return `<style type="text/css">` + "\n" + loadFile(fname) + `</style>` + "\n"
	}
	loadJS := func(fname string) string {
		return `<script>` + "\n" + loadFile(fname) + `</script>` + "\n"
	}

	// Define a named template with specified contents.
	def := func(name, contents string) {
		sub := template.New(name)
		template.Must(sub.Parse(contents))
		template.Must(templates.AddParseTree(name, sub.Tree))
	}

	// Embedded files.
	def("css", loadCSS("html/common.css"))
	def("header", loadFile("html/header.html"))
	def("graph", loadFile("html/graph.html"))
	def("graph_css", loadCSS("html/graph.css"))
	def("script", loadJS("html/common.js"))
	def("top", loadFile("html/top.html"))
	def("sourcelisting", loadFile("html/source.html"))
	def("plaintext", loadFile("html/plaintext.html"))
	// TODO: Rename "stacks" to "flamegraph" to seal moving off d3 flamegraph.
	def("stacks", loadFile("html/stacks.html"))
	def("stacks_css", loadCSS("html/stacks.css"))
	def("stacks_js", loadJS("html/stacks.js"))
}
