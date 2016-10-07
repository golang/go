// Copyright 2012 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

package present

import (
	"fmt"
	"html/template"
	"strings"
	"testing"
)

func TestParseCode(t *testing.T) {
	// Enable play but revert the change at the end.
	defer func(play bool) { PlayEnabled = play }(PlayEnabled)
	PlayEnabled = true

	helloTest := []byte(`
package main

import "fmt"

func main() {
	fmt.Println("hello, test")
}
`)
	helloTestHTML := template.HTML(`
<pre><span num="2">package main</span>
<span num="3"></span>
<span num="4">import &#34;fmt&#34;</span>
<span num="5"></span>
<span num="6">func main() {</span>
<span num="7">    fmt.Println(&#34;hello, test&#34;)</span>
<span num="8">}</span>
</pre>
`)
	helloTestHL := []byte(`
package main

import "fmt" // HLimport

func main() { // HLfunc
	fmt.Println("hello, test") // HL
}
`)
	highlight := func(h template.HTML, s string) template.HTML {
		return template.HTML(strings.Replace(string(h), s, "<b>"+s+"</b>", -1))
	}
	read := func(b []byte, err error) func(string) ([]byte, error) {
		return func(string) ([]byte, error) { return b, err }
	}

	tests := []struct {
		name       string
		readFile   func(string) ([]byte, error)
		sourceFile string
		sourceLine int
		cmd        string
		err        string
		Code
	}{
		{
			name:       "all code, no play",
			readFile:   read(helloTest, nil),
			sourceFile: "main.go",
			cmd:        ".code main.go",
			Code: Code{
				Ext:      ".go",
				FileName: "main.go",
				Raw:      helloTest,
				Text:     helloTestHTML,
			},
		},
		{
			name:       "all code, play",
			readFile:   read(helloTest, nil),
			sourceFile: "main.go",
			cmd:        ".play main.go",
			Code: Code{
				Ext:      ".go",
				FileName: "main.go",
				Play:     true,
				Raw:      helloTest,
				Text:     helloTestHTML,
			},
		},
		{
			name:       "all code, highlighted",
			readFile:   read(helloTestHL, nil),
			sourceFile: "main.go",
			cmd:        ".code main.go",
			Code: Code{
				Ext:      ".go",
				FileName: "main.go",
				Raw:      helloTestHL,
				Text:     highlight(helloTestHTML, "fmt.Println(&#34;hello, test&#34;)"),
			},
		},
		{
			name:       "highlight only func",
			readFile:   read(helloTestHL, nil),
			sourceFile: "main.go",
			cmd:        ".code main.go HLfunc",
			Code: Code{
				Ext:      ".go",
				FileName: "main.go",
				Play:     false,
				Raw:      []byte("package main\n\nimport \"fmt\" // HLimport\n\nfunc main() { // HLfunc\n\tfmt.Println(\"hello, test\") // HL\n}"),
				Text:     highlight(helloTestHTML, "func main() {"),
			},
		},
		{
			name:       "bad highlight syntax",
			readFile:   read(helloTest, nil),
			sourceFile: "main.go",
			cmd:        ".code main.go HL",
			err:        "invalid highlight syntax",
		},
		{
			name:       "error reading file",
			readFile:   read(nil, fmt.Errorf("nope")),
			sourceFile: "main.go",
			cmd:        ".code main.go",
			err:        "main.go:0: nope",
		},
		{
			name:       "from func main to the end",
			readFile:   read(helloTest, nil),
			sourceFile: "main.go",
			cmd:        ".code main.go /func main/,",
			Code: Code{
				Ext:      ".go",
				FileName: "main.go",
				Play:     false,
				Raw:      []byte("func main() {\n\tfmt.Println(\"hello, test\")\n}"),
				Text:     "<pre><span num=\"6\">func main() {</span>\n<span num=\"7\">    fmt.Println(&#34;hello, test&#34;)</span>\n<span num=\"8\">}</span>\n</pre>",
			},
		},
		{
			name:       "just func main",
			readFile:   read(helloTest, nil),
			sourceFile: "main.go",
			cmd:        ".code main.go /func main/",
			Code: Code{
				Ext:      ".go",
				FileName: "main.go",
				Play:     false,
				Raw:      []byte("func main() {"),
				Text:     "<pre><span num=\"6\">func main() {</span>\n</pre>",
			},
		},
		{
			name:       "bad address",
			readFile:   read(helloTest, nil),
			sourceFile: "main.go",
			cmd:        ".code main.go /function main/",
			err:        "main.go:0: no match for function main",
		},
		{
			name:       "all code with  numbers",
			readFile:   read(helloTest, nil),
			sourceFile: "main.go",
			cmd:        ".code -numbers main.go",
			Code: Code{
				Ext:      ".go",
				FileName: "main.go",
				Raw:      helloTest,
				// Replacing the first "<pre>"
				Text: "<pre class=\"numbers\">" + helloTestHTML[6:],
			},
		},
		{
			name:       "all code editable",
			readFile:   read(helloTest, nil),
			sourceFile: "main.go",
			cmd:        ".code -edit main.go",
			Code: Code{
				Ext:      ".go",
				FileName: "main.go",
				Raw:      helloTest,
				Text:     "<pre contenteditable=\"true\" spellcheck=\"false\">" + helloTestHTML[6:],
			},
		},
	}

	trimHTML := func(t template.HTML) string { return strings.TrimSpace(string(t)) }
	trimBytes := func(b []byte) string { return strings.TrimSpace(string(b)) }

	for _, tt := range tests {
		ctx := &Context{tt.readFile}
		e, err := parseCode(ctx, tt.sourceFile, 0, tt.cmd)
		if err != nil {
			if tt.err == "" {
				t.Errorf("%s: unexpected error %v", tt.name, err)
			} else if !strings.Contains(err.Error(), tt.err) {
				t.Errorf("%s: expected error %s; got %v", tt.name, tt.err, err)
			}
			continue
		}
		if tt.err != "" {
			t.Errorf("%s: expected error %s; but got none", tt.name, tt.err)
			continue
		}
		c, ok := e.(Code)
		if !ok {
			t.Errorf("%s: expected a Code value; got %T", tt.name, e)
			continue
		}
		if c.FileName != tt.FileName {
			t.Errorf("%s: expected FileName %s; got %s", tt.name, tt.FileName, c.FileName)
		}
		if c.Ext != tt.Ext {
			t.Errorf("%s: expected Ext %s; got %s", tt.name, tt.Ext, c.Ext)
		}
		if c.Play != tt.Play {
			t.Errorf("%s: expected Play %v; got %v", tt.name, tt.Play, c.Play)
		}
		if got, wants := trimBytes(c.Raw), trimBytes(tt.Raw); got != wants {
			t.Errorf("%s: expected Raw \n%q\n; got \n%q\n", tt.name, wants, got)
		}
		if got, wants := trimHTML(c.Text), trimHTML(tt.Text); got != wants {
			t.Errorf("%s: expected Text \n%q\n; got \n%q\n", tt.name, wants, got)
		}
	}
}
