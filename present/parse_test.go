// Copyright 2020 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

package present

import (
	"bytes"
	"html/template"
	"io/ioutil"
	"os"
	"os/exec"
	"path/filepath"
	"runtime"
	"testing"
)

func TestTestdata(t *testing.T) {
	tmpl := template.Must(Template().Parse(testTmpl))
	filesP, err := filepath.Glob("testdata/*.p")
	if err != nil {
		t.Fatal(err)
	}
	filesMD, err := filepath.Glob("testdata/*.md")
	if err != nil {
		t.Fatal(err)
	}
	files := append(filesP, filesMD...)
	for _, file := range files {
		file := file
		name := filepath.Base(file)
		if name == "README" {
			continue
		}
		t.Run(name, func(t *testing.T) {
			data, err := ioutil.ReadFile(file)
			if err != nil {
				t.Fatalf("%s: %v", file, err)
			}
			marker := []byte("\n---\n")
			i := bytes.Index(data, marker)
			if i < 0 {
				t.Fatalf("%s: cannot find --- marker in input", file)
			}
			input, html := data[:i+1], data[i+len(marker):]
			doc, err := Parse(bytes.NewReader(input), name, 0)
			if err != nil {
				t.Fatalf("%s: %v", file, err)
			}
			var buf bytes.Buffer
			if err := doc.Render(&buf, tmpl); err != nil {
				t.Fatalf("%s: %v", file, err)
			}
			if !bytes.Equal(buf.Bytes(), html) {
				diffText, err := diff("present-test-", "want", html, "have", buf.Bytes())
				if err != nil {
					t.Fatalf("%s: diff: %v", file, err)
				}
				t.Errorf("%s: incorrect html:\n%s", file, diffText)
			}
		})
	}
}

func diff(prefix string, name1 string, b1 []byte, name2 string, b2 []byte) ([]byte, error) {
	f1, err := writeTempFile(prefix, b1)
	if err != nil {
		return nil, err
	}
	defer os.Remove(f1)

	f2, err := writeTempFile(prefix, b2)
	if err != nil {
		return nil, err
	}
	defer os.Remove(f2)

	cmd := "diff"
	if runtime.GOOS == "plan9" {
		cmd = "/bin/ape/diff"
	}

	data, err := exec.Command(cmd, "-u", f1, f2).CombinedOutput()
	if len(data) > 0 {
		// diff exits with a non-zero status when the files don't match.
		// Ignore that failure as long as we get output.
		err = nil
	}

	data = bytes.Replace(data, []byte(f1), []byte(name1), -1)
	data = bytes.Replace(data, []byte(f2), []byte(name2), -1)

	return data, err
}

func writeTempFile(prefix string, data []byte) (string, error) {
	file, err := ioutil.TempFile("", prefix)
	if err != nil {
		return "", err
	}
	_, err = file.Write(data)
	if err1 := file.Close(); err == nil {
		err = err1
	}
	if err != nil {
		os.Remove(file.Name())
		return "", err
	}
	return file.Name(), nil
}

var testTmpl = `
{{define "root" -}}
<h1>{{.Title}}</h1>
{{with .Subtitle}}<h2>{{.}}</h2>
{{end -}}
{{range .Authors}}<author>
{{range .Elem}}{{elem $.Template .}}{{end}}</author>
{{end -}}
{{range .Sections}}<section>{{elem $.Template .}}</section>
{{end -}}
{{end}}

{{define "newline"}}{{/* No automatic line break. Paragraphs are free-form. */}}
{{end}}

{{define "section"}}
{{if .Title}}<h2 id="TOC_{{.FormattedNumber}}">{{.Title}}</h2>
{{end -}}
{{range .Elem}}{{elem $.Template .}}{{end}}
{{- end}}

{{define "list" -}}
<ul>
{{range .Bullet -}}
<li>{{style .}}</li>
{{end -}}
</ul>
{{end}}

{{define "text" -}}
{{if .Pre -}}
<pre>{{range .Lines}}{{.}}{{end}}</pre>
{{else -}}
<p>{{range $i, $l := .Lines}}{{if $i}}{{template "newline"}}{{end}}{{style $l}}{{end}}</p>
{{end -}}
{{end}}

{{define "code" -}}
{{if .Play -}}
<div class="playground">{{.Text}}</div>
{{else -}}
<div class="code">{{.Text}}</div>
{{end -}}
{{end}}

{{define "image" -}}
<img src="{{.URL}}"{{with .Height}} height="{{.}}"{{end}}{{with .Width}} width="{{.}}"{{end}} alt="">
{{end}}

{{define "caption" -}}
<figcaption>{{style .Text}}</figcaption>
{{end}}

{{define "iframe" -}}
<iframe src="{{.URL}}"{{with .Height}} height="{{.}}"{{end}}{{with .Width}} width="{{.}}"{{end}}></iframe>
{{end}}

{{define "link" -}}
<p class="link"><a href="{{.URL}}">{{style .Label}}</a></p>
{{end}}

{{define "html" -}}{{.HTML}}{{end}}
`
