// Copyright 2011 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

// This benchmark tests text/template throughput,
// converting a large data structure with a simple template.

package go1

import (
	"bytes"
	"io"
	"strings"
	"testing"
	"text/template"
)

// After removing \t and \n this generates identical output to
// json.Marshal, making it easy to test for correctness.
const tmplText = `
{
	"tree":{{template "node" .Tree}},
	"username":"{{.Username}}"
}
{{define "node"}}
{
	"name":"{{.Name}}",
	"kids":[
	{{range $i, $k := .Kids}}
		{{if $i}}
			,
		{{end}}
		{{template "node" $k}}
	{{end}}
	],
	"cl_weight":{{.CLWeight}},
	"touches":{{.Touches}},
	"min_t":{{.MinT}},
	"max_t":{{.MaxT}},
	"mean_t":{{.MeanT}}
}
{{end}}
`

func stripTabNL(r rune) rune {
	if r == '\t' || r == '\n' {
		return -1
	}
	return r
}

func makeTemplate(jsonbytes []byte, jsondata *JSONResponse) *template.Template {
	tmpl := template.Must(template.New("main").Parse(strings.Map(stripTabNL, tmplText)))

	var buf bytes.Buffer
	if err := tmpl.Execute(&buf, &jsondata); err != nil {
		panic(err)
	}
	if !bytes.Equal(buf.Bytes(), jsonbytes) {
		println(buf.Len(), len(jsonbytes))
		panic("wrong output")
	}
	return tmpl
}

func tmplexec(tmpl *template.Template, jsondata *JSONResponse) {
	if err := tmpl.Execute(io.Discard, jsondata); err != nil {
		panic(err)
	}
}

func BenchmarkTemplate(b *testing.B) {
	jsonbytes := makeJsonBytes()
	jsondata := makeJsonData(jsonbytes)
	tmpl := makeTemplate(jsonbytes, jsondata)
	b.ResetTimer()
	b.SetBytes(int64(len(jsonbytes)))
	for i := 0; i < b.N; i++ {
		tmplexec(tmpl, jsondata)
	}
}
