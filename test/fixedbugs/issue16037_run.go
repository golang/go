// run

//go:build !nacl && !js && !wasip1 && !android && !gccgo

// Copyright 2016 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

package main

import (
	"bytes"
	"fmt"
	"html/template"
	"io/ioutil"
	"log"
	"os"
	"os/exec"
	"path/filepath"
)

var tmpl = template.Must(template.New("main").Parse(`
package main

type T struct {
    {{range .Names}}
	{{.Name}} *string
	{{end}}
}

{{range .Names}}
func (t *T) Get{{.Name}}() string {
	if t.{{.Name}} == nil {
		return ""
	}
	return *t.{{.Name}}
}
{{end}}

func main() {}
`))

func main() {
	const n = 5000

	type Name struct{ Name string }
	var t struct{ Names []Name }
	for i := 0; i < n; i++ {
		t.Names = append(t.Names, Name{Name: fmt.Sprintf("H%06X", i)})
	}

	buf := new(bytes.Buffer)
	if err := tmpl.Execute(buf, t); err != nil {
		log.Fatal(err)
	}

	dir, err := ioutil.TempDir("", "issue16037-")
	if err != nil {
		log.Fatal(err)
	}
	defer os.RemoveAll(dir)
	path := filepath.Join(dir, "ridiculous_number_of_fields.go")
	if err := ioutil.WriteFile(path, buf.Bytes(), 0664); err != nil {
		log.Fatal(err)
	}

	out, err := exec.Command("go", "build", "-o="+filepath.Join(dir, "out"), path).CombinedOutput()
	if err != nil {
		log.Fatalf("build failed: %v\n%s", err, out)
	}
}
