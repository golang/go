// Copyright 2011 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

package html

import (
	"bytes"
	"exp/template"
	"testing"
)

type data struct {
	World  string
	Coming bool
}

func TestReverse(t *testing.T) {
	templateSource :=
		"{{if .Coming}}Hello{{else}}Goodbye{{end}}, {{.World}}!"
	templateData := data{
		World:  "Cincinatti",
		Coming: true,
	}

	tmpl := template.New("test")
	tmpl, err := tmpl.Parse(templateSource)
	if err != nil {
		t.Errorf("failed to parse template: %s", err)
		return
	}

	Reverse(tmpl)

	buffer := new(bytes.Buffer)

	err = tmpl.Execute(buffer, templateData)
	if err != nil {
		t.Errorf("failed to execute reversed template: %s", err)
		return
	}

	golden := "!ittanicniC ,olleH"
	actual := buffer.String()
	if golden != actual {
		t.Errorf("reversed output: %q != %q", golden, actual)
	}
}
