// Copyright 2019 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

package cmdtest

import (
	"fmt"
	"runtime"
	"strings"
	"testing"

	"golang.org/x/tools/internal/lsp/tests"
	"golang.org/x/tools/internal/span"
)

const (
	expectedDefinitionsCount     = 28
	expectedTypeDefinitionsCount = 2
)

type godefMode int

const (
	plainGodef = godefMode(1 << iota)
	jsonGoDef
)

var godefModes = []godefMode{
	plainGodef,
	jsonGoDef,
}

func (r *runner) Definition(t *testing.T, spn span.Span, d tests.Definition) {
	if d.IsType || d.OnlyHover {
		// TODO: support type definition, hover queries
		return
	}
	d.Src = span.New(d.Src.URI(), span.NewPoint(0, 0, d.Src.Start().Offset()), span.Point{})
	for _, mode := range godefModes {
		args := []string{"query"}
		tag := d.Name + "-definition"
		if mode&jsonGoDef != 0 {
			tag += "-json"
			args = append(args, "-json")
		}
		args = append(args, "definition")
		uri := d.Src.URI()
		args = append(args, fmt.Sprint(d.Src))
		got, _ := r.NormalizeGoplsCmd(t, args...)
		if mode&jsonGoDef != 0 && runtime.GOOS == "windows" {
			got = strings.Replace(got, "file:///", "file://", -1)
		}
		expect := strings.TrimSpace(string(r.data.Golden(tag, uri.Filename(), func() ([]byte, error) {
			return []byte(got), nil
		})))
		if expect != "" && !strings.HasPrefix(got, expect) {
			t.Errorf("definition %v failed with %#v expected:\n%q\ngot:\n%q", tag, args, expect, got)
		}
	}
}
