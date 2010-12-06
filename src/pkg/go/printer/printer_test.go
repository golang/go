// Copyright 2009 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

package printer

import (
	"bytes"
	"flag"
	"io/ioutil"
	"go/ast"
	"go/parser"
	"go/token"
	"path"
	"testing"
)


const (
	dataDir  = "testdata"
	tabwidth = 8
)


var update = flag.Bool("update", false, "update golden files")


var fset = token.NewFileSet()


func lineString(text []byte, i int) string {
	i0 := i
	for i < len(text) && text[i] != '\n' {
		i++
	}
	return string(text[i0:i])
}


type checkMode uint

const (
	export checkMode = 1 << iota
	rawFormat
)


func check(t *testing.T, source, golden string, mode checkMode) {
	// parse source
	prog, err := parser.ParseFile(fset, source, nil, parser.ParseComments)
	if err != nil {
		t.Error(err)
		return
	}

	// filter exports if necessary
	if mode&export != 0 {
		ast.FileExports(prog) // ignore result
		prog.Comments = nil   // don't print comments that are not in AST
	}

	// determine printer configuration
	cfg := Config{Tabwidth: tabwidth}
	if mode&rawFormat != 0 {
		cfg.Mode |= RawFormat
	}

	// format source
	var buf bytes.Buffer
	if _, err := cfg.Fprint(&buf, fset, prog); err != nil {
		t.Error(err)
	}
	res := buf.Bytes()

	// update golden files if necessary
	if *update {
		if err := ioutil.WriteFile(golden, res, 0644); err != nil {
			t.Error(err)
		}
		return
	}

	// get golden
	gld, err := ioutil.ReadFile(golden)
	if err != nil {
		t.Error(err)
		return
	}

	// compare lengths
	if len(res) != len(gld) {
		t.Errorf("len = %d, expected %d (= len(%s))", len(res), len(gld), golden)
	}

	// compare contents
	for i, line, offs := 0, 1, 0; i < len(res) && i < len(gld); i++ {
		ch := res[i]
		if ch != gld[i] {
			t.Errorf("%s:%d:%d: %s", source, line, i-offs+1, lineString(res, offs))
			t.Errorf("%s:%d:%d: %s", golden, line, i-offs+1, lineString(gld, offs))
			t.Error()
			return
		}
		if ch == '\n' {
			line++
			offs = i + 1
		}
	}
}


type entry struct {
	source, golden string
	mode           checkMode
}

// Use gotest -update to create/update the respective golden files.
var data = []entry{
	{"empty.input", "empty.golden", 0},
	{"comments.input", "comments.golden", 0},
	{"comments.input", "comments.x", export},
	{"linebreaks.input", "linebreaks.golden", 0},
	{"expressions.input", "expressions.golden", 0},
	{"expressions.input", "expressions.raw", rawFormat},
	{"declarations.input", "declarations.golden", 0},
	{"statements.input", "statements.golden", 0},
}


func Test(t *testing.T) {
	for _, e := range data {
		source := path.Join(dataDir, e.source)
		golden := path.Join(dataDir, e.golden)
		check(t, source, golden, e.mode)
		// TODO(gri) check that golden is idempotent
		//check(t, golden, golden, e.mode);
	}
}
