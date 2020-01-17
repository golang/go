// Copyright 2020 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

package go2go

import (
	"bufio"
	"fmt"
	"go/ast"
	"go/printer"
	"go/token"
	"path/filepath"
	"os"
	"strings"
)

var config = printer.Config{
	Mode: printer.UseSpaces | printer.TabIndent | printer.SourcePos,
	Tabwidth: 8,
}

// rewrite rewrites the contents of one file.
func rewrite(dir string, fset *token.FileSet, filename string, ast *ast.File) (err error) {
	filename = filepath.Base(filename)
	goFile := strings.TrimSuffix(filename, filepath.Ext(filename)) + ".go"
	o, err := os.Create(filepath.Join(dir, goFile))
	if err != nil {
		return err
	}
	defer func() {
		if closeErr := o.Close(); err == nil {
			err = closeErr
		}
	}()

	w := bufio.NewWriter(o)
	defer func() {
		if flushErr := w.Flush(); err == nil {
			err = flushErr
		}
	}()
	fmt.Fprintln(w, rewritePrefix)

	return config.Fprint(w, fset, ast)
}
