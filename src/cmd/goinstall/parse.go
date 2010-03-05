// Copyright 2010 The Go Authors.  All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

// Wrappers for Go parser.

package main

import (
	"path"
	"os"
	"log"
	"strings"
	"strconv"
	"go/ast"
	"go/parser"
)

// goFiles returns a list of the *.go source files in dir,
// excluding those in package main or ending in _test.go.
// It also returns a map giving the packages imported
// by those files.  The map keys are the imported paths.
// The key's value is one file that imports that path.
func goFiles(dir string) (files []string, imports map[string]string, err os.Error) {
	f, err := os.Open(dir, os.O_RDONLY, 0)
	if err != nil {
		return nil, nil, err
	}
	dirs, err := f.Readdir(-1)
	f.Close()
	if err != nil {
		return nil, nil, err
	}

	files = make([]string, 0, len(dirs))
	imports = make(map[string]string)
	pkgName := ""
	for i := range dirs {
		d := &dirs[i]
		if !strings.HasSuffix(d.Name, ".go") || strings.HasSuffix(d.Name, "_test.go") {
			continue
		}
		filename := path.Join(dir, d.Name)
		pf, err := parser.ParseFile(filename, nil, nil, parser.ImportsOnly)
		if err != nil {
			return nil, nil, err
		}
		s := string(pf.Name.Name())
		if s == "main" {
			continue
		}
		if pkgName == "" {
			pkgName = s
		} else if pkgName != s {
			return nil, nil, os.ErrorString("multiple package names in " + dir)
		}
		n := len(files)
		files = files[0 : n+1]
		files[n] = filename
		for _, decl := range pf.Decls {
			for _, spec := range decl.(*ast.GenDecl).Specs {
				quoted := string(spec.(*ast.ImportSpec).Path.Value)
				unquoted, err := strconv.Unquote(quoted)
				if err != nil {
					log.Crashf("%s: parser returned invalid quoted string: <%s>", filename, quoted)
				}
				imports[unquoted] = filename
			}
		}
	}
	return files, imports, nil
}
