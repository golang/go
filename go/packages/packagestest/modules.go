// Copyright 2018 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

package packagestest

import (
	"bytes"
	"fmt"
	"io/ioutil"
	"path"
	"path/filepath"
)

// Modules is the exporter that produces module layouts.
// Each "repository" is put in it's own module, and the module file generated
// will have replace directives for all other modules.
// Given the two files
//     golang.org/repoa#a/a.go
//     golang.org/repob#b/b.go
// You would get the directory layout
//     /sometemporarydirectory
//     ├── repoa
//     │   ├── a
//     │   │   └── a.go
//     │   └── go.mod
//     └── repob
//         ├── b
//         │   └── b.go
//         └── go.mod
// and the working directory would be
//     /sometemporarydirectory/repoa
var Modules = modules{}

type modules struct{}

func (modules) Name() string {
	return "Modules"
}

func (modules) Filename(exported *Exported, module, fragment string) string {
	return filepath.Join(moduleDir(exported, module), fragment)
}

func (modules) Finalize(exported *Exported) error {
	exported.Config.Env = append(exported.Config.Env, "GO111MODULE=on")
	for module, files := range exported.written {
		dir := gopathDir(exported, module)
		if module == gorootModule {
			exported.Config.Env = append(exported.Config.Env, "GOROOT="+dir)
			continue
		}
		buf := &bytes.Buffer{}
		fmt.Fprintf(buf, "module %v\n", module)
		// add replace directives to the paths of all other modules written
		for other := range exported.written {
			if other == gorootModule || other == module {
				continue
			}
			fmt.Fprintf(buf, "replace %v => %v\n", other, moduleDir(exported, other))
		}
		modfile := filepath.Join(dir, "go.mod")
		if err := ioutil.WriteFile(modfile, buf.Bytes(), 0644); err != nil {
			return err
		}
		files["go.mod"] = modfile
		if module == exported.primary {
			exported.Config.Dir = dir
		}
	}
	return nil
}

func moduleDir(exported *Exported, module string) string {
	return filepath.Join(exported.temp, path.Base(module))
}
