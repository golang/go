// Copyright 2018 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

package packagestest

import (
	"path"
	"path/filepath"
)

// GOPATH is the exporter that produces GOPATH layouts.
// Each "module" is put in it's own GOPATH entry to help test complex cases.
// Given the two files
//     golang.org/repoa#a/a.go
//     golang.org/repob#b/b.go
// You would get the directory layout
//     /sometemporarydirectory
//     ├── repoa
//     │   └── src
//     │       └── golang.org
//     │           └── repoa
//     │               └── a
//     │                   └── a.go
//     └── repob
//         └── src
//             └── golang.org
//                 └── repob
//                     └── b
//                         └── b.go
// GOPATH would be set to
//     /sometemporarydirectory/repoa;/sometemporarydirectory/repob
// and the working directory would be
//     /sometemporarydirectory/repoa/src
var GOPATH = gopath{}

func init() {
	All = append(All, GOPATH)
}

type gopath struct{}

func (gopath) Name() string {
	return "GOPATH"
}

func (gopath) Filename(exported *Exported, module, fragment string) string {
	return filepath.Join(gopathDir(exported, module), "src", module, fragment)
}

func (gopath) Finalize(exported *Exported) error {
	exported.Config.Env = append(exported.Config.Env, "GO111MODULE=off")
	gopath := ""
	for module := range exported.written {
		if gopath != "" {
			gopath += string(filepath.ListSeparator)
		}
		dir := gopathDir(exported, module)
		gopath += dir
		if module == exported.primary {
			exported.Config.Dir = filepath.Join(dir, "src")
		}
	}
	exported.Config.Env = append(exported.Config.Env, "GOPATH="+gopath)
	return nil
}

func gopathDir(exported *Exported, module string) string {
	dir := path.Base(module)
	if versionSuffixRE.MatchString(dir) {
		dir = path.Base(path.Dir(module)) + "_" + dir
	}
	return filepath.Join(exported.temp, dir)
}
