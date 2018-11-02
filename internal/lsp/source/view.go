// Copyright 2018 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

package source

import (
	"fmt"
	"go/token"
	"sync"

	"golang.org/x/tools/go/packages"
	"golang.org/x/tools/internal/lsp/protocol"
)

type View struct {
	Config *packages.Config

	activeFilesMu sync.Mutex
	activeFiles   map[protocol.DocumentURI][]byte

	fset *token.FileSet
}

func NewView() *View {
	fset := token.NewFileSet()
	return &View{
		Config: &packages.Config{
			Mode:  packages.LoadSyntax,
			Fset:  fset,
			Tests: true,
		},
		activeFiles: make(map[protocol.DocumentURI][]byte),
		fset:        fset,
	}
}

func (v *View) overlay() map[string][]byte {
	over := make(map[string][]byte)

	v.activeFilesMu.Lock()
	defer v.activeFilesMu.Unlock()

	for uri, content := range v.activeFiles {
		filename, err := FromURI(uri)
		if err == nil {
			over[filename] = content
		}
	}
	return over
}

func (v *View) SetActiveFileContent(uri protocol.DocumentURI, content []byte) {
	v.activeFilesMu.Lock()
	v.activeFiles[uri] = content
	v.activeFilesMu.Unlock()
}

func (v *View) ReadActiveFile(uri protocol.DocumentURI) ([]byte, error) {
	v.activeFilesMu.Lock()
	content, ok := v.activeFiles[uri]
	v.activeFilesMu.Unlock()
	if !ok {
		return nil, fmt.Errorf("uri not found: %s", uri)
	}
	return content, nil
}

func (v *View) ClearActiveFile(uri protocol.DocumentURI) {
	v.activeFilesMu.Lock()
	delete(v.activeFiles, uri)
	v.activeFilesMu.Unlock()
}

// TypeCheck type-checks the package for the given package path.
func (v *View) TypeCheck(uri protocol.DocumentURI) (*packages.Package, error) {
	v.Config.Overlay = v.overlay()
	path, err := FromURI(uri)
	if err != nil {
		return nil, err
	}
	pkgs, err := packages.Load(v.Config, fmt.Sprintf("file=%s", path))
	if len(pkgs) == 0 {
		return nil, err
	}
	pkg := pkgs[0]
	return pkg, nil
}
