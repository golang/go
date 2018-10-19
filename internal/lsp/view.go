package lsp

import (
	"fmt"
	"go/token"
	"strings"
	"sync"

	"golang.org/x/tools/go/packages"
	"golang.org/x/tools/internal/lsp/protocol"
)

type view struct {
	activeFilesMu sync.Mutex
	activeFiles   map[protocol.DocumentURI][]byte

	fset *token.FileSet
}

func newView() *view {
	return &view{
		activeFiles: make(map[protocol.DocumentURI][]byte),
		fset:        token.NewFileSet(),
	}
}

func (v *view) overlay() map[string][]byte {
	over := make(map[string][]byte)

	v.activeFilesMu.Lock()
	defer v.activeFilesMu.Unlock()

	for uri, content := range v.activeFiles {
		over[uriToFilename(uri)] = content
	}
	return over
}

func (v *view) readActiveFile(uri protocol.DocumentURI) ([]byte, error) {
	v.activeFilesMu.Lock()
	defer v.activeFilesMu.Unlock()

	content, ok := v.activeFiles[uri]
	if !ok {
		return nil, fmt.Errorf("file not found: %s", uri)
	}
	return content, nil
}

func (v *view) clearActiveFile(uri protocol.DocumentURI) {
	v.activeFilesMu.Lock()
	delete(v.activeFiles, uri)
	v.activeFilesMu.Unlock()
}

// typeCheck type-checks the package for the given package path.
func (v *view) typeCheck(uri protocol.DocumentURI) (*packages.Package, error) {
	cfg := &packages.Config{
		Mode:    packages.LoadSyntax,
		Fset:    v.fset,
		Overlay: v.overlay(),
		Tests:   true,
	}
	pkgs, err := packages.Load(cfg, fmt.Sprintf("file=%s", uriToFilename(uri)))
	if len(pkgs) == 0 {
		return nil, err
	}
	pkg := pkgs[0]
	return pkg, nil
}

func uriToFilename(uri protocol.DocumentURI) string {
	return strings.TrimPrefix(string(uri), "file://")
}

func filenameToURI(filename string) protocol.DocumentURI {
	return protocol.DocumentURI("file://" + filename)
}
