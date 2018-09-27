package lsp

import (
	"fmt"
	"sync"

	"golang.org/x/tools/internal/lsp/protocol"
)

type view struct {
	activeFilesMu sync.Mutex
	activeFiles   map[protocol.DocumentURI]string
}

func newView() *view {
	return &view{
		activeFiles: make(map[protocol.DocumentURI]string),
	}
}

func (v *view) cacheActiveFile(uri protocol.DocumentURI, text string) {
	v.activeFilesMu.Lock()
	v.activeFiles[uri] = text
	v.activeFilesMu.Unlock()
}

func (v *view) readActiveFile(uri protocol.DocumentURI) (string, error) {
	v.activeFilesMu.Lock()
	defer v.activeFilesMu.Unlock()

	content, ok := v.activeFiles[uri]
	if !ok {
		return "", fmt.Errorf("file not found: %s", uri)
	}
	return content, nil
}

func (v *view) clearActiveFile(uri protocol.DocumentURI) {
	v.activeFilesMu.Lock()
	delete(v.activeFiles, uri)
	v.activeFilesMu.Unlock()
}
