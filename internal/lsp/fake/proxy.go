// Copyright 2020 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

package fake

import (
	"fmt"

	"golang.org/x/tools/internal/proxydir"
)

// Proxy is a file-based module proxy.
type Proxy struct {
	proxydir string
}

// NewProxy creates a new proxy file tree using the txtar-encoded content.
func NewProxy(tmpdir, txt string) (*Proxy, error) {
	files := unpackTxt(txt)
	type moduleVersion struct {
		modulePath, version string
	}
	// Transform into the format expected by the proxydir package.
	filesByModule := make(map[moduleVersion]map[string][]byte)
	for name, data := range files {
		modulePath, version, suffix := splitModuleVersionPath(name)
		mv := moduleVersion{modulePath, version}
		if _, ok := filesByModule[mv]; !ok {
			filesByModule[mv] = make(map[string][]byte)
		}
		filesByModule[mv][suffix] = data
	}
	for mv, files := range filesByModule {
		if err := proxydir.WriteModuleVersion(tmpdir, mv.modulePath, mv.version, files); err != nil {
			return nil, fmt.Errorf("error writing %s@%s: %v", mv.modulePath, mv.version, err)
		}
	}
	return &Proxy{proxydir: tmpdir}, nil
}

// GOPROXY returns the GOPROXY environment variable value for this proxy
// directory.
func (p *Proxy) GOPROXY() string {
	return proxydir.ToURL(p.proxydir)
}
