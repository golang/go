// Copyright 2020 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

package fake

import (
	"fmt"

	"golang.org/x/tools/internal/proxydir"
)

// WriteProxy creates a new proxy file tree using the txtar-encoded content,
// and returns its URL.
func WriteProxy(tmpdir, txt string) (string, error) {
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
			return "", fmt.Errorf("error writing %s@%s: %v", mv.modulePath, mv.version, err)
		}
	}
	return proxydir.ToURL(tmpdir), nil
}
