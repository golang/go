// Copyright 2020 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

// Package proxydir provides functions for writing module data to a directory
// in proxy format, so that it can be used as a module proxy by setting
// GOPROXY="file://<dir>".
package proxydir

import (
	"archive/zip"
	"fmt"
	"io"
	"io/ioutil"
	"os"
	"path/filepath"
)

// WriteModuleVersion creates a directory in the proxy dir for a module.
func WriteModuleVersion(rootDir, module, ver string, files map[string][]byte) (rerr error) {
	dir := filepath.Join(rootDir, module, "@v")
	if err := os.MkdirAll(dir, 0755); err != nil {
		return err
	}

	// The go command checks for versions by looking at the "list" file.  Since
	// we are supporting multiple versions, create this file if it does not exist
	// or append the version number to the preexisting file.
	f, err := os.OpenFile(filepath.Join(dir, "list"), os.O_APPEND|os.O_CREATE|os.O_WRONLY, 0644)
	if err != nil {
		return err
	}
	defer checkClose("list file", f, &rerr)
	if _, err := f.WriteString(ver + "\n"); err != nil {
		return err
	}

	// Serve the go.mod file on the <version>.mod url, if it exists. Otherwise,
	// serve a stub.
	modContents, ok := files["go.mod"]
	if !ok {
		modContents = []byte("module " + module)
	}
	if err := ioutil.WriteFile(filepath.Join(dir, ver+".mod"), modContents, 0644); err != nil {
		return err
	}

	// info file, just the bare bones.
	infoContents := []byte(fmt.Sprintf(`{"Version": "%v", "Time":"2017-12-14T13:08:43Z"}`, ver))
	if err := ioutil.WriteFile(filepath.Join(dir, ver+".info"), infoContents, 0644); err != nil {
		return err
	}

	// zip of all the source files.
	f, err = os.OpenFile(filepath.Join(dir, ver+".zip"), os.O_CREATE|os.O_WRONLY, 0644)
	if err != nil {
		return err
	}
	defer checkClose("zip file", f, &rerr)
	z := zip.NewWriter(f)
	defer checkClose("zip writer", z, &rerr)
	for name, contents := range files {
		zf, err := z.Create(module + "@" + ver + "/" + name)
		if err != nil {
			return err
		}
		if _, err := zf.Write(contents); err != nil {
			return err
		}
	}

	return nil
}

func checkClose(name string, closer io.Closer, err *error) {
	if cerr := closer.Close(); cerr != nil && *err == nil {
		*err = fmt.Errorf("closing %s: %v", name, cerr)
	}
}
