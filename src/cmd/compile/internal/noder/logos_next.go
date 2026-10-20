// Copyright 2026 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

package noder

import (
	"encoding/json"
	"os"

	"cmd/compile/internal/base"
	"cmd/compile/internal/logos"
	"cmd/compile/internal/syntax"
	"cmd/compile/internal/types2"
)

// logosNextHook scans a checked package for logos next migration
// rewrites when -logos.mode=next names a sidecar. It runs for Go and
// loGo/s packages alike after the successful check, appends one JSON
// line per package with findings, and never fails the build.
func logosNextHook(pkg *types2.Package, info *types2.Info, files []*syntax.File) {
	if base.LogosMode != "next" || base.LogosNextOut == "" {
		return
	}
	src, names := nextInputs(files)
	if len(src) == 0 {
		return
	}
	reports := logos.ScanNext(info, files, src, names)
	if len(reports) == 0 {
		return
	}
	line, err := json.Marshal(logos.NextPackageReport{
		Version: logos.NextVersion,
		Package: pkg.Path(),
		Files:   reports,
	})
	if err != nil {
		return
	}
	line = append(line, '\n')
	f, err := os.OpenFile(base.LogosNextOut, os.O_APPEND|os.O_CREATE|os.O_WRONLY, 0666)
	if err != nil {
		return
	}
	defer f.Close()
	_, _ = f.Write(line)
}

// nextInputs copies the stashed sources and filenames of files.
func nextInputs(files []*syntax.File) (map[*syntax.File][]byte, map[*syntax.File]string) {
	logosStore.Lock()
	defer logosStore.Unlock()
	src := make(map[*syntax.File][]byte, len(files))
	names := make(map[*syntax.File]string, len(files))
	for _, file := range files {
		if data, ok := logosStore.src[file]; ok {
			src[file] = data
		}
		if name, ok := logosStore.names[file]; ok {
			names[file] = name
		}
	}
	return src, names
}
