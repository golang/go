// Copyright 2018 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

package debug

import (
	"strings"
)

// exported from runtime
func modinfo() string

// ReadBuildInfo returns the build information embedded
// in the running binary. The information is available only
// in binaries built with module support.
func ReadBuildInfo() (info *BuildInfo, ok bool) {
	return readBuildInfo(modinfo())
}

// BuildInfo represents the build information read from
// the running binary.
type BuildInfo struct {
	Path string    // The main package path
	Main Module    // The main module information
	Deps []*Module // Module dependencies
}

// Module represents a module.
type Module struct {
	Path    string  // module path
	Version string  // module version
	Sum     string  // checksum
	Replace *Module // replaced by this module
}

func readBuildInfo(data string) (*BuildInfo, bool) {
	if len(data) < 32 {
		return nil, false
	}
	data = data[16 : len(data)-16]

	const (
		pathLine = "path\t"
		modLine  = "mod\t"
		depLine  = "dep\t"
		repLine  = "=>\t"
	)

	info := &BuildInfo{}

	var line string
	// Reverse of cmd/go/internal/modload.PackageBuildInfo
	for len(data) > 0 {
		i := strings.IndexByte(data, '\n')
		if i < 0 {
			break
		}
		line, data = data[:i], data[i+1:]
		switch {
		case strings.HasPrefix(line, pathLine):
			elem := line[len(pathLine):]
			info.Path = elem
		case strings.HasPrefix(line, modLine):
			elem := strings.Split(line[len(modLine):], "\t")
			if len(elem) != 3 {
				return nil, false
			}
			info.Main = Module{
				Path:    elem[0],
				Version: elem[1],
				Sum:     elem[2],
			}
		case strings.HasPrefix(line, depLine):
			elem := strings.Split(line[len(depLine):], "\t")
			if len(elem) != 2 && len(elem) != 3 {
				return nil, false
			}
			sum := ""
			if len(elem) == 3 {
				sum = elem[2]
			}
			info.Deps = append(info.Deps, &Module{
				Path:    elem[0],
				Version: elem[1],
				Sum:     sum,
			})
		case strings.HasPrefix(line, repLine):
			elem := strings.Split(line[len(repLine):], "\t")
			if len(elem) != 3 {
				return nil, false
			}
			last := len(info.Deps) - 1
			if last < 0 {
				return nil, false
			}
			info.Deps[last].Replace = &Module{
				Path:    elem[0],
				Version: elem[1],
				Sum:     elem[2],
			}
		}
	}
	return info, true
}
