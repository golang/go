// Copyright 2018 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

package modconv

import (
	"cmd/go/internal/module"
	"strings"
)

func ParseGlideLock(file string, data []byte) ([]module.Version, error) {
	var list []module.Version
	imports := false
	name := ""
	for lineno, line := range strings.Split(string(data), "\n") {
		lineno++
		if line == "" {
			continue
		}
		if strings.HasPrefix(line, "imports:") {
			imports = true
		} else if line[0] != '-' && line[0] != ' ' && line[0] != '\t' {
			imports = false
		}
		if !imports {
			continue
		}
		if strings.HasPrefix(line, "- name:") {
			name = strings.TrimSpace(line[len("- name:"):])
		}
		if strings.HasPrefix(line, "  version:") {
			version := strings.TrimSpace(line[len("  version:"):])
			if name != "" && version != "" {
				list = append(list, module.Version{Path: name, Version: version})
			}
		}
	}
	return list, nil
}
