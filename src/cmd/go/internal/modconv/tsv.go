// Copyright 2018 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

package modconv

import (
	"strings"

	"cmd/go/internal/module"
)

func ParseDependenciesTSV(file string, data []byte) ([]module.Version, error) {
	var list []module.Version
	for lineno, line := range strings.Split(string(data), "\n") {
		lineno++
		f := strings.Split(line, "\t")
		if len(f) >= 3 {
			list = append(list, module.Version{Path: f[0], Version: f[2]})
		}
	}
	return list, nil
}
