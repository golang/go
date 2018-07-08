// Copyright 2018 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

package modconv

import (
	"strings"

	"cmd/go/internal/module"
)

func ParseVendorConf(file string, data []byte) ([]module.Version, error) {
	var list []module.Version
	for lineno, line := range strings.Split(string(data), "\n") {
		lineno++
		if i := strings.Index(line, "#"); i >= 0 {
			line = line[:i]
		}
		f := strings.Fields(line)
		if len(f) >= 2 {
			list = append(list, module.Version{Path: f[0], Version: f[1]})
		}
	}
	return list, nil
}
