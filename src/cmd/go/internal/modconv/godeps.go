// Copyright 2018 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

package modconv

import (
	"encoding/json"

	"cmd/go/internal/module"
)

func ParseGodepsJSON(file string, data []byte) ([]module.Version, error) {
	var cfg struct {
		ImportPath string
		Deps       []struct {
			ImportPath string
			Rev        string
		}
	}
	if err := json.Unmarshal(data, &cfg); err != nil {
		return nil, err
	}
	var list []module.Version
	for _, d := range cfg.Deps {
		list = append(list, module.Version{Path: d.ImportPath, Version: d.Rev})
	}
	return list, nil
}
