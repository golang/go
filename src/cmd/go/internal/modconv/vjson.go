// Copyright 2018 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

package modconv

import (
	"encoding/json"

	"cmd/go/internal/modfile"
	"cmd/go/internal/module"
)

func ParseVendorJSON(file string, data []byte) (*modfile.File, error) {
	var cfg struct {
		Package []struct {
			Path     string
			Revision string
		}
	}
	if err := json.Unmarshal(data, &cfg); err != nil {
		return nil, err
	}
	mf := new(modfile.File)
	for _, d := range cfg.Package {
		mf.Require = append(mf.Require, &modfile.Require{Mod: module.Version{Path: d.Path, Version: d.Revision}})
	}
	return mf, nil
}
