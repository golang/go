// Copyright 2018 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

package modconv

import (
	"encoding/json"

	"golang.org/x/mod/modfile"
	"golang.org/x/mod/module"
)

func ParseGodepsJSON(file string, data []byte) (*modfile.File, error) {
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
	mf := new(modfile.File)
	for _, d := range cfg.Deps {
		mf.Require = append(mf.Require, &modfile.Require{Mod: module.Version{Path: d.ImportPath, Version: d.Rev}})
	}
	return mf, nil
}
