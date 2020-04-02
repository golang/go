// Copyright 2019 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

package packages

import (
	"fmt"
	"strings"
)

var allModes = []LoadMode{
	NeedName,
	NeedFiles,
	NeedCompiledGoFiles,
	NeedImports,
	NeedDeps,
	NeedExportsFile,
	NeedTypes,
	NeedSyntax,
	NeedTypesInfo,
	NeedTypesSizes,
}

var modeStrings = []string{
	"NeedName",
	"NeedFiles",
	"NeedCompiledGoFiles",
	"NeedImports",
	"NeedDeps",
	"NeedExportsFile",
	"NeedTypes",
	"NeedSyntax",
	"NeedTypesInfo",
	"NeedTypesSizes",
}

func (mod LoadMode) String() string {
	m := mod
	if m == 0 {
		return "LoadMode(0)"
	}
	var out []string
	for i, x := range allModes {
		if x > m {
			break
		}
		if (m & x) != 0 {
			out = append(out, modeStrings[i])
			m = m ^ x
		}
	}
	if m != 0 {
		out = append(out, "Unknown")
	}
	return fmt.Sprintf("LoadMode(%s)", strings.Join(out, "|"))
}
