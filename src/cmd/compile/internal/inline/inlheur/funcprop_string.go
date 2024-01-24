// Copyright 2023 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

package inlheur

import (
	"fmt"
	"strings"
)

func (fp *FuncProps) String() string {
	return fp.ToString("")
}

func (fp *FuncProps) ToString(prefix string) string {
	var sb strings.Builder
	if fp.Flags != 0 {
		fmt.Fprintf(&sb, "%sFlags %s\n", prefix, fp.Flags)
	}
	flagSliceToSB[ParamPropBits](&sb, fp.ParamFlags,
		prefix, "ParamFlags")
	flagSliceToSB[ResultPropBits](&sb, fp.ResultFlags,
		prefix, "ResultFlags")
	return sb.String()
}

func flagSliceToSB[T interface {
	~uint32
	String() string
}](sb *strings.Builder, sl []T, prefix string, tag string) {
	var sb2 strings.Builder
	foundnz := false
	fmt.Fprintf(&sb2, "%s%s\n", prefix, tag)
	for i, e := range sl {
		if e != 0 {
			foundnz = true
		}
		fmt.Fprintf(&sb2, "%s  %d %s\n", prefix, i, e.String())
	}
	if foundnz {
		sb.WriteString(sb2.String())
	}
}
