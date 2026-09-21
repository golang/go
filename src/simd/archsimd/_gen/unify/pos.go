// Copyright 2025 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

package unify

import (
	"fmt"
)

type Pos struct {
	Path string
	Line int
}

func (p Pos) String() string {
	var b []byte
	b, _ = p.AppendText(b)
	return string(b)
}

func (p Pos) AppendText(b []byte) ([]byte, error) {
	if p.Line == 0 {
		if p.Path == "" {
			return append(b, "?:?"...), nil
		} else {
			return append(b, p.Path...), nil
		}
	} else if p.Path == "" {
		return fmt.Appendf(b, "?:%d", p.Line), nil
	}
	return fmt.Appendf(b, "%s:%d", p.Path, p.Line), nil
}
