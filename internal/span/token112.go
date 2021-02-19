// Copyright 2019 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

//go:build go1.12
// +build go1.12

package span

import (
	"go/token"
)

// TODO(rstambler): Delete this file when we no longer support Go 1.11.
func lineStart(f *token.File, line int) token.Pos {
	return f.LineStart(line)
}
