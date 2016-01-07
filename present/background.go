// Copyright 2016 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

package present

import (
	"strings"
)

func init() {
	Register("background", parseBackground)
}

type Background struct {
	URL string
}

func (i Background) TemplateName() string { return "background" }

func parseBackground(ctx *Context, fileName string, lineno int, text string) (Elem, error) {
	args := strings.Fields(text)
	background := Background{URL: args[1]}
	return background, nil
}
