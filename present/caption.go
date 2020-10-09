// Copyright 2012 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

package present

import "strings"

func init() {
	Register("caption", parseCaption)
}

type Caption struct {
	Cmd  string // original command from present source
	Text string
}

func (c Caption) PresentCmd() string   { return c.Cmd }
func (c Caption) TemplateName() string { return "caption" }

func parseCaption(_ *Context, _ string, _ int, cmd string) (Elem, error) {
	text := strings.TrimSpace(strings.TrimPrefix(cmd, ".caption"))
	return Caption{cmd, text}, nil
}
