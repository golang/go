// Copyright 2013 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

package present

import (
	"fmt"
	"strings"
)

func init() {
	Register("iframe", parseIframe)
}

type Iframe struct {
	Cmd    string // original command from present source
	URL    string
	Width  int
	Height int
}

func (i Iframe) PresentCmd() string   { return i.Cmd }
func (i Iframe) TemplateName() string { return "iframe" }

func parseIframe(ctx *Context, fileName string, lineno int, text string) (Elem, error) {
	args := strings.Fields(text)
	if len(args) < 2 {
		return nil, fmt.Errorf("incorrect iframe invocation: %q", text)
	}
	i := Iframe{Cmd: text, URL: args[1]}
	a, err := parseArgs(fileName, lineno, args[2:])
	if err != nil {
		return nil, err
	}
	switch len(a) {
	case 0:
		// no size parameters
	case 2:
		if v, ok := a[0].(int); ok {
			i.Height = v
		}
		if v, ok := a[1].(int); ok {
			i.Width = v
		}
	default:
		return nil, fmt.Errorf("incorrect iframe invocation: %q", text)
	}
	return i, nil
}
