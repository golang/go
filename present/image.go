// Copyright 2012 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

package present

import (
	"fmt"
	"strings"
)

func init() {
	Register("image", parseImage)
}

type Image struct {
	Cmd    string // original command from present source
	URL    string
	Width  int
	Height int
}

func (i Image) PresentCmd() string   { return i.Cmd }
func (i Image) TemplateName() string { return "image" }

func parseImage(ctx *Context, fileName string, lineno int, text string) (Elem, error) {
	args := strings.Fields(text)
	if len(args) < 2 {
		return nil, fmt.Errorf("incorrect image invocation: %q", text)
	}
	img := Image{Cmd: text, URL: args[1]}
	a, err := parseArgs(fileName, lineno, args[2:])
	if err != nil {
		return nil, err
	}
	switch len(a) {
	case 0:
		// no size parameters
	case 2:
		// If a parameter is empty (underscore) or invalid
		// leave the field set to zero. The "image" action
		// template will then omit that img tag attribute and
		// the browser will calculate the value to preserve
		// the aspect ratio.
		if v, ok := a[0].(int); ok {
			img.Height = v
		}
		if v, ok := a[1].(int); ok {
			img.Width = v
		}
	default:
		return nil, fmt.Errorf("incorrect image invocation: %q", text)
	}
	return img, nil
}
