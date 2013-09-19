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
	URL    string
	Width  int
	Height int
}

func (i Image) TemplateName() string { return "image" }

func parseImage(ctx *Context, fileName string, lineno int, text string) (Elem, error) {
	args := strings.Fields(text)
	img := Image{URL: args[1]}
	a, err := parseArgs(fileName, lineno, args[2:])
	if err != nil {
		return nil, err
	}
	switch len(a) {
	case 0:
		// no size parameters
	case 2:
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
