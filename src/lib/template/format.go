// Copyright 2009 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

// Template library: default formatters

package template

import (
	"fmt";
	"io";
	"reflect";
)

// HtmlFormatter formats arbitrary values for HTML
// TODO: do something for real.
func HtmlFormatter(w io.Write, value interface{}, format string) {
	fmt.Fprint(w, value);
}

// StringFormatter formats into the default string representation.
// It is stored under the name "str" and is the default formatter.
// You can override the default formatter by storing your default
// under the name "" in your custom formatter map.
func StringFormatter(w io.Write, value interface{}, format string) {
	fmt.Fprint(w, value);
}
