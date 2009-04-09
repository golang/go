// Copyright 2009 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

// Template library: default formatters

package template

import (
	"fmt";
	"reflect";
)

// HtmlFormatter formats arbitrary values for HTML
// TODO: do something for real.
func HtmlFormatter(v reflect.Value) string {
	s := fmt.Sprint(reflect.Indirect(v).Interface());
	return s;
}

// StringFormatter formats returns the default string representation.
// It is stored under the name "str" and is the default formatter.
// You can override the default formatter by storing your default
// under the name "" in your custom formatter map.
func StringFormatter(v reflect.Value) string {
	s := fmt.Sprint(reflect.Indirect(v).Interface());
	return s;
}
