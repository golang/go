// errorcheck

// Copyright 2010 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

package main

func main() {
	var x map[string]string{"a":"b"}		// ERROR "unexpected { at end of statement|expected ';' or '}' or newline"
