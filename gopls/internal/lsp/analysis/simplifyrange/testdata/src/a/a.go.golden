// Copyright 2020 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

package testdata

import "log"

func m() {
	maps := make(map[string]string)
	for k := range maps { // want "simplify range expression"
		log.Println(k)
	}
	for range maps { // want "simplify range expression"
	}
}
