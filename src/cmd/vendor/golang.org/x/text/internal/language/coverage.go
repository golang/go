// Copyright 2014 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

package language

// BaseLanguages returns the list of all supported base languages. It generates
// the list by traversing the internal structures.
func BaseLanguages() []Language {
	base := make([]Language, 0, NumLanguages)
	for i := 0; i < langNoIndexOffset; i++ {
		// We included "und" already for the value 0.
		if i != nonCanonicalUnd {
			base = append(base, Language(i))
		}
	}
	i := langNoIndexOffset
	for _, v := range langNoIndex {
		for k := 0; k < 8; k++ {
			if v&1 == 1 {
				base = append(base, Language(i))
			}
			v >>= 1
			i++
		}
	}
	return base
}
