// Copyright 2018 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

package doc

import (
	"os"
	"regexp"
	"strings"
	"sync"
)

type lazyRE struct {
	str  string
	once sync.Once
	rx   *regexp.Regexp
}

func (r *lazyRE) re() *regexp.Regexp {
	r.once.Do(r.build)
	return r.rx
}

func (r *lazyRE) build() {
	r.rx = regexp.MustCompile(r.str)
	r.str = ""
}

func (r *lazyRE) FindStringSubmatchIndex(s string) []int {
	return r.re().FindStringSubmatchIndex(s)
}

func (r *lazyRE) ReplaceAllString(src, repl string) string {
	return r.re().ReplaceAllString(src, repl)
}

func (r *lazyRE) MatchString(s string) bool {
	return r.re().MatchString(s)
}

var inTest = len(os.Args) > 0 && strings.HasSuffix(strings.TrimSuffix(os.Args[0], ".exe"), ".test")

func newLazyRE(str string) *lazyRE {
	lr := &lazyRE{str: str}
	if inTest {
		// In tests, always compile the regexps early.
		lr.re()
	}
	return lr
}
