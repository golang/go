// Copyright 2018 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

package lazyregexp

import (
	"os"
	"regexp"
	"strings"
	"sync"
)

type Regexp struct {
	str  string
	once sync.Once
	rx   *regexp.Regexp
}

func (r *Regexp) re() *regexp.Regexp {
	r.once.Do(r.build)
	return r.rx
}

func (r *Regexp) build() {
	r.rx = regexp.MustCompile(r.str)
	r.str = ""
}

func (r *Regexp) FindSubmatch(s []byte) [][]byte {
	return r.re().FindSubmatch(s)
}

func (r *Regexp) FindStringSubmatch(s string) []string {
	return r.re().FindStringSubmatch(s)
}

func (r *Regexp) FindStringSubmatchIndex(s string) []int {
	return r.re().FindStringSubmatchIndex(s)
}

func (r *Regexp) ReplaceAllString(src, repl string) string {
	return r.re().ReplaceAllString(src, repl)
}

func (r *Regexp) FindString(s string) string {
	return r.re().FindString(s)
}

func (r *Regexp) FindAllString(s string, n int) []string {
	return r.re().FindAllString(s, n)
}

func (r *Regexp) MatchString(s string) bool {
	return r.re().MatchString(s)
}

func (r *Regexp) SubexpNames() []string {
	return r.re().SubexpNames()
}

var inTest = len(os.Args) > 0 && strings.HasSuffix(strings.TrimSuffix(os.Args[0], ".exe"), ".test")

func New(str string) *Regexp {
	lr := &Regexp{str: str}
	if inTest {
		// In tests, always compile the regexps early.
		lr.re()
	}
	return lr
}
