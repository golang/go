// Copyright 2015 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

package internal

// This file contains matchers that implement CLDR inheritance.
//
//     See https://unicode.org/reports/tr35/#Locale_Inheritance.
//
// Some of the inheritance described in this document is already handled by
// the cldr package.

import (
	"golang.org/x/text/language"
)

// TODO: consider if (some of the) matching algorithm needs to be public after
// getting some feel about what is generic and what is specific.

// NewInheritanceMatcher returns a matcher that matches based on the inheritance
// chain.
//
// The matcher uses canonicalization and the parent relationship to find a
// match. The resulting match will always be either Und or a language with the
// same language and script as the requested language. It will not match
// languages for which there is understood to be mutual or one-directional
// intelligibility.
//
// A Match will indicate an Exact match if the language matches after
// canonicalization and High if the matched tag is a parent.
func NewInheritanceMatcher(t []language.Tag) *InheritanceMatcher {
	tags := &InheritanceMatcher{make(map[language.Tag]int)}
	for i, tag := range t {
		ct, err := language.All.Canonicalize(tag)
		if err != nil {
			ct = tag
		}
		tags.index[ct] = i
	}
	return tags
}

type InheritanceMatcher struct {
	index map[language.Tag]int
}

func (m InheritanceMatcher) Match(want ...language.Tag) (language.Tag, int, language.Confidence) {
	for _, t := range want {
		ct, err := language.All.Canonicalize(t)
		if err != nil {
			ct = t
		}
		conf := language.Exact
		for {
			if index, ok := m.index[ct]; ok {
				return ct, index, conf
			}
			if ct == language.Und {
				break
			}
			ct = ct.Parent()
			conf = language.High
		}
	}
	return language.Und, 0, language.No
}
