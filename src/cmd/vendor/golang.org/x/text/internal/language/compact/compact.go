// Copyright 2018 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

// Package compact defines a compact representation of language tags.
//
// Common language tags (at least all for which locale information is defined
// in CLDR) are assigned a unique index. Each Tag is associated with such an
// ID for selecting language-related resources (such as translations) as well
// as one for selecting regional defaults (currency, number formatting, etc.)
//
// It may want to export this functionality at some point, but at this point
// this is only available for use within x/text.
package compact // import "golang.org/x/text/internal/language/compact"

import (
	"sort"
	"strings"

	"golang.org/x/text/internal/language"
)

// ID is an integer identifying a single tag.
type ID uint16

func getCoreIndex(t language.Tag) (id ID, ok bool) {
	cci, ok := language.GetCompactCore(t)
	if !ok {
		return 0, false
	}
	i := sort.Search(len(coreTags), func(i int) bool {
		return cci <= coreTags[i]
	})
	if i == len(coreTags) || coreTags[i] != cci {
		return 0, false
	}
	return ID(i), true
}

// Parent returns the ID of the parent or the root ID if id is already the root.
func (id ID) Parent() ID {
	return parents[id]
}

// Tag converts id to an internal language Tag.
func (id ID) Tag() language.Tag {
	if int(id) >= len(coreTags) {
		return specialTags[int(id)-len(coreTags)]
	}
	return coreTags[id].Tag()
}

var specialTags []language.Tag

func init() {
	tags := strings.Split(specialTagsStr, " ")
	specialTags = make([]language.Tag, len(tags))
	for i, t := range tags {
		specialTags[i] = language.MustParse(t)
	}
}
