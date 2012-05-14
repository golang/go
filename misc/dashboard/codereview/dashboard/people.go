// Copyright 2012 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

package dashboard

// This file handles identities of people.

import (
	"sort"
)

var (
	emailToPerson  = make(map[string]string) // email => person
	preferredEmail = make(map[string]string) // person => email
	personList     []string
)

func init() {
	// People we assume have golang.org and google.com accounts,
	// and prefer to use their golang.org address for code review.
	gophers := [...]string{
		"adg",
		"bradfitz",
		"dsymonds",
		"gri",
		"iant",
		"nigeltao",
		"r",
		"rsc",
		"sameer",
	}
	for _, p := range gophers {
		personList = append(personList, p)
		emailToPerson[p+"@golang.org"] = p
		emailToPerson[p+"@google.com"] = p
		preferredEmail[p] = p + "@golang.org"
	}

	sort.Strings(personList)
}
