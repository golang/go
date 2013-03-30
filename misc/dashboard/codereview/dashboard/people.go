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
		"agl",
		"bradfitz",
		"campoy",
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
	// Other people.
	others := map[string]string{
		"adonovan": "adonovan@google.com",
		"brainman": "alex.brainman@gmail.com",
		"ality":    "ality@pbrane.org",
		"dfc":      "dave@cheney.net",
		"dvyukov":  "dvyukov@google.com",
		"gustavo":  "gustavo@niemeyer.net",
		"jsing":    "jsing@google.com",
		"mikio":    "mikioh.mikioh@gmail.com",
		"minux":    "minux.ma@gmail.com",
		"remy":     "remyoudompheng@gmail.com",
		"rminnich": "rminnich@gmail.com",
	}
	for p, e := range others {
		personList = append(personList, p)
		emailToPerson[e] = p
		preferredEmail[p] = e
	}

	sort.Strings(personList)
}
