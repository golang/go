package dashboard

// This file handles identities of people.

import (
	"sort"
)

var (
	emailToPerson = make(map[string]string)
	personList    []string
)

func init() {
	// People we assume have golang.org and google.com accounts.
	gophers := [...]string{
		"adg",
		"bradfitz",
		"dsymonds",
		"gri",
		"iant",
		"nigeltao",
		"r",
		"rsc",
	}
	for _, p := range gophers {
		personList = append(personList, p)
		emailToPerson[p+"@golang.org"] = p
		emailToPerson[p+"@google.com"] = p
	}

	sort.Strings(personList)
}
