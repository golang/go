// Copyright 2012 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

package cookiejar

import (
	"time"
)

// Storage is a Jar's storage. It is a multi-map, mapping keys to one or more
// entries. Each entry consists of a subkey, creation time, last access time,
// and some arbitrary data.
//
// The Add and Delete methods have undefined behavior if the key is invalid.
// A valid key must use only bytes in the character class [a-z0-9.-] and
// must have at least one non-. byte. Note that this excludes any key
// containing a capital ASCII letter as well as the empty string.
type Storage interface {
	// A client must call Lock before calling other methods and must call
	// Unlock when finished. Between the calls to Lock and Unlock, a client
	// can assume that other clients are not modifying the Storage.
	Lock()
	Unlock()

	// Add adds entries to the storage. Each entry's Subkey and Data must
	// both be non-empty.
	//
	// If the Storage already contains an entry with the same key and
	// subkey then the new entry is stored with the creation time of the
	// old entry, and the old entry is deleted.
	//
	// Adding entries may cause other entries to be deleted, to maintain an
	// implementation-specific storage constraint.
	Add(key string, entries ...Entry) error

	// Delete deletes all entries for the given key.
	Delete(key string) error

	// Entries calls f for each entry stored for that key. If f returns a
	// non-nil error then the iteration stops and Entries returns that
	// error. Iteration is not guaranteed to be in any particular order.
	//
	// If f returns an Update action then that stored entry's LastAccess
	// time will be set to the time that f returned. If f returns a Delete
	// action then that entry will be deleted from the Storage.
	//
	// f may call a Storage's Add and Delete methods; those modifications
	// will not affect the list of entries visited in this call to Entries.
	Entries(key string, f func(entry Entry) (Action, time.Time, error)) error

	// Keys calls f for each key stored. f will not be called on a key with
	// zero entries. If f returns a non-nil error then the iteration stops
	// and Keys returns that error. Iteration is not guaranteed to be in any
	// particular order.
	//
	// f may call a Storage's Add, Delete and Entries methods; those
	// modifications will not affect the list of keys visited in this call
	// to Keys.
	Keys(f func(key string) error) error
}

// Entry is an entry in a Storage.
type Entry struct {
	Subkey     string
	Data       string
	Creation   time.Time
	LastAccess time.Time
}

// Action is an action returned by the function passed to Entries.
type Action int

const (
	// Pass means to take no further action with an Entry.
	Pass Action = iota
	// Update means to update the LastAccess time of an Entry.
	Update
	// Delete means to delete an Entry.
	Delete
)

// ValidStorageKey returns whether the given key is valid for a Storage.
func ValidStorageKey(key string) bool {
	hasNonDot := false
	for i := 0; i < len(key); i++ {
		switch c := key[i]; {
		case 'a' <= c && c <= 'z':
			fallthrough
		case '0' <= c && c <= '9':
			fallthrough
		case c == '-':
			hasNonDot = true
		case c == '.':
			// No-op.
		default:
			return false
		}
	}
	return hasNonDot
}
