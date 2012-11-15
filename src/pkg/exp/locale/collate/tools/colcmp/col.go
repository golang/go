// Copyright 2012 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

package main

import (
	"exp/locale/collate"
	"log"
	"unicode/utf16"
)

// Input holds an input string in both UTF-8 and UTF-16 format.
type Input struct {
	index int // used for restoring to original random order
	UTF8  []byte
	UTF16 []uint16
	key   []byte // used for sorting
}

func (i Input) String() string {
	return string(i.UTF8)
}

func makeInput(s8 []byte, s16 []uint16) Input {
	return Input{UTF8: s8, UTF16: s16}
}

func makeInputString(s string) Input {
	return Input{
		UTF8:  []byte(s),
		UTF16: utf16.Encode([]rune(s)),
	}
}

// Collator is an interface for architecture-specific implementations of collation.
type Collator interface {
	// Key generates a sort key for the given input.  Implemenations
	// may return nil if a collator does not support sort keys.
	Key(s Input) []byte

	// Compare returns -1 if a < b, 1 if a > b and 0 if a == b.
	Compare(a, b Input) int
}

// CollatorFactory creates a Collator for a given locale.
type CollatorFactory struct {
	name        string
	makeFn      func(locale string) (Collator, error)
	description string
}

var collators = []CollatorFactory{}

// AddFactory registers f as a factory for an implementation of Collator.
func AddFactory(f CollatorFactory) {
	collators = append(collators, f)
}

func getCollator(name, locale string) Collator {
	for _, f := range collators {
		if f.name == name {
			col, err := f.makeFn(locale)
			if err != nil {
				log.Fatal(err)
			}
			return col
		}
	}
	log.Fatalf("collator of type %q not found", name)
	return nil
}

// goCollator is an implemention of Collator using go's own collator.
type goCollator struct {
	c   *collate.Collator
	buf collate.Buffer
}

func init() {
	AddFactory(CollatorFactory{"go", newGoCollator, "Go's native collator implementation."})
}

func newGoCollator(locale string) (Collator, error) {
	c := &goCollator{c: collate.New(locale)}
	return c, nil
}

func (c *goCollator) Key(b Input) []byte {
	return c.c.Key(&c.buf, b.UTF8)
}

func (c *goCollator) Compare(a, b Input) int {
	return c.c.Compare(a.UTF8, b.UTF8)
}
