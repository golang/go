// Copyright 2024 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

// Package impl is a registry of alternative implementations of cryptographic
// primitives, to allow selecting them for testing.
package impl

import "strings"

type implementation struct {
	Package   string
	Name      string
	Available bool
	Toggle    *bool
}

var allImplementations []implementation

// Register records an alternative implementation of a cryptographic primitive.
// The implementation might be available or not based on CPU support. If
// available is false, the implementation is unavailable and can't be tested on
// this machine. If available is true, it can be set to false to disable the
// implementation. If all alternative implementations but one are disabled, the
// remaining one must be used (i.e. disabling one implementation must not
// implicitly disable any other). Each package has an implicit base
// implementation that is selected when all alternatives are unavailable or
// disabled. pkg must be the package name, not path (e.g. "aes" not "crypto/aes").
func Register(pkg, name string, available *bool) {
	if strings.Contains(pkg, "/") {
		panic("impl: package name must not contain slashes")
	}
	allImplementations = append(allImplementations, implementation{
		Package:   pkg,
		Name:      name,
		Available: *available,
		Toggle:    available,
	})
}

// List returns the names of all alternative implementations registered for the
// given package, whether available or not. The implicit base implementation is
// not included.
func List(pkg string) []string {
	var names []string
	for _, i := range allImplementations {
		if i.Package == pkg {
			names = append(names, i.Name)
		}
	}
	return names
}

func available(pkg, name string) bool {
	for _, i := range allImplementations {
		if i.Package == pkg && i.Name == name {
			return i.Available
		}
	}
	panic("unknown implementation")
}

// Select disables all implementations for the given package except the one
// with the given name. If name is empty, the base implementation is selected.
// It returns whether the selected implementation is available.
func Select(pkg, name string) bool {
	if name == "" {
		for _, i := range allImplementations {
			if i.Package == pkg {
				*i.Toggle = false
			}
		}
		return true
	}
	if !available(pkg, name) {
		return false
	}
	for _, i := range allImplementations {
		if i.Package == pkg {
			*i.Toggle = i.Name == name
		}
	}
	return true
}

func Reset(pkg string) {
	for _, i := range allImplementations {
		if i.Package == pkg {
			*i.Toggle = i.Available
			return
		}
	}
}
