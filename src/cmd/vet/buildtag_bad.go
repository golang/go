// This file contains misplaced or malformed build constraints.
// The Go tool will skip it, because the constraints are invalid.
// It serves only to test the tag checker during make test.

// Mention +build // ERROR "possible malformed \+build comment"

// +build !!bang // ERROR "invalid double negative in build constraint"
// +build @#$ // ERROR "invalid non-alphanumeric build constraint"

// +build toolate // ERROR "build comment appears too late in file"
package main
