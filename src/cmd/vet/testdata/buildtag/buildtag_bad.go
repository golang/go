// This file contains misplaced or malformed build constraints.
// The Go tool will skip it, because the constraints are invalid.
// It serves only to test the tag checker during make test.

// Mention +build // ERROR "possible malformed \+build comment"

// +build !!bang // ERROR "invalid double negative in build constraint"
// +build @#$ // ERROR "invalid non-alphanumeric build constraint"

// +build toolate // ERROR "build comment must appear before package clause and be followed by a blank line"
package bad

// This is package 'bad' rather than 'main' so the erroneous build
// tag doesn't end up looking like a package doc for the vet command
// when examined by godoc.
