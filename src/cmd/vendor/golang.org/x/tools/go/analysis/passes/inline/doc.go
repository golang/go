// Copyright 2025 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

/*
Package inline defines an analyzer that inlines calls to functions
and uses of constants marked with a "//go:fix inline" directive.

# Analyzer inline

inline: apply fixes based on 'go:fix inline' comment directives

The inline analyzer inlines functions, constants, and type aliases
that are marked for inlining.

Use this command to apply (just) inline fixes en masse:

	$ go fix -inline ./...

## Functions

Given a function that is marked for inlining, like this one:

	//go:fix inline
	func Square(x int) int { return Pow(x, 2) }

this analyzer will recommend that calls to the function elsewhere, in the same
or other packages, should be inlined.

Inlining can be used to move off of a deprecated function:

	// Deprecated: prefer Pow(x, 2).
	//go:fix inline
	func Square(x int) int { return Pow(x, 2) }

It can also be used to move off of an obsolete package,
as when the import path has changed or a higher major version is available:

	package pkg

	import pkg2 "pkg/v2"

	//go:fix inline
	func F() { pkg2.F(nil) }

Replacing a call pkg.F() by pkg2.F(nil) can have no effect on the program,
so this mechanism provides a low-risk way to update large numbers of calls.
We recommend, where possible, expressing the old API in terms of the new one
to enable automatic migration.

The inliner takes care to avoid behavior changes, even subtle ones,
such as changes to the order in which argument expressions are
evaluated. When it cannot safely eliminate all parameter variables,
it may introduce a "binding declaration" of the form

	var params = args

to evaluate argument expressions in the correct order and bind them to
parameter variables. Since the resulting code transformation may be
stylistically suboptimal, such inlinings may be disabled by specifying
the -inline.allow_binding_decl=false flag to the analyzer driver.

(In cases where it is not safe to "reduce" a call—that is, to replace
a call f(x) by the body of function f, suitably substituted—the
inliner machinery is capable of replacing f by a function literal,
func(){...}(). However, the inline analyzer discards all such
"literalizations" unconditionally, again on grounds of style.)

## Constants

Given a constant that is marked for inlining, like this one:

	//go:fix inline
	const Ptr = Pointer

this analyzer will recommend that uses of Ptr should be replaced with Pointer.

As with functions, inlining can be used to replace deprecated constants and
constants in obsolete packages.

A constant definition can be marked for inlining only if it refers to another
named constant.

The "//go:fix inline" comment must appear before a single const declaration on its own,
as above; before a const declaration that is part of a group, as in this case:

	const (
	   C = 1
	   //go:fix inline
	   Ptr = Pointer
	)

or before a group, applying to every constant in the group:

	//go:fix inline
	const (
		Ptr = Pointer
		Val = Value
	)

## Type aliases

Similar to named constants, a type alias can also be marked for inlining:

	//go:fix inline
	type A = newpkg.A

The analyzer will replace all references to the annotated type
(A) by the type on the right-hand side of the declaration (newpkg.A).

## Tests

A use of a function, named constant, or type alias X from its
dedicated test (TestX), is not inlined, since the purpose of the test
is to exercise X itself, even if it is deprecated and other uses of it
should be inlined.
This applies to benchmarks and examples too, and follows the usual
conventions of test function naming.

Similarly, if the symbol X is declared in a file named foo.go, any use
of it within a file named foo_test.go will also not be inlined.

# Analyzer gofixdirective

gofixdirective: validate uses of //go:fix comment directives

The gofixdirective analyzer checks "//go:fix inline" directives for correctness.
See the documentation for the gofix analyzer for more about "/go:fix inline".
*/
package inline
