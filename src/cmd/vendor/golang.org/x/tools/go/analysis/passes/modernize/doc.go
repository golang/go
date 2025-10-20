// Copyright 2024 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

/*
Package modernize provides a suite of analyzers that suggest
simplifications to Go code, using modern language and library
features.

Each diagnostic provides a fix. Our intent is that these fixes may
be safely applied en masse without changing the behavior of your
program. In some cases the suggested fixes are imperfect and may
lead to (for example) unused imports or unused local variables,
causing build breakage. However, these problems are generally
trivial to fix. We regard any modernizer whose fix changes program
behavior to have a serious bug and will endeavor to fix it.

To apply all modernization fixes en masse, you can use the
following command:

	$ go run golang.org/x/tools/go/analysis/passes/modernize/cmd/modernize@latest -fix ./...

(Do not use "go get -tool" to add gopls as a dependency of your
module; gopls commands must be built from their release branch.)

If the tool warns of conflicting fixes, you may need to run it more
than once until it has applied all fixes cleanly. This command is
not an officially supported interface and may change in the future.

Changes produced by this tool should be reviewed as usual before
being merged. In some cases, a loop may be replaced by a simple
function call, causing comments within the loop to be discarded.
Human judgment may be required to avoid losing comments of value.

The modernize suite contains many analyzers. Diagnostics from some,
such as "any" (which replaces "interface{}" with "any" where it
is safe to do so), are particularly numerous. It may ease the burden of
code review to apply fixes in two steps, the first consisting only of
fixes from the "any" analyzer, the second consisting of all
other analyzers. This can be achieved using flags, as in this example:

	$ modernize -any=true  -fix ./...
	$ modernize -any=false -fix ./...

# Analyzer appendclipped

appendclipped: simplify append chains using slices.Concat

The appendclipped analyzer suggests replacing chains of append calls with a
single call to slices.Concat, which was added in Go 1.21. For example,
append(append(s, s1...), s2...) would be simplified to slices.Concat(s, s1, s2).

In the simple case of appending to a newly allocated slice, such as
append([]T(nil), s...), the analyzer suggests the more concise slices.Clone(s).
For byte slices, it will prefer bytes.Clone if the "bytes" package is
already imported.

This fix is only applied when the base of the append tower is a
"clipped" slice, meaning its length and capacity are equal (e.g.
x[:0:0] or []T{}). This is to avoid changing program behavior by
eliminating intended side effects on the base slice's underlying
array.

This analyzer is currently disabled by default as the
transformation does not preserve the nilness of the base slice in
all cases; see https://go.dev/issue/73557.

# Analyzer bloop

bloop: replace for-range over b.N with b.Loop

The bloop analyzer suggests replacing benchmark loops of the form
`for i := 0; i < b.N; i++` or `for range b.N` with the more modern
`for b.Loop()`, which was added in Go 1.24.

This change makes benchmark code more readable and also removes the need for
manual timer control, so any preceding calls to b.StartTimer, b.StopTimer,
or b.ResetTimer within the same function will also be removed.

Caveats: The b.Loop() method is designed to prevent the compiler from
optimizing away the benchmark loop, which can occasionally result in
slower execution due to increased allocations in some specific cases.

# Analyzer any

any: replace interface{} with any

The any analyzer suggests replacing uses of the empty interface type,
`interface{}`, with the `any` alias, which was introduced in Go 1.18.
This is a purely stylistic change that makes code more readable.

# Analyzer errorsastype

errorsastype: replace errors.As with errors.AsType[T]

This analyzer suggests fixes to simplify uses of [errors.As] of
this form:

	var myerr *MyErr
	if errors.As(err, &myerr) {
		handle(myerr)
	}

by using the less error-prone generic [errors.AsType] function,
introduced in Go 1.26:

	if myerr, ok := errors.AsType[*MyErr](err); ok {
		handle(myerr)
	}

The fix is only offered if the var declaration has the form shown and
there are no uses of myerr outside the if statement.

# Analyzer fmtappendf

fmtappendf: replace []byte(fmt.Sprintf) with fmt.Appendf

The fmtappendf analyzer suggests replacing `[]byte(fmt.Sprintf(...))` with
`fmt.Appendf(nil, ...)`. This avoids the intermediate allocation of a string
by Sprintf, making the code more efficient. The suggestion also applies to
fmt.Sprint and fmt.Sprintln.

# Analyzer forvar

forvar: remove redundant re-declaration of loop variables

The forvar analyzer removes unnecessary shadowing of loop variables.
Before Go 1.22, it was common to write `for _, x := range s { x := x ... }`
to create a fresh variable for each iteration. Go 1.22 changed the semantics
of `for` loops, making this pattern redundant. This analyzer removes the
unnecessary `x := x` statement.

This fix only applies to `range` loops.

# Analyzer mapsloop

mapsloop: replace explicit loops over maps with calls to maps package

The mapsloop analyzer replaces loops of the form

	for k, v := range x { m[k] = v }

with a single call to a function from the `maps` package, added in Go 1.23.
Depending on the context, this could be `maps.Copy`, `maps.Insert`,
`maps.Clone`, or `maps.Collect`.

The transformation to `maps.Clone` is applied conservatively, as it
preserves the nilness of the source map, which may be a subtle change in
behavior if the original code did not handle a nil map in the same way.

# Analyzer minmax

minmax: replace if/else statements with calls to min or max

The minmax analyzer simplifies conditional assignments by suggesting the use
of the built-in `min` and `max` functions, introduced in Go 1.21. For example,

	if a < b { x = a } else { x = b }

is replaced by

	x = min(a, b).

This analyzer avoids making suggestions for floating-point types,
as the behavior of `min` and `max` with NaN values can differ from
the original if/else statement.

# Analyzer newexpr

newexpr: simplify code by using go1.26's new(expr)

This analyzer finds declarations of functions of this form:

	func varOf(x int) *int { return &x }

and suggests a fix to turn them into inlinable wrappers around
go1.26's built-in new(expr) function:

	func varOf(x int) *int { return new(x) }

In addition, this analyzer suggests a fix for each call
to one of the functions before it is transformed, so that

	use(varOf(123))

is replaced by:

	use(new(123))

(Wrapper functions such as varOf are common when working with Go
serialization packages such as for JSON or protobuf, where pointers
are often used to express optionality.)

# Analyzer omitzero

omitzero: suggest replacing omitempty with omitzero for struct fields

The omitzero analyzer identifies uses of the `omitempty` JSON struct tag on
fields that are themselves structs. The `omitempty` tag has no effect on
struct-typed fields. The analyzer offers two suggestions: either remove the
tag, or replace it with `omitzero` (added in Go 1.24), which correctly
omits the field if the struct value is zero.

Replacing `omitempty` with `omitzero` is a change in behavior. The
original code would always encode the struct field, whereas the
modified code will omit it if it is a zero-value.

# Analyzer plusbuild

plusbuild: remove obsolete //+build comments

The plusbuild analyzer suggests a fix to remove obsolete build tags
of the form:

	//+build linux,amd64

in files that also contain a Go 1.18-style tag such as:

	//go:build linux && amd64

(It does not check that the old and new tags are consistent;
that is the job of the 'buildtag' analyzer in the vet suite.)

# Analyzer rangeint

rangeint: replace 3-clause for loops with for-range over integers

The rangeint analyzer suggests replacing traditional for loops such
as

	for i := 0; i < n; i++ { ... }

with the more idiomatic Go 1.22 style:

	for i := range n { ... }

This transformation is applied only if (a) the loop variable is not
modified within the loop body and (b) the loop's limit expression
is not modified within the loop, as `for range` evaluates its
operand only once.

# Analyzer reflecttypefor

reflecttypefor: replace reflect.TypeOf(x) with TypeFor[T]()

This analyzer suggests fixes to replace uses of reflect.TypeOf(x) with
reflect.TypeFor, introduced in go1.22, when the desired runtime type
is known at compile time, for example:

	reflect.TypeOf(uint32(0))        -> reflect.TypeFor[uint32]()
	reflect.TypeOf((*ast.File)(nil)) -> reflect.TypeFor[*ast.File]()

It also offers a fix to simplify the construction below, which uses
reflect.TypeOf to return the runtime type for an interface type,

	reflect.TypeOf((*io.Reader)(nil)).Elem()

to:

	reflect.TypeFor[io.Reader]()

No fix is offered in cases when the runtime type is dynamic, such as:

	var r io.Reader = ...
	reflect.TypeOf(r)

or when the operand has potential side effects.

# Analyzer slicescontains

slicescontains: replace loops with slices.Contains or slices.ContainsFunc

The slicescontains analyzer simplifies loops that check for the existence of
an element in a slice. It replaces them with calls to `slices.Contains` or
`slices.ContainsFunc`, which were added in Go 1.21.

If the expression for the target element has side effects, this
transformation will cause those effects to occur only once, not
once per tested slice element.

# Analyzer slicesdelete

slicesdelete: replace append-based slice deletion with slices.Delete

The slicesdelete analyzer suggests replacing the idiom

	s = append(s[:i], s[j:]...)

with the more explicit

	s = slices.Delete(s, i, j)

introduced in Go 1.21.

This analyzer is disabled by default. The `slices.Delete` function
zeros the elements between the new length and the old length of the
slice to prevent memory leaks, which is a subtle difference in
behavior compared to the append-based idiom; see https://go.dev/issue/73686.

# Analyzer slicessort

slicessort: replace sort.Slice with slices.Sort for basic types

The slicessort analyzer simplifies sorting slices of basic ordered
types. It replaces

	sort.Slice(s, func(i, j int) bool { return s[i] < s[j] })

with the simpler `slices.Sort(s)`, which was added in Go 1.21.

# Analyzer stditerators

stditerators: use iterators instead of Len/At-style APIs

This analyzer suggests a fix to replace each loop of the form:

	for i := 0; i < x.Len(); i++ {
		use(x.At(i))
	}

or its "for elem := range x.Len()" equivalent by a range loop over an
iterator offered by the same data type:

	for elem := range x.All() {
		use(x.At(i)
	}

where x is one of various well-known types in the standard library.

# Analyzer stringscutprefix

stringscutprefix: replace HasPrefix/TrimPrefix with CutPrefix

The stringscutprefix analyzer simplifies a common pattern where code first
checks for a prefix with `strings.HasPrefix` and then removes it with
`strings.TrimPrefix`. It replaces this two-step process with a single call
to `strings.CutPrefix`, introduced in Go 1.20. The analyzer also handles
the equivalent functions in the `bytes` package.

For example, this input:

	if strings.HasPrefix(s, prefix) {
	    use(strings.TrimPrefix(s, prefix))
	}

is fixed to:

	if after, ok := strings.CutPrefix(s, prefix); ok {
	    use(after)
	}

The analyzer also offers fixes to use CutSuffix in a similar way.
This input:

	if strings.HasSuffix(s, suffix) {
	    use(strings.TrimSuffix(s, suffix))
	}

is fixed to:

	if before, ok := strings.CutSuffix(s, suffix); ok {
	    use(before)
	}

# Analyzer stringsseq

stringsseq: replace ranging over Split/Fields with SplitSeq/FieldsSeq

The stringsseq analyzer improves the efficiency of iterating over substrings.
It replaces

	for range strings.Split(...)

with the more efficient

	for range strings.SplitSeq(...)

which was added in Go 1.24 and avoids allocating a slice for the
substrings. The analyzer also handles strings.Fields and the
equivalent functions in the bytes package.

# Analyzer stringsbuilder

stringsbuilder: replace += with strings.Builder

This analyzer replaces repeated string += string concatenation
operations with calls to Go 1.10's strings.Builder.

For example:

	var s = "["
	for x := range seq {
		s += x
		s += "."
	}
	s += "]"
	use(s)

is replaced by:

	var s strings.Builder
	s.WriteString("[")
	for x := range seq {
		s.WriteString(x)
		s.WriteString(".")
	}
	s.WriteString("]")
	use(s.String())

This avoids quadratic memory allocation and improves performance.

The analyzer requires that all references to s except the final one
are += operations. To avoid warning about trivial cases, at least one
must appear within a loop. The variable s must be a local
variable, not a global or parameter.

The sole use of the finished string must be the last reference to the
variable s. (It may appear within an intervening loop or function literal,
since even s.String() is called repeatedly, it does not allocate memory.)

# Analyzer testingcontext

testingcontext: replace context.WithCancel with t.Context in tests

The testingcontext analyzer simplifies context management in tests. It
replaces the manual creation of a cancellable context,

	ctx, cancel := context.WithCancel(context.Background())
	defer cancel()

with a single call to t.Context(), which was added in Go 1.24.

This change is only suggested if the `cancel` function is not used
for any other purpose.

# Analyzer waitgroup

waitgroup: replace wg.Add(1)/go/wg.Done() with wg.Go

The waitgroup analyzer simplifies goroutine management with `sync.WaitGroup`.
It replaces the common pattern

	wg.Add(1)
	go func() {
		defer wg.Done()
		...
	}()

with a single call to

	wg.Go(func(){ ... })

which was added in Go 1.25.
*/
package modernize
