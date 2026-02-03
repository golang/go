// Copyright 2023 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

/*
Package inline implements inlining of Go function calls.

The client provides information about the caller and callee,
including the source text, syntax tree, and type information, and
the inliner returns the modified source file for the caller, or an
error if the inlining operation is invalid (for example because the
function body refers to names that are inaccessible to the caller).

Although this interface demands more information from the client
than might seem necessary, it enables smoother integration with
existing batch and interactive tools that have their own ways of
managing the processes of reading, parsing, and type-checking
packages. In particular, this package does not assume that the
caller and callee belong to the same token.FileSet or
types.Importer realms.

There are many aspects to a function call. It is the only construct
that can simultaneously bind multiple variables of different
explicit types, with implicit assignment conversions. (Neither var
nor := declarations can do that.) It defines the scope of control
labels, of return statements, and of defer statements. Arguments
and results of function calls may be tuples even though tuples are
not first-class values in Go, and a tuple-valued call expression
may be "spread" across the argument list of a call or the operands
of a return statement. All these unique features mean that in the
general case, not everything that can be expressed by a function
call can be expressed without one.

So, in general, inlining consists of modifying a function or method
call expression f(a1, ..., an) so that the name of the function f
is replaced ("literalized") by a literal copy of the function
declaration, with free identifiers suitably modified to use the
locally appropriate identifiers or perhaps constant argument
values.

Inlining must not change the semantics of the call. Semantics
preservation is crucial for clients such as codebase maintenance
tools that automatically inline all calls to designated functions
on a large scale. Such tools must not introduce subtle behavior
changes. (Fully inlining a call is dynamically observable using
reflection over the call stack, but this exception to the rule is
explicitly allowed.)

In many cases it is possible to entirely replace ("reduce") the
call by a copy of the function's body in which parameters have been
replaced by arguments. The inliner supports a number of reduction
strategies, and we expect this set to grow. Nonetheless, sound
reduction is surprisingly tricky.

The inliner is in some ways like an optimizing compiler. A compiler
is considered correct if it doesn't change the meaning of the
program in translation from source language to target language. An
optimizing compiler exploits the particulars of the input to
generate better code, where "better" usually means more efficient.
When a case is found in which it emits suboptimal code, the
compiler is improved to recognize more cases, or more rules, and
more exceptions to rules; this process has no end. Inlining is
similar except that "better" code means tidier code. The baseline
translation (literalization) is correct, but there are endless
rules--and exceptions to rules--by which the output can be
improved.

The following section lists some of the challenges, and ways in
which they can be addressed.

  - All effects of the call argument expressions must be preserved,
    both in their number (they must not be eliminated or repeated),
    and in their order (both with respect to other arguments, and any
    effects in the callee function).

    This must be the case even if the corresponding parameters are
    never referenced, are referenced multiple times, referenced in
    a different order from the arguments, or referenced within a
    nested function that may be executed an arbitrary number of
    times.

    Currently, parameter replacement is not applied to arguments
    with effects, but with further analysis of the sequence of
    strict effects within the callee we could relax this constraint.

  - When not all parameters can be substituted by their arguments
    (e.g. due to possible effects), if the call appears in a
    statement context, the inliner may introduce a var declaration
    that declares the parameter variables (with the correct types)
    and assigns them to their corresponding argument values.
    The rest of the function body may then follow.
    For example, the call

    f(1, 2)

    to the function

    func f(x, y int32) { stmts }

    may be reduced to

    { var x, y int32 = 1, 2; stmts }.

    There are many reasons why this is not always possible. For
    example, true parameters are statically resolved in the same
    scope, and are dynamically assigned their arguments in
    parallel; but each spec in a var declaration is statically
    resolved in sequence and dynamically executed in sequence, so
    earlier parameters may shadow references in later ones.

  - Even an argument expression as simple as ptr.x may not be
    referentially transparent, because another argument may have the
    effect of changing the value of ptr.

    This constraint could be relaxed by some kind of alias or
    escape analysis that proves that ptr cannot be mutated during
    the call.

  - Although constants are referentially transparent, as a matter of
    style we do not wish to duplicate literals that are referenced
    multiple times in the body because this undoes proper factoring.
    Also, string literals may be arbitrarily large.

  - If the function body consists of statements other than just
    "return expr", in some contexts it may be syntactically
    impossible to reduce the call. Consider:

    if x := f(); cond { ... }

    Go has no equivalent to Lisp's progn or Rust's blocks,
    nor ML's let expressions (let param = arg in body);
    its closest equivalent is func(param){body}(arg).
    Reduction strategies must therefore consider the syntactic
    context of the call.

    In such situations we could work harder to extract a statement
    context for the call, by transforming it to:

    { x := f(); if cond { ... } }

  - Similarly, without the equivalent of Rust-style blocks and
    first-class tuples, there is no general way to reduce a call
    to a function such as

    func(params)(args)(results) { stmts; return expr }

    to an expression such as

    { var params = args; stmts; expr }

    or even a statement such as

    results = { var params = args; stmts; expr }

    Consequently the declaration and scope of the result variables,
    and the assignment and control-flow implications of the return
    statement, must be dealt with by cases.

  - A standalone call statement that calls a function whose body is
    "return expr" cannot be simply replaced by the body expression
    if it is not itself a call or channel receive expression; it is
    necessary to explicitly discard the result using "_ = expr".

    Similarly, if the body is a call expression, only calls to some
    built-in functions with no result (such as copy or panic) are
    permitted as statements, whereas others (such as append) return
    a result that must be used, even if just by discarding.

  - If a parameter or result variable is updated by an assignment
    within the function body, it cannot always be safely replaced
    by a variable in the caller. For example, given

    func f(a int) int { a++; return a }

    The call y = f(x) cannot be replaced by { x++; y = x } because
    this would change the value of the caller's variable x.
    Only if the caller is finished with x is this safe.

    A similar argument applies to parameter or result variables
    that escape: by eliminating a variable, inlining would change
    the identity of the variable that escapes.

  - If the function body uses 'defer' and the inlined call is not a
    tail-call, inlining may delay the deferred effects.

  - Because the scope of a control label is the entire function, a
    call cannot be reduced if the caller and callee have intersecting
    sets of control labels. (It is possible to α-rename any
    conflicting ones, but our colleagues building C++ refactoring
    tools report that, when tools must choose new identifiers, they
    generally do a poor job.)

  - Given

    func f() uint8 { return 0 }

    var x any = f()

    reducing the call to var x any = 0 is unsound because it
    discards the implicit conversion to uint8. We may need to make
    each argument-to-parameter conversion explicit if the types
    differ. Assignments to variadic parameters may need to
    explicitly construct a slice.

    An analogous problem applies to the implicit assignments in
    return statements:

    func g() any { return f() }

    Replacing the call f() with 0 would silently lose a
    conversion to uint8 and change the behavior of the program.

  - When inlining a call f(1, x, g()) where those parameters are
    unreferenced, we should be able to avoid evaluating 1 and x
    since they are pure and thus have no effect. But x may be the
    last reference to a local variable in the caller, so removing
    it would cause a compilation error. Parameter substitution must
    avoid making the caller's local variables unreferenced (or must
    be prepared to eliminate the declaration too---this is where an
    iterative framework for simplification would really help).

  - An expression such as s[i] may be valid if s and i are
    variables but invalid if either or both of them are constants.
    For example, a negative constant index s[-1] is always out of
    bounds, and even a non-negative constant index may be out of
    bounds depending on the particular string constant (e.g.
    "abc"[4]).

    So, if a parameter participates in any expression that is
    subject to additional compile-time checks when its operands are
    constant, it may be unsafe to substitute that parameter by a
    constant argument value (#62664).

More complex callee functions are inlinable with more elaborate and
invasive changes to the statements surrounding the call expression.

TODO(adonovan): future work:

  - Handle more of the above special cases by careful analysis,
    thoughtful factoring of the large design space, and thorough
    test coverage.

  - Compute precisely (not conservatively) when parameter
    substitution would remove the last reference to a caller local
    variable, and blank out the local instead of retreating from
    the substitution.

  - Afford the client more control such as a limit on the total
    increase in line count, or a refusal to inline using the
    general approach (replacing name by function literal). This
    could be achieved by returning metadata alongside the result
    and having the client conditionally discard the change.

  - Support inlining of generic functions, replacing type parameters
    by their instantiations.

  - Support inlining of calls to function literals ("closures").
    But note that the existing algorithm makes widespread assumptions
    that the callee is a package-level function or method.

  - Eliminate explicit conversions of "untyped" literals inserted
    conservatively when they are redundant. For example, the
    conversion int32(1) is redundant when this value is used only as a
    slice index; but it may be crucial if it is used in x := int32(1)
    as it changes the type of x, which may have further implications.
    The conversions may also be important to the falcon analysis.

  - Allow non-'go' build systems such as Bazel/Blaze a chance to
    decide whether an import is accessible using logic other than
    "/internal/" path segments. This could be achieved by returning
    the list of added import paths instead of a text diff.

  - Inlining a function from another module may change the
    effective version of the Go language spec that governs it. We
    should probably make the client responsible for rejecting
    attempts to inline from newer callees to older callers, since
    there's no way for this package to access module versions.

  - Use an alternative implementation of the import-organizing
    operation that doesn't require operating on a complete file
    (and reformatting). Then return the results in a higher-level
    form as a set of import additions and deletions plus a single
    diff that encloses the call expression. This interface could
    perhaps be implemented atop imports.Process by post-processing
    its result to obtain the abstract import changes and discarding
    its formatted output.
*/
package inline
