# Analyzers

This document describes the analyzers that `gopls` uses inside the editor.

Details about how to enable/disable these analyses can be found
[here](settings.md#analyses).

<!-- BEGIN Analyzers: DO NOT MANUALLY EDIT THIS SECTION -->
## **asmdecl**

report mismatches between assembly files and Go declarations

**Enabled by default.**

## **assign**

check for useless assignments

This checker reports assignments of the form x = x or a[i] = a[i].
These are almost always useless, and even when they aren't they are
usually a mistake.

**Enabled by default.**

## **atomic**

check for common mistakes using the sync/atomic package

The atomic checker looks for assignment statements of the form:

	x = atomic.AddUint64(&x, 1)

which are not atomic.

**Enabled by default.**

## **atomicalign**

check for non-64-bits-aligned arguments to sync/atomic functions

**Enabled by default.**

## **bools**

check for common mistakes involving boolean operators

**Enabled by default.**

## **buildtag**

check that +build tags are well-formed and correctly located

**Enabled by default.**

## **cgocall**

detect some violations of the cgo pointer passing rules

Check for invalid cgo pointer passing.
This looks for code that uses cgo to call C code passing values
whose types are almost always invalid according to the cgo pointer
sharing rules.
Specifically, it warns about attempts to pass a Go chan, map, func,
or slice to C, either directly, or via a pointer, array, or struct.

**Enabled by default.**

## **composites**

check for unkeyed composite literals

This analyzer reports a diagnostic for composite literals of struct
types imported from another package that do not use the field-keyed
syntax. Such literals are fragile because the addition of a new field
(even if unexported) to the struct will cause compilation to fail.

As an example,

	err = &net.DNSConfigError{err}

should be replaced by:

	err = &net.DNSConfigError{Err: err}


**Enabled by default.**

## **copylocks**

check for locks erroneously passed by value

Inadvertently copying a value containing a lock, such as sync.Mutex or
sync.WaitGroup, may cause both copies to malfunction. Generally such
values should be referred to through a pointer.

**Enabled by default.**

## **deepequalerrors**

check for calls of reflect.DeepEqual on error values

The deepequalerrors checker looks for calls of the form:

    reflect.DeepEqual(err1, err2)

where err1 and err2 are errors. Using reflect.DeepEqual to compare
errors is discouraged.

**Enabled by default.**

## **errorsas**

report passing non-pointer or non-error values to errors.As

The errorsas analysis reports calls to errors.As where the type
of the second argument is not a pointer to a type implementing error.

**Enabled by default.**

## **fieldalignment**

find structs that would use less memory if their fields were sorted

This analyzer find structs that can be rearranged to use less memory, and provides
a suggested edit with the optimal order.

Note that there are two different diagnostics reported. One checks struct size,
and the other reports "pointer bytes" used. Pointer bytes is how many bytes of the
object that the garbage collector has to potentially scan for pointers, for example:

	struct { uint32; string }

have 16 pointer bytes because the garbage collector has to scan up through the string's
inner pointer.

	struct { string; *uint32 }

has 24 pointer bytes because it has to scan further through the *uint32.

	struct { string; uint32 }

has 8 because it can stop immediately after the string pointer.


**Disabled by default. Enable it by setting `"analyses": {"fieldalignment": true}`.**

## **httpresponse**

check for mistakes using HTTP responses

A common mistake when using the net/http package is to defer a function
call to close the http.Response Body before checking the error that
determines whether the response is valid:

	resp, err := http.Head(url)
	defer resp.Body.Close()
	if err != nil {
		log.Fatal(err)
	}
	// (defer statement belongs here)

This checker helps uncover latent nil dereference bugs by reporting a
diagnostic for such mistakes.

**Enabled by default.**

## **ifaceassert**

detect impossible interface-to-interface type assertions

This checker flags type assertions v.(T) and corresponding type-switch cases
in which the static type V of v is an interface that cannot possibly implement
the target interface T. This occurs when V and T contain methods with the same
name but different signatures. Example:

	var v interface {
		Read()
	}
	_ = v.(io.Reader)

The Read method in v has a different signature than the Read method in
io.Reader, so this assertion cannot succeed.


**Enabled by default.**

## **infertypeargs**

check for unnecessary type arguments in call expressions

Explicit type arguments may be omitted from call expressions if they can be
inferred from function arguments, or from other type arguments:

	func f[T any](T) {}
	
	func _() {
		f[string]("foo") // string could be inferred
	}


**Enabled by default.**

## **loopclosure**

check references to loop variables from within nested functions

This analyzer checks for references to loop variables from within a
function literal inside the loop body. It checks only instances where
the function literal is called in a defer or go statement that is the
last statement in the loop body, as otherwise we would need whole
program analysis.

For example:

	for i, v := range s {
		go func() {
			println(i, v) // not what you might expect
		}()
	}

See: https://golang.org/doc/go_faq.html#closures_and_goroutines

**Enabled by default.**

## **lostcancel**

check cancel func returned by context.WithCancel is called

The cancellation function returned by context.WithCancel, WithTimeout,
and WithDeadline must be called or the new context will remain live
until its parent context is cancelled.
(The background context is never cancelled.)

**Enabled by default.**

## **nilfunc**

check for useless comparisons between functions and nil

A useless comparison is one like f == nil as opposed to f() == nil.

**Enabled by default.**

## **nilness**

check for redundant or impossible nil comparisons

The nilness checker inspects the control-flow graph of each function in
a package and reports nil pointer dereferences, degenerate nil
pointers, and panics with nil values. A degenerate comparison is of the form
x==nil or x!=nil where x is statically known to be nil or non-nil. These are
often a mistake, especially in control flow related to errors. Panics with nil
values are checked because they are not detectable by

	if r := recover(); r != nil {

This check reports conditions such as:

	if f == nil { // impossible condition (f is a function)
	}

and:

	p := &v
	...
	if p != nil { // tautological condition
	}

and:

	if p == nil {
		print(*p) // nil dereference
	}

and:

	if p == nil {
		panic(p)
	}


**Disabled by default. Enable it by setting `"analyses": {"nilness": true}`.**

## **printf**

check consistency of Printf format strings and arguments

The check applies to known functions (for example, those in package fmt)
as well as any detected wrappers of known functions.

A function that wants to avail itself of printf checking but is not
found by this analyzer's heuristics (for example, due to use of
dynamic calls) can insert a bogus call:

	if false {
		_ = fmt.Sprintf(format, args...) // enable printf checking
	}

The -funcs flag specifies a comma-separated list of names of additional
known formatting functions or methods. If the name contains a period,
it must denote a specific function using one of the following forms:

	dir/pkg.Function
	dir/pkg.Type.Method
	(*dir/pkg.Type).Method

Otherwise the name is interpreted as a case-insensitive unqualified
identifier such as "errorf". Either way, if a listed name ends in f, the
function is assumed to be Printf-like, taking a format string before the
argument list. Otherwise it is assumed to be Print-like, taking a list
of arguments with no format string.


**Enabled by default.**

## **shadow**

check for possible unintended shadowing of variables

This analyzer check for shadowed variables.
A shadowed variable is a variable declared in an inner scope
with the same name and type as a variable in an outer scope,
and where the outer variable is mentioned after the inner one
is declared.

(This definition can be refined; the module generates too many
false positives and is not yet enabled by default.)

For example:

	func BadRead(f *os.File, buf []byte) error {
		var err error
		for {
			n, err := f.Read(buf) // shadows the function variable 'err'
			if err != nil {
				break // causes return of wrong value
			}
			foo(buf)
		}
		return err
	}


**Disabled by default. Enable it by setting `"analyses": {"shadow": true}`.**

## **shift**

check for shifts that equal or exceed the width of the integer

**Enabled by default.**

## **simplifycompositelit**

check for composite literal simplifications

An array, slice, or map composite literal of the form:
	[]T{T{}, T{}}
will be simplified to:
	[]T{{}, {}}

This is one of the simplifications that "gofmt -s" applies.

**Enabled by default.**

## **simplifyrange**

check for range statement simplifications

A range of the form:
	for x, _ = range v {...}
will be simplified to:
	for x = range v {...}

A range of the form:
	for _ = range v {...}
will be simplified to:
	for range v {...}

This is one of the simplifications that "gofmt -s" applies.

**Enabled by default.**

## **simplifyslice**

check for slice simplifications

A slice expression of the form:
	s[a:len(s)]
will be simplified to:
	s[a:]

This is one of the simplifications that "gofmt -s" applies.

**Enabled by default.**

## **sortslice**

check the argument type of sort.Slice

sort.Slice requires an argument of a slice type. Check that
the interface{} value passed to sort.Slice is actually a slice.

**Enabled by default.**

## **stdmethods**

check signature of methods of well-known interfaces

Sometimes a type may be intended to satisfy an interface but may fail to
do so because of a mistake in its method signature.
For example, the result of this WriteTo method should be (int64, error),
not error, to satisfy io.WriterTo:

	type myWriterTo struct{...}
        func (myWriterTo) WriteTo(w io.Writer) error { ... }

This check ensures that each method whose name matches one of several
well-known interface methods from the standard library has the correct
signature for that interface.

Checked method names include:
	Format GobEncode GobDecode MarshalJSON MarshalXML
	Peek ReadByte ReadFrom ReadRune Scan Seek
	UnmarshalJSON UnreadByte UnreadRune WriteByte
	WriteTo


**Enabled by default.**

## **stringintconv**

check for string(int) conversions

This checker flags conversions of the form string(x) where x is an integer
(but not byte or rune) type. Such conversions are discouraged because they
return the UTF-8 representation of the Unicode code point x, and not a decimal
string representation of x as one might expect. Furthermore, if x denotes an
invalid code point, the conversion cannot be statically rejected.

For conversions that intend on using the code point, consider replacing them
with string(rune(x)). Otherwise, strconv.Itoa and its equivalents return the
string representation of the value in the desired base.


**Enabled by default.**

## **structtag**

check that struct field tags conform to reflect.StructTag.Get

Also report certain struct tags (json, xml) used with unexported fields.

**Enabled by default.**

## **testinggoroutine**

report calls to (*testing.T).Fatal from goroutines started by a test.

Functions that abruptly terminate a test, such as the Fatal, Fatalf, FailNow, and
Skip{,f,Now} methods of *testing.T, must be called from the test goroutine itself.
This checker detects calls to these functions that occur within a goroutine
started by the test. For example:

func TestFoo(t *testing.T) {
    go func() {
        t.Fatal("oops") // error: (*T).Fatal called from non-test goroutine
    }()
}


**Enabled by default.**

## **tests**

check for common mistaken usages of tests and examples

The tests checker walks Test, Benchmark and Example functions checking
malformed names, wrong signatures and examples documenting non-existent
identifiers.

Please see the documentation for package testing in golang.org/pkg/testing
for the conventions that are enforced for Tests, Benchmarks, and Examples.

**Enabled by default.**

## **unmarshal**

report passing non-pointer or non-interface values to unmarshal

The unmarshal analysis reports calls to functions such as json.Unmarshal
in which the argument type is not a pointer or an interface.

**Enabled by default.**

## **unreachable**

check for unreachable code

The unreachable analyzer finds statements that execution can never reach
because they are preceded by an return statement, a call to panic, an
infinite loop, or similar constructs.

**Enabled by default.**

## **unsafeptr**

check for invalid conversions of uintptr to unsafe.Pointer

The unsafeptr analyzer reports likely incorrect uses of unsafe.Pointer
to convert integers to pointers. A conversion from uintptr to
unsafe.Pointer is invalid if it implies that there is a uintptr-typed
word in memory that holds a pointer value, because that word will be
invisible to stack copying and to the garbage collector.

**Enabled by default.**

## **unusedparams**

check for unused parameters of functions

The unusedparams analyzer checks functions to see if there are
any parameters that are not being used.

To reduce false positives it ignores:
- methods
- parameters that do not have a name or are underscored
- functions in test files
- functions with empty bodies or those with just a return stmt

**Disabled by default. Enable it by setting `"analyses": {"unusedparams": true}`.**

## **unusedresult**

check for unused results of calls to some functions

Some functions like fmt.Errorf return a result and have no side effects,
so it is always a mistake to discard the result. This analyzer reports
calls to certain functions in which the result of the call is ignored.

The set of functions may be controlled using flags.

**Enabled by default.**

## **unusedwrite**

checks for unused writes

The analyzer reports instances of writes to struct fields and
arrays that are never read. Specifically, when a struct object
or an array is copied, its elements are copied implicitly by
the compiler, and any element write to this copy does nothing
with the original object.

For example:

	type T struct { x int }
	func f(input []T) {
		for i, v := range input {  // v is a copy
			v.x = i  // unused write to field x
		}
	}

Another example is about non-pointer receiver:

	type T struct { x int }
	func (t T) f() {  // t is a copy
		t.x = i  // unused write to field x
	}


**Disabled by default. Enable it by setting `"analyses": {"unusedwrite": true}`.**

## **useany**

check for constraints that could be simplified to "any"

**Enabled by default.**

## **fillreturns**

suggest fixes for errors due to an incorrect number of return values

This checker provides suggested fixes for type errors of the
type "wrong number of return values (want %d, got %d)". For example:
	func m() (int, string, *bool, error) {
		return
	}
will turn into
	func m() (int, string, *bool, error) {
		return 0, "", nil, nil
	}

This functionality is similar to https://github.com/sqs/goreturns.


**Enabled by default.**

## **nonewvars**

suggested fixes for "no new vars on left side of :="

This checker provides suggested fixes for type errors of the
type "no new vars on left side of :=". For example:
	z := 1
	z := 2
will turn into
	z := 1
	z = 2


**Enabled by default.**

## **noresultvalues**

suggested fixes for "no result values expected"

This checker provides suggested fixes for type errors of the
type "no result values expected". For example:
	func z() { return nil }
will turn into
	func z() { return }


**Enabled by default.**

## **undeclaredname**

suggested fixes for "undeclared name: <>"

This checker provides suggested fixes for type errors of the
type "undeclared name: <>". It will either insert a new statement,
such as:

"<> := "

or a new function declaration, such as:

func <>(inferred parameters) {
	panic("implement me!")
}


**Enabled by default.**

## **fillstruct**

note incomplete struct initializations

This analyzer provides diagnostics for any struct literals that do not have
any fields initialized. Because the suggested fix for this analysis is
expensive to compute, callers should compute it separately, using the
SuggestedFix function below.


**Enabled by default.**

<!-- END Analyzers: DO NOT MANUALLY EDIT THIS SECTION -->
