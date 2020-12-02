# Analyzers

<!--TODO: Generate this file from the documentation in golang/org/x/tools/go/analysis/passes and golang/org/x/tools/go/lsp/source/options.go.-->

This document describes the analyzers that `gopls` uses inside the editor.

A value of `true` means that the analyzer is enabled by default and a value of `false` means it is disabled by default.

Details about how to enable/disable these analyses can be found [here](settings.md#analyses).

## Go vet suite

Below is the list of general analyzers that are used in `go vet`.

### **asmdecl**

report mismatches between assembly files and Go declarations

Default value: `true`.

### **assign**

check for useless assignments

This checker reports assignments of the form `x = x` or `a[i] = a[i]`.
These are almost always useless, and even when they aren't they are
usually a mistake.

Default value: `true`.

### **atomic**

check for common mistakes using the sync/atomic package

The atomic checker looks for assignment statements of the form:

`x = atomic.AddUint64(&x, 1)`

which are not atomic.

Default value: `true`.

### **atomicalign**

check for non-64-bits-aligned arguments to sync/atomic functions

Default value: `true`.

### **bools**

check for common mistakes involving boolean operators

Default value: `true`.

### **buildtag**

check that +build tags are well-formed and correctly located

Default value: `true`.

### **cgocall**

detect some violations of the cgo pointer passing rules

Check for invalid cgo pointer passing.
This looks for code that uses cgo to call C code passing values
whose types are almost always invalid according to the cgo pointer
sharing rules.
Specifically, it warns about attempts to pass a Go chan, map, func,
or slice to C, either directly, or via a pointer, array, or struct.

Default value: `true`.

### **composites**

check for unkeyed composite literals

This analyzer reports a diagnostic for composite literals of struct
types imported from another package that do not use the field-keyed
syntax. Such literals are fragile because the addition of a new field
(even if unexported) to the struct will cause compilation to fail.

As an example,
`err = &net.DNSConfigError{err}`

should be replaced by:
`err = &net.DNSConfigError{Err: err}`

Default value: `true`.

### **copylock**

check for locks erroneously passed by value

Inadvertently copying a value containing a lock, such as sync.Mutex or
sync.WaitGroup, may cause both copies to malfunction. Generally such
values should be referred to through a pointer.

Default value: `true`.

### **errorsas**

report passing non-pointer or non-error values to errors.As

The errorsas analysis reports calls to errors.As where the type
of the second argument is not a pointer to a type implementing error.

Default value: `true`.

### **httpresponse**

check for mistakes using HTTP responses

A common mistake when using the net/http package is to defer a function
call to close the http.Response Body before checking the error that
determines whether the response is valid:

```go
resp, err := http.Head(url)
defer resp.Body.Close()
if err != nil {
  log.Fatal(err)
}
// (defer statement belongs here)
```

This checker helps uncover latent nil dereference bugs by reporting a
diagnostic for such mistakes.

Default value: `true`.

### **loopclosure**

check references to loop variables from within nested functions

This analyzer checks for references to loop variables from within a
function literal inside the loop body. It checks only instances where
the function literal is called in a defer or go statement that is the
last statement in the loop body, as otherwise we would need whole
program analysis.

For example:
```go
for i, v := range s {
  go func() {
    println(i, v) // not what you might expect
  }()
}
```

See: https://golang.org/doc/go_faq.html#closures_and_goroutines

Default value: `true`.

### **lostcancel**

check cancel func returned by context.WithCancel is called

The cancellation function returned by context.WithCancel, WithTimeout,
and WithDeadline must be called or the new context will remain live
until its parent context is cancelled.
(The background context is never cancelled.)

Default value: `true`.

### **nilfunc**

check for useless comparisons between functions and nil

A useless comparison is one like f == nil as opposed to f() == nil.

Default value: `true`.

### **printf**

check consistency of Printf format strings and arguments

The check applies to known functions (for example, those in package fmt)
as well as any detected wrappers of known functions.

A function that wants to avail itself of printf checking but is not
found by this analyzer's heuristics (for example, due to use of
dynamic calls) can insert a bogus call:

```go
if false {
  _ = fmt.Sprintf(format, args...) // enable printf checking
}
```

The -funcs flag specifies a comma-separated list of names of additional
known formatting functions or methods. If the name contains a period,
it must denote a specific function using one of the following forms:

```
	dir/pkg.Function
	dir/pkg.Type.Method
	(*dir/pkg.Type).Method
```

Otherwise the name is interpreted as a case-insensitive unqualified
identifier such as "errorf". Either way, if a listed name ends in f, the
function is assumed to be Printf-like, taking a format string before the
argument list. Otherwise it is assumed to be Print-like, taking a list
of arguments with no format string.

Default value: `true`.

### **shift**

check for shifts that equal or exceed the width of the integer

Default value: `true`.

### **stdmethods**

check signature of methods of well-known interfaces

Sometimes a type may be intended to satisfy an interface but may fail to
do so because of a mistake in its method signature.
For example, the result of this WriteTo method should be (int64, error),
not error, to satisfy io.WriterTo:

```go
	type myWriterTo struct{...}
        func (myWriterTo) WriteTo(w io.Writer) error { ... }
```

This check ensures that each method whose name matches one of several
well-known interface methods from the standard library has the correct
signature for that interface.

Checked method names include:
	Format GobEncode GobDecode MarshalJSON MarshalXML
	Peek ReadByte ReadFrom ReadRune Scan Seek
	UnmarshalJSON UnreadByte UnreadRune WriteByte
	WriteTo

Default value: `true`.

### **structtag**

check that struct field tags conform to reflect.StructTag.Get

Also report certain struct tags (json, xml) used with unexported fields.

Default value: `true`.

### **tests**

check for common mistaken usages of tests and examples

The tests checker walks Test, Benchmark and Example functions checking
malformed names, wrong signatures and examples documenting non-existent
identifiers.

Please see the documentation for package testing in golang.org/pkg/testing
for the conventions that are enforced for Tests, Benchmarks, and Examples.

Default value: `true`.

### **unmarshal**

report passing non-pointer or non-interface values to unmarshal

The unmarshal analysis reports calls to functions such as json.Unmarshal
in which the argument type is not a pointer or an interface.

Default value: `true`.

### **unreachable**

check for unreachable code

The unreachable analyzer finds statements that execution can never reach
because they are preceded by an return statement, a call to panic, an
infinite loop, or similar constructs.

Default value: `true`.

### **unsafeptr**

check for invalid conversions of uintptr to unsafe.Pointer

The unsafeptr analyzer reports likely incorrect uses of unsafe.Pointer
to convert integers to pointers. A conversion from uintptr to
unsafe.Pointer is invalid if it implies that there is a uintptr-typed
word in memory that holds a pointer value, because that word will be
invisible to stack copying and to the garbage collector.

Default value: `true`.

### **unusedresult**

check for unused results of calls to some functions

Some functions like fmt.Errorf return a result and have no side effects,
so it is always a mistake to discard the result. This analyzer reports
calls to certain functions in which the result of the call is ignored.

The set of functions may be controlled using flags.

Default value: `true`.

## gopls suite

Below is the list of analyzers that are used by `gopls`.

### **deepequalerrors**

check for calls of reflect.DeepEqual on error values

The deepequalerrors checker looks for calls of the form:

```go
    reflect.DeepEqual(err1, err2)
```

where err1 and err2 are errors. Using reflect.DeepEqual to compare
errors is discouraged.

Default value: `true`.

### **fieldalignment**

This analyzer find structs that can be rearranged to take less memory, and provides
a suggested edit with the optimal order.

Default value: `false`.

### **fillreturns**

suggested fixes for "wrong number of return values (want %d, got %d)"

This checker provides suggested fixes for type errors of the
type "wrong number of return values (want %d, got %d)". For example:
```go
func m() (int, string, *bool, error) {
  return
}
```
will turn into
```go
func m() (int, string, *bool, error) {
  return 0, "", nil, nil
}
```

This functionality is similar to [goreturns](https://github.com/sqs/goreturns).

Default value: `false`.

### **nonewvars**

suggested fixes for "no new vars on left side of :="

This checker provides suggested fixes for type errors of the
type "no new vars on left side of :=". For example:
```go
z := 1
z := 2
```
will turn into
```go
z := 1
z = 2
```

Default value: `false`.

### **noresultvalues**

suggested fixes for "no result values expected"

This checker provides suggested fixes for type errors of the
type "no result values expected". For example:
```go
func z() { return nil }
```
will turn into
```go
func z() { return }
```

Default value: `true`.

### **simplifycompositelit**

check for composite literal simplifications

An array, slice, or map composite literal of the form:
```go
[]T{T{}, T{}}
```
will be simplified to:
```go
[]T{{}, {}}
```

This is one of the simplifications that "gofmt -s" applies.

Default value: `true`.

### **simplifyrange**

check for range statement simplifications

A range of the form:
```go
for x, _ = range v {...}
```
will be simplified to:
```go
for x = range v {...}
```

A range of the form:
```go
for _ = range v {...}
```
will be simplified to:
```go
for range v {...}
```

This is one of the simplifications that "gofmt -s" applies.

Default value: `true`.

### **simplifyslice**

check for slice simplifications

A slice expression of the form:
```go
s[a:len(s)]
```
will be simplified to:
```go
s[a:]
```

This is one of the simplifications that "gofmt -s" applies.

Default value: `true`.

### **sortslice**

check the argument type of sort.Slice

sort.Slice requires an argument of a slice type. Check that
the interface{} value passed to sort.Slice is actually a slice.

Default value: `true`.

### **testinggoroutine**

report calls to (*testing.T).Fatal from goroutines started by a test.

Functions that abruptly terminate a test, such as the Fatal, Fatalf, FailNow, and
Skip{,f,Now} methods of *testing.T, must be called from the test goroutine itself.
This checker detects calls to these functions that occur within a goroutine
started by the test. For example:

```go
func TestFoo(t *testing.T) {
    go func() {
        t.Fatal("oops") // error: (*T).Fatal called from non-test goroutine
    }()
}
```

Default value: `true`.

### **undeclaredname**

suggested fixes for "undeclared name: <>"

This checker provides suggested fixes for type errors of the
type `undeclared name: <>`. It will insert a new statement:
`<> := `.

Default value: `false`.

### **unusedparams**

check for unused parameters of functions

The unusedparams analyzer checks functions to see if there are
any parameters that are not being used.

To reduce false positives it ignores:
- methods
- parameters that do not have a name or are underscored
- functions in test files
- functions with empty bodies or those with just a return stmt

Default value: `false`.
