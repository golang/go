# signature-fuzzer

This directory contains utilities for fuzz testing of Go function signatures, for use in developing/testing a Go compiler.

The basic idea of the fuzzer is that it emits source code for a stand-alone Go program; this generated program is a series of pairs of functions, a "Caller" function and a "Checker" function. The signature of the Checker function is generated randomly (random number of parameters and returns, each with randomly chosen types). The "Caller" func contains invocations of the "Checker" function, each passing randomly chosen values to the params of the "Checker", then the caller verifies that expected values are returned correctly. The "Checker" function in turn has code to verify that the expected values (more details below).

There are three main parts to the fuzzer: a generator package, a driver package, and a runner package.

The "generator" contains the guts of the fuzzer, the bits that actually emit the random code.

The "driver" is a stand-alone program that invokes the generator to create a single test program. It is not terribly useful on its own (since it doesn't actually build or run the generated program), but it is handy for debugging the generator or looking at examples of the emitted code.

The "runner" is a more complete test harness; it repeatedly runs the generator to create a new test program, builds the test program, then runs it (checking for errors along the way). If at any point a build or test fails, the "runner" harness attempts a minimization process to try to narrow down the failure to a single package and/or function.

## What the generated code looks like

Generated Go functions will have an "interesting" set of signatures (mix of
arrays, scalars, structs), intended to pick out corner cases and odd bits in the
Go compiler's code that handles function calls and returns.

The first generated file is genChecker.go, which contains function that look something
like this (simplified):

```
type StructF4S0 struct {
F0 float64
F1 int16
F2 uint16
}

// 0 returns 2 params
func Test4(p0 int8, p1 StructF4S0)  {
  c0 := int8(-1)
  if p0 != c0 {
    NoteFailure(4, "parm", 0)
  }
  c1 := StructF4S0{float64(2), int16(-3), uint16(4)}
  if p1 != c1 {
    NoteFailure(4, "parm", 1)
  }
  return
}
```

Here the test generator has randomly selected 0 return values and 2 params, then randomly generated types for the params.

The generator then emits code on the calling side into the file "genCaller.go", which might look like:

```
func Caller4() {
var p0 int8
p0 = int8(-1)
var p1 StructF4S0
p1 = StructF4S0{float64(2), int16(-3), uint16(4)}
// 0 returns 2 params
Test4(p0, p1)
}
```

The generator then emits some utility functions (ex: NoteFailure) and a main routine that cycles through all of the tests.

## Trying a single run of the generator

To generate a set of source files just to see what they look like, you can build and run the test generator as follows. This creates a new directory "cabiTest" containing generated test files:

```
$ git clone https://golang.org/x/tools
$ cd tools/cmd/signature-fuzzer/fuzz-driver
$ go build .
$ ./fuzz-driver -numpkgs 3 -numfcns 5 -seed 12345 -outdir /tmp/sigfuzzTest -pkgpath foobar
$ cd /tmp/sigfuzzTest
$ find . -type f -print
./genCaller1/genCaller1.go
./genUtils/genUtils.go
./genChecker1/genChecker1.go
./genChecker0/genChecker0.go
./genCaller2/genCaller2.go
./genCaller0/genCaller0.go
./genMain.go
./go.mod
./genChecker2/genChecker2.go
$
```

You can build and run the generated files in the usual way:

```
$ cd /tmp/sigfuzzTest
$ go build .
$ ./foobar
starting main
finished 15 tests
$

```

## Example usage for the test runner

The test runner orchestrates multiple runs of the fuzzer, iteratively emitting code, building it, and testing the resulting binary. To use the runner, build and invoke it with a specific number of iterations; it will select a new random seed on each invocation. The runner will terminate as soon as it finds a failure. Example:

```
$ git clone https://golang.org/x/tools
$ cd tools/cmd/signature-fuzzer/fuzz-runner
$ go build .
$ ./fuzz-runner -numit=3
... begin iteration 0 with current seed 67104558
starting main
finished 1000 tests
... begin iteration 1 with current seed 67104659
starting main
finished 1000 tests
... begin iteration 2 with current seed 67104760
starting main
finished 1000 tests
$
```

If the runner encounters a failure, it will try to perform test-case "minimization", e.g. attempt to isolate the failure

```
$ cd tools/cmd/signature-fuzzer/fuzz-runner
$ go build .
$ ./fuzz-runner -n=10
./fuzz-runner -n=10
... begin iteration 0 with current seed 40661762
Error: fail [reflect] |20|3|1| =Checker3.Test1= return 1
error executing cmd ./fzTest: exit status 1
... starting minimization for failed directory /tmp/fuzzrun1005327337/fuzzTest
package minimization succeeded: found bad pkg 3
function minimization succeeded: found bad fcn 1
$
```

Here the runner has generates a failure, minimized it down to a single function and package, and left the resulting program in the output directory /tmp/fuzzrun1005327337/fuzzTest.

## Limitations, future work

No support yet for variadic functions.

The set of generated types is still a bit thin; it has fairly limited support for interface values, and doesn't include channels.

Todos:

- better interface value coverage

- implement testing of reflect.MakeFunc

- extend to work with generic code of various types

- extend to work in a debugging scenario (e.g. instead of just emitting code,
  emit a script of debugger commands to run the program with expected
  responses from the debugger)

- rework things so that instead of always checking all of a given parameter
  value, we sometimes skip over elements (or just check the length of a slice
  or string as opposed to looking at its value)

- consider adding runtime.GC() calls at some points in the generated code

