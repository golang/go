// Copyright 2023 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

Notes on the format of the testcase files in
cmd/compile/internal/inline/inlheur/testdata/props:

- each (compilable) file contains input Go code and expected results
  in the form of column-0 comments.

- functions or methods that begin with "T_" are targeted for testing,
  as well as "init" functions; all other functions are ignored.

- function header comments begin with a line containing
  the file name, function name, definition line, then index
  and a count of the number of funcs that share that same
  definition line (needed to support generics). Example:

	  // foo.go T_mumble 35 1 4

  Here "T_mumble" is defined at line 35, and it is func 0
  out of the 4 funcs that share that same line.

- function property expected results appear as comments in immediately
  prior to the function. For example, here we have first the function
  name ("T_feeds_if_simple"), then human-readable dump of the function
  properties, as well as the JSON for the properties object, each
  section separated by a "<>" delimiter.

	  // params.go T_feeds_if_simple 35 0 1
	  // RecvrParamFlags:
	  //   0: ParamFeedsIfOrSwitch
	  // <endpropsdump>
	  // {"Flags":0,"RecvrParamFlags":[8],"ReturnFlags":[]}
	  // callsite: params.go:34:10|0 "CallSiteOnPanicPath" 2
	  // <endcallsites>
	  // <endfuncpreamble>
	  func T_feeds_if_simple(x int) {
		if x < 100 {
			os.Exit(1)
		}
		println(x)
	}

- when the test runs, it will compile the Go source file with an
  option to dump out function properties, then compare the new dump
  for each function with the JSON appearing in the header comment for
  the function (in the example above, the JSON appears between
  "<endpropsdump>" and "<endfuncpreamble>". The material prior to the
  dump is simply there for human consumption, so that a developer can
  easily see that "RecvrParamFlags":[8] means that the first parameter
  has flag ParamFeedsIfOrSwitch.

- when making changes to the compiler (which can alter the expected
  results) or edits/additions to the go code in the testcase files,
  you can remaster the results by running

    go test -v -count=1 .

  In the trace output of this run, you'll see messages of the form

      === RUN   TestFuncProperties
       funcprops_test.go:NNN: update-expected: emitted updated file
                              testdata/props/XYZ.go.new
       funcprops_test.go:MMM: please compare the two files, then overwrite
                              testdata/props/XYZ.go with testdata/props/XYZ.go.new

  at which point you can compare the old and new files by hand, then
  overwrite the *.go file with the *.go.new file if you are happy with
  the diffs.

- note that the remastering process will strip out any existing
  column-0 (unindented) comments; if you write comments that you
  want to see preserved, use "/* */" or indent them.



