// Copyright 2011 The Go Authors.  All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

package main

func init() {
	addTestCases(errorTests, errorFn)
}

var errorTests = []testCase{
	{
		Name: "error.0",
		In: `package main

func error() {}

var error int
`,
		Out: `package main

func error() {}

var error int
`,
	},
	{
		Name: "error.1",
		In: `package main

import "os"

func f() os.Error {
	return os.EOF
}

func error() {}

var error int

func g() {
	error := 1
	_ = error
}

func h(os.Error) {}

func i(...os.Error) {}
`,
		Out: `package main

import "io"

func f() error {
	return io.EOF
}

func error_() {}

var error_ int

func g() {
	error := 1
	_ = error
}

func h(error) {}

func i(...error) {}
`,
	},
	{
		Name: "error.2",
		In: `package main

import "os"

func f() os.Error {
	return os.EOF
}

func g() string {
	// these all convert because f is known
	if err := f(); err != nil {
		return err.String()
	}
	if err1 := f(); err1 != nil {
		return err1.String()
	}
	if e := f(); e != nil {
		return e.String()
	}
	if x := f(); x != nil {
		return x.String()
	}

	// only the error names (err, err1, e) convert; u is not known
	if err := u(); err != nil {
		return err.String()
	}
	if err1 := u(); err1 != nil {
		return err1.String()
	}
	if e := u(); e != nil {
		return e.String()
	}
	if x := u(); x != nil {
		return x.String()
	}
	return ""
}

type T int

func (t T) String() string { return "t" }

type PT int

func (p *PT) String() string { return "pt" }

type MyError int

func (t MyError) String() string { return "myerror" }

type PMyError int

func (p *PMyError) String() string { return "pmyerror" }

func error() {}

var error int
`,
		Out: `package main

import "io"

func f() error {
	return io.EOF
}

func g() string {
	// these all convert because f is known
	if err := f(); err != nil {
		return err.Error()
	}
	if err1 := f(); err1 != nil {
		return err1.Error()
	}
	if e := f(); e != nil {
		return e.Error()
	}
	if x := f(); x != nil {
		return x.Error()
	}

	// only the error names (err, err1, e) convert; u is not known
	if err := u(); err != nil {
		return err.Error()
	}
	if err1 := u(); err1 != nil {
		return err1.Error()
	}
	if e := u(); e != nil {
		return e.Error()
	}
	if x := u(); x != nil {
		return x.String()
	}
	return ""
}

type T int

func (t T) String() string { return "t" }

type PT int

func (p *PT) String() string { return "pt" }

type MyError int

func (t MyError) Error() string { return "myerror" }

type PMyError int

func (p *PMyError) Error() string { return "pmyerror" }

func error_() {}

var error_ int
`,
	},
	{
		Name: "error.3",
		In: `package main

import "os"

func f() os.Error {
	return os.EOF
}

type PathError struct {
	Name  string
	Error os.Error
}

func (p *PathError) String() string {
	return p.Name + ": " + p.Error.String()
}

func (p *PathError) Error1() string {
	p = &PathError{Error: nil}
	return fmt.Sprint(p.Name, ": ", p.Error)
}
`,
		Out: `package main

import "io"

func f() error {
	return io.EOF
}

type PathError struct {
	Name string
	Err  error
}

func (p *PathError) Error() string {
	return p.Name + ": " + p.Err.Error()
}

func (p *PathError) Error1() string {
	p = &PathError{Err: nil}
	return fmt.Sprint(p.Name, ": ", p.Err)
}
`,
	},
}
