// Copyright 2020 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

// go2go is a command for trying out generic Go code.
// It supports a small number of commands similar to cmd/go.
//
// Usage:
//
//	go2go <command> [arguments]
//
// The commands are:
//
//      build      translate and then run "go build packages"
//      run        translate and then run a list of files
//      test       translate and then run "go test packages"
//      translate  translate .go2 files into .go files for listed packages
//
// A package is expected to contain .go2 files but no .go files.
//
// Non-local imported packages will be first looked up using the GO2PATH
// environment variable, which should point to a GOPATH-like directory.
// For example, import "x" will first look for GO2PATHDIR/src/x,
// for each colon-separated component in GO2PATH. If not found in GO2PATH,
// imports will be looked up in the usual way. If an import includes
// .go2 files, they will be translated into .go files.
//
// There is a sample GO2PATH in cmd/go2go/testdata/go2path. It provides
// several packages that serve as examples of using generics, and may
// be useful in experimenting with your own generic code.
//
// Translation into standard Go requires generating Go code with mangled names.
// The mangled names will always include Odia (Oriya) digits, such as ୦ and ୮.
// Do not use Oriya digits in identifiers in your own code.
//
// Because this tool generates Go files, and because instantiated types
// and functions need to refer to the types with which they are instantiated,
// using function-local types as type arguments is not supported.
// Similarly, function-local parameterized types do not work.
// These are deficiencies of the tool, they will work as expected in
// any complete implementation.
//
// Similarly, generic function and type bodies that refer to unexported,
// non-generic, names can't be instantiated by different packages.
//
// Because this tool generates Go files, and because it generates type
// and function instantiations alongside other code in the package that
// instantiates those functions and types, and because those instantiatations
// may refer to names in packages imported by the original generic code,
// this tool will add imports as necessary to support the instantiations.
// Therefore, packages that use generic code must not use top level
// definitions whose names are the same as the names of packages imported
// by the generic code. For example, don't write, in package scope,
//
//     var strings = []string{"a", "b"}
//
// because if the generic code imports "strings", the variable name will
// conflict with the package name, even if your code doesn't import "strings".
// This is a deficiency of the tool, it will not be a deficiency in
// any complete implementation.
package main
