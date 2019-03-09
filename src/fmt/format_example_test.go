// Copyright 2018 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

package fmt_test

import (
	"errors"
	"fmt"
	"path/filepath"
	"regexp"
)

func baz() error { return errors.New("baz flopped") }
func bar() error { return fmt.Errorf("bar(nameserver 139): %v", baz()) }
func foo() error { return fmt.Errorf("foo: %s", bar()) }

func Example_formatting() {
	err := foo()
	fmt.Println("Error:")
	fmt.Printf("%v\n", err)
	fmt.Println()
	fmt.Println("Detailed error:")
	fmt.Println(stripPath(fmt.Sprintf("%+v\n", err)))
	// Output:
	// Error:
	// foo: bar(nameserver 139): baz flopped
	//
	// Detailed error:
	// foo:
	//     fmt_test.foo
	//         fmt/format_example_test.go:16
	//   - bar(nameserver 139):
	//     fmt_test.bar
	//         fmt/format_example_test.go:15
	//   - baz flopped:
	//     fmt_test.baz
	//         fmt/format_example_test.go:14
}

func stripPath(s string) string {
	rePath := regexp.MustCompile(`( [^ ]*)fmt`)
	s = rePath.ReplaceAllString(s, " fmt")
	s = filepath.ToSlash(s)
	return s
}
