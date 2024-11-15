// Copyright 2024 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

/*
Package weak provides weak pointers with the goal of memory efficiency.
The primary use-cases for weak pointers are for implementing caches,
canonicalization maps (like the unique package), and for tying together
the lifetimes of separate values.

## Advice

This package is intended to target niche use-cases like the unique
package, not as a general replacement for regular Go pointers, maps,
etc.
Misuse of the structures in this package will generate unexpected and
hard-to-reproduce bugs.
Using the facilities in this package to try and resolve out-of-memory
issues and/or memory leaks is very likely the wrong answer.

The structures in this package are intended to be an implementation
detail of the package they are used by (again, see the unique package).
Avoid exposing weak structures across API boundaries, since that exposes
users of your package to the subtleties of this package.
*/
package weak
