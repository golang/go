// compiledir

// Copyright 2019 The Go Authors. All rights reserved. Use of this
// source code is governed by a BSD-style license that can be found in
// the LICENSE file.

// This directory contains a pair of packages that triggers a compiler
// crash in gccgo (problem with tracking indirectly referenced
// packages during exporting). See issue 32778 for details.

package ignored
