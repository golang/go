// rundir

// Copyright 2017 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

// This test failed when the compiler didn't use the
// correct code to identify the type reflect.Method.
// The failing code relied on Type.String() which had
// formatting that depended on whether a package (in
// this case "reflect") was imported more than once.

package ignored
