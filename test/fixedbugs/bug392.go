// $G $D/$F.dir/one.go && $G -ll $D/$F.dir/two.go

// Copyright 2011 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.
//
// Issue 2678
// -ll flag in command above is to force typecheck on import, needed to trigger the bug.
// fixedbugs/bug392.dir/two.go:3: cannot call non-function *one.file (type one.file)

package ignored
