// Copyright 2019 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

// +build mips mipsle

package runtime

// Called from assembly only; declared for go vet.
func load_g()
func save_g()
