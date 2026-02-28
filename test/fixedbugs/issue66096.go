// compile

// Copyright 2024 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

package p

type Message struct {
	Header map[string][]string
}

func f() {
	m := Message{Header: map[string][]string{}}
	m.Header[""] = append([]string(m.Header[""]), "")
	_ = m
}
