// compile

// Copyright 2020 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

package p

func f() {
	if len([]int{})-1 < len([]int{}) {
	}

	var st struct {
		i int
	}
	g := func() string {
		return ""
	}
	h := func(string) string {
		return g() + g()
	}
	s, i := "", 0

	st.i = len(s)
	i = len(h(s[i+0:i+1])) + len(s[len(s)+1:i+1])
	s = s[(len(s[i+1:len(s)+1])+1):len(h(""))+1] + (s[i+1 : len([]int{})+i])
	i = 1 + len([]int{len([]string{s[i+len([]int{}) : len(s)+i]})})

	var ch chan int
	ch <- len(h("")) - len(s)
}
