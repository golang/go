// Copyright 2009 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

package main

import "fmt"

func main() {
	var u64 uint64 = 1<<64 - 1;
	fmt.Printf("%d %d\n", u64, int64(u64));

	// harder stuff
	type T struct {
		a	int;
		b	string;
	}
	t := T{77, "Sunset Strip"};
	a := []int{1, 2, 3, 4};
	fmt.Printf("%v %v %v\n", u64, t, a);
	fmt.Print(u64, " ", t, " ", a, "\n");
	fmt.Println(u64, t, a);
}
