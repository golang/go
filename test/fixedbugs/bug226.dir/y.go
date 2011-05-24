// Copyright 2009 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

package y

import "./x"

func f() {
	ok := new(x.T);
	var ok1 x.T;
	ok2 := &ok1;
	ok3 := &x.T{};
	ok4 := &x.T{Y:2};
	_ = x.T{};
	_ = x.T{Y:2};
	
	ok1.M();
	bad1 := *ok;	// ERROR "assignment.*T"
	bad2 := ok1;	// ERROR "assignment.*T"
	*ok4 = ok1;	// ERROR "assignment.*T"
	*ok4 = *ok2;	// ERROR "assignment.*T"
	ok1 = *ok4;	// ERROR "assignment.*T"
	_ = bad1;
	_ = bad2;
	_ = ok4;
	_ = ok3;
	_ = ok2;
	_ = ok1;
	_ = ok;
}
