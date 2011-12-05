// Copyright 2011 The Go Authors.  All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

package main

func init() {
	addTestCases(strconvTests, strconvFn)
}

var strconvTests = []testCase{
	{
		Name: "strconv.0",
		In: `package main

import "strconv"

func f() {
	foo.Atob("abc")

	strconv.Atob("true")
	strconv.Btoa(false)

	strconv.Atof32("1.2")
	strconv.Atof64("1.2")
	strconv.AtofN("1.2", 64)
	strconv.Ftoa32(1.2, 'g', 17)
	strconv.Ftoa64(1.2, 'g', 17)
	strconv.FtoaN(1.2, 'g', 17, 64)

	strconv.Atoi("3")
	strconv.Atoi64("3")
	strconv.Btoi64("1234", 5)

	strconv.Atoui("3")
	strconv.Atoui64("3")
	strconv.Btoui64("1234", 5)

	strconv.Itoa(123)
	strconv.Itoa64(1234)
	strconv.Itob(123, 5)
	strconv.Itob64(1234, 5)

	strconv.Uitoa(123)
	strconv.Uitoa64(1234)
	strconv.Uitob(123, 5)
	strconv.Uitob64(1234, 5)

	strconv.Uitoa(uint(x))
	strconv.Uitoa(f(x))
}
`,
		Out: `package main

import "strconv"

func f() {
	foo.Atob("abc")

	strconv.ParseBool("true")
	strconv.FormatBool(false)

	strconv.ParseFloat("1.2", 32)
	strconv.ParseFloat("1.2", 64)
	strconv.ParseFloat("1.2", 64)
	strconv.FormatFloat(float64(1.2), 'g', 17, 32)
	strconv.FormatFloat(1.2, 'g', 17, 64)
	strconv.FormatFloat(1.2, 'g', 17, 64)

	strconv.Atoi("3")
	strconv.ParseInt("3", 10, 64)
	strconv.ParseInt("1234", 5, 64)

	strconv.ParseUint("3", 10, 0)
	strconv.ParseUint("3", 10, 64)
	strconv.ParseUint("1234", 5, 64)

	strconv.Itoa(123)
	strconv.FormatInt(1234, 10)
	strconv.FormatInt(int64(123), 5)
	strconv.FormatInt(1234, 5)

	strconv.FormatUint(uint64(123), 10)
	strconv.FormatUint(1234, 10)
	strconv.FormatUint(uint64(123), 5)
	strconv.FormatUint(1234, 5)

	strconv.FormatUint(uint64(x), 10)
	strconv.FormatUint(uint64(f(x)), 10)
}
`,
	},
}
