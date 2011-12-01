// Copyright 2011 The Go Authors.  All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

package main

func init() {
	addTestCases(timefileinfoTests, timefileinfo)
}

var timefileinfoTests = []testCase{
	{
		Name: "timefileinfo.0",
		In: `package main

import "os"

func main() {
	st, _ := os.Stat("/etc/passwd")
	_ = st.Name
}
`,
		Out: `package main

import "os"

func main() {
	st, _ := os.Stat("/etc/passwd")
	_ = st.Name()
}
`,
	},
	{
		Name: "timefileinfo.1",
		In: `package main

import "os"

func main() {
	st, _ := os.Stat("/etc/passwd")
	_ = st.Size
	_ = st.Mode
	_ = st.Mtime_ns
	_ = st.IsDirectory()
	_ = st.IsRegular()
}
`,
		Out: `package main

import "os"

func main() {
	st, _ := os.Stat("/etc/passwd")
	_ = st.Size()
	_ = st.Mode()
	_ = st.ModTime()
	_ = st.IsDir()
	_ = !st.IsDir()
}
`,
	},
	{
		Name: "timefileinfo.2",
		In: `package main

import "os"

func f(st *os.FileInfo) {
	_ = st.Name
	_ = st.Size
	_ = st.Mode
	_ = st.Mtime_ns
	_ = st.IsDirectory()
	_ = st.IsRegular()
}
`,
		Out: `package main

import "os"

func f(st os.FileInfo) {
	_ = st.Name()
	_ = st.Size()
	_ = st.Mode()
	_ = st.ModTime()
	_ = st.IsDir()
	_ = !st.IsDir()
}
`,
	},
	{
		Name: "timefileinfo.3",
		In: `package main

import "time"

func main() {
	_ = time.Seconds()
	_ = time.Nanoseconds()
	_ = time.LocalTime()
	_ = time.UTC()
	_ = time.SecondsToLocalTime(sec)
	_ = time.SecondsToUTC(sec)
	_ = time.NanosecondsToLocalTime(nsec)
	_ = time.NanosecondsToUTC(nsec)
}
`,
		Out: `package main

import "time"

func main() {
	_ = time.Now()
	_ = time.Now()
	_ = time.Now()
	_ = time.Now().UTC()
	_ = time.Unix(sec, 0)
	_ = time.Unix(sec, 0).UTC()
	_ = time.Unix(0, nsec)
	_ = time.Unix(0, nsec).UTC()
}
`,
	},
	{
		Name: "timefileinfo.4",
		In: `package main

import "time"

func f(*time.Time)

func main() {
	t := time.LocalTime()
	_ = t.Seconds()
	_ = t.Nanoseconds()

	t1 := time.Nanoseconds()
	f(nil)
	t2 := time.Nanoseconds()
	dt := t2 - t1
}
`,
		Out: `package main

import "time"

func f(time.Time)

func main() {
	t := time.Now()
	_ = t.Unix()
	_ = t.UnixNano()

	t1 := time.Now()
	f(nil)
	t2 := time.Now()
	dt := t2.Sub(t1)
}
`,
	},
}
