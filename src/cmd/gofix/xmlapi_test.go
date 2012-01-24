// Copyright 2012 The Go Authors.  All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

package main

func init() {
	addTestCases(xmlapiTests, xmlapi)
}

var xmlapiTests = []testCase{
	{
		Name: "xmlapi.0",
		In: `package main

import "encoding/xml"

func f() {
	xml.Marshal(a, b)
	xml.Unmarshal(a, b)

	p1 := xml.NewParser(stream)
	p1.Unmarshal(v, start)

	var p2 xml.Parser
	p2.Unmarshal(v, start)
}
`,
		Out: `package main

import "encoding/xml"

func f() {
	xml.NewEncoder(a).Encode(b)
	xml.Unmarshal(a, b)

	p1 := xml.NewDecoder(stream)
	p1.DecodeElement(v, start)

	var p2 xml.Decoder
	p2.DecodeElement(v, start)
}
`,
	},
}
