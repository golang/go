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

	var buf1 bytes.Buffer
	buf2 := &bytes.Buffer{}
	buf3 := bytes.NewBuffer(data)
	buf4 := bytes.NewBufferString(data)
	buf5 := bufio.NewReader(r)
	xml.Unmarshal(&buf1, v)
	xml.Unmarshal(buf2, v)
	xml.Unmarshal(buf3, v)
	xml.Unmarshal(buf4, v)
	xml.Unmarshal(buf5, v)

	f := os.Open("foo.xml")
	xml.Unmarshal(f, v)

	p1 := xml.NewParser(stream)
	p1.Unmarshal(v, start)

	var p2 *xml.Parser
	p2.Unmarshal(v, start)
}

func g(r io.Reader, f *os.File, b []byte) {
	xml.Unmarshal(r, v)
	xml.Unmarshal(f, v)
	xml.Unmarshal(b, v)
}
`,
		Out: `package main

import "encoding/xml"

func f() {
	xml.NewEncoder(a).Encode(b)
	xml.Unmarshal(a, b)

	var buf1 bytes.Buffer
	buf2 := &bytes.Buffer{}
	buf3 := bytes.NewBuffer(data)
	buf4 := bytes.NewBufferString(data)
	buf5 := bufio.NewReader(r)
	xml.NewDecoder(&buf1).Decode(v)
	xml.NewDecoder(buf2).Decode(v)
	xml.NewDecoder(buf3).Decode(v)
	xml.NewDecoder(buf4).Decode(v)
	xml.NewDecoder(buf5).Decode(v)

	f := os.Open("foo.xml")
	xml.NewDecoder(f).Decode(v)

	p1 := xml.NewDecoder(stream)
	p1.DecodeElement(v, start)

	var p2 *xml.Decoder
	p2.DecodeElement(v, start)
}

func g(r io.Reader, f *os.File, b []byte) {
	xml.NewDecoder(r).Decode(v)
	xml.NewDecoder(f).Decode(v)
	xml.Unmarshal(b, v)
}
`,
	},
}
