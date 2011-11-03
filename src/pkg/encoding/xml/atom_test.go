// Copyright 2011 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

package xml

var atomValue = &Feed{
	Title:   "Example Feed",
	Link:    []Link{{Href: "http://example.org/"}},
	Updated: ParseTime("2003-12-13T18:30:02Z"),
	Author:  Person{Name: "John Doe"},
	Id:      "urn:uuid:60a76c80-d399-11d9-b93C-0003939e0af6",

	Entry: []Entry{
		{
			Title:   "Atom-Powered Robots Run Amok",
			Link:    []Link{{Href: "http://example.org/2003/12/13/atom03"}},
			Id:      "urn:uuid:1225c695-cfb8-4ebb-aaaa-80da344efa6a",
			Updated: ParseTime("2003-12-13T18:30:02Z"),
			Summary: NewText("Some text."),
		},
	},
}

var atomXml = `` +
	`<feed xmlns="http://www.w3.org/2005/Atom">` +
	`<Title>Example Feed</Title>` +
	`<Id>urn:uuid:60a76c80-d399-11d9-b93C-0003939e0af6</Id>` +
	`<Link href="http://example.org/"></Link>` +
	`<Updated>2003-12-13T18:30:02Z</Updated>` +
	`<Author><Name>John Doe</Name><URI></URI><Email></Email></Author>` +
	`<Entry>` +
	`<Title>Atom-Powered Robots Run Amok</Title>` +
	`<Id>urn:uuid:1225c695-cfb8-4ebb-aaaa-80da344efa6a</Id>` +
	`<Link href="http://example.org/2003/12/13/atom03"></Link>` +
	`<Updated>2003-12-13T18:30:02Z</Updated>` +
	`<Author><Name></Name><URI></URI><Email></Email></Author>` +
	`<Summary>Some text.</Summary>` +
	`</Entry>` +
	`</feed>`

func ParseTime(str string) Time {
	return Time(str)
}

func NewText(text string) Text {
	return Text{
		Body: text,
	}
}
