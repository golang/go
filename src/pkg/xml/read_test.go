// Copyright 2009 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

package xml

import (
	"reflect"
	"testing"
)

// Stripped down Atom feed data structures.

func TestUnmarshalFeed(t *testing.T) {
	var f Feed
	if err := Unmarshal(StringReader(atomFeedString), &f); err != nil {
		t.Fatalf("Unmarshal: %s", err)
	}
	if !reflect.DeepEqual(f, atomFeed) {
		t.Fatalf("have %#v\nwant %#v", f, atomFeed)
	}
}

// hget http://codereview.appspot.com/rss/mine/rsc
const atomFeedString = `
<?xml version="1.0" encoding="utf-8"?>
<feed xmlns="http://www.w3.org/2005/Atom" xml:lang="en-us"><title>Code Review - My issues</title><link href="http://codereview.appspot.com/" rel="alternate"></link><li-nk href="http://codereview.appspot.com/rss/mine/rsc" rel="self"></li-nk><id>http://codereview.appspot.com/</id><updated>2009-10-04T01:35:58+00:00</updated><author><name>rietveld&lt;&gt;</name></author><entry><title>rietveld: an attempt at pubsubhubbub
</title><link hre-f="http://codereview.appspot.com/126085" rel="alternate"></link><updated>2009-10-04T01:35:58+00:00</updated><author><name>email-address-removed</name></author><id>urn:md5:134d9179c41f806be79b3a5f7877d19a</id><summary type="html">
  An attempt at adding pubsubhubbub support to Rietveld.
http://code.google.com/p/pubsubhubbub
http://code.google.com/p/rietveld/issues/detail?id=155

The server side of the protocol is trivial:
  1. add a &amp;lt;link rel=&amp;quot;hub&amp;quot; href=&amp;quot;hub-server&amp;quot;&amp;gt; tag to all
     feeds that will be pubsubhubbubbed.
  2. every time one of those feeds changes, tell the hub
     with a simple POST request.

I have tested this by adding debug prints to a local hub
server and checking that the server got the right publish
requests.

I can&amp;#39;t quite get the server to work, but I think the bug
is not in my code.  I think that the server expects to be
able to grab the feed and see the feed&amp;#39;s actual URL in
the link rel=&amp;quot;self&amp;quot;, but the default value for that drops
the :port from the URL, and I cannot for the life of me
figure out how to get the Atom generator deep inside
django not to do that, or even where it is doing that,
or even what code is running to generate the Atom feed.
(I thought I knew but I added some assert False statements
and it kept running!)

Ignoring that particular problem, I would appreciate
feedback on the right way to get the two values at
the top of feeds.py marked NOTE(rsc).


</summary></entry><entry><title>rietveld: correct tab handling
</title><link href="http://codereview.appspot.com/124106" rel="alternate"></link><updated>2009-10-03T23:02:17+00:00</updated><author><name>email-address-removed</name></author><id>urn:md5:0a2a4f19bb815101f0ba2904aed7c35a</id><summary type="html">
  This fixes the buggy tab rendering that can be seen at
http://codereview.appspot.com/116075/diff/1/2

The fundamental problem was that the tab code was
not being told what column the text began in, so it
didn&amp;#39;t know where to put the tab stops.  Another problem
was that some of the code assumed that string byte
offsets were the same as column offsets, which is only
true if there are no tabs.

In the process of fixing this, I cleaned up the arguments
to Fold and ExpandTabs and renamed them Break and
_ExpandTabs so that I could be sure that I found all the
call sites.  I also wanted to verify that ExpandTabs was
not being used from outside intra_region_diff.py.


</summary></entry></feed> 	   `

type Feed struct {
	XMLName Name `xml:"http://www.w3.org/2005/Atom feed"`
	Title   string
	Id      string
	Link    []Link
	Updated Time
	Author  Person
	Entry   []Entry
}

type Entry struct {
	Title   string
	Id      string
	Link    []Link
	Updated Time
	Author  Person
	Summary Text
}

type Link struct {
	Rel  string `xml:"attr"`
	Href string `xml:"attr"`
}

type Person struct {
	Name     string
	URI      string
	Email    string
	InnerXML string `xml:"innerxml"`
}

type Text struct {
	Type string `xml:"attr"`
	Body string `xml:"chardata"`
}

type Time string

var atomFeed = Feed{
	XMLName: Name{"http://www.w3.org/2005/Atom", "feed"},
	Title:   "Code Review - My issues",
	Link: []Link{
		{Rel: "alternate", Href: "http://codereview.appspot.com/"},
		{Rel: "self", Href: "http://codereview.appspot.com/rss/mine/rsc"},
	},
	Id:      "http://codereview.appspot.com/",
	Updated: "2009-10-04T01:35:58+00:00",
	Author: Person{
		Name:     "rietveld<>",
		InnerXML: "<name>rietveld&lt;&gt;</name>",
	},
	Entry: []Entry{
		{
			Title: "rietveld: an attempt at pubsubhubbub\n",
			Link: []Link{
				{Rel: "alternate", Href: "http://codereview.appspot.com/126085"},
			},
			Updated: "2009-10-04T01:35:58+00:00",
			Author: Person{
				Name:     "email-address-removed",
				InnerXML: "<name>email-address-removed</name>",
			},
			Id: "urn:md5:134d9179c41f806be79b3a5f7877d19a",
			Summary: Text{
				Type: "html",
				Body: `
  An attempt at adding pubsubhubbub support to Rietveld.
http://code.google.com/p/pubsubhubbub
http://code.google.com/p/rietveld/issues/detail?id=155

The server side of the protocol is trivial:
  1. add a &lt;link rel=&quot;hub&quot; href=&quot;hub-server&quot;&gt; tag to all
     feeds that will be pubsubhubbubbed.
  2. every time one of those feeds changes, tell the hub
     with a simple POST request.

I have tested this by adding debug prints to a local hub
server and checking that the server got the right publish
requests.

I can&#39;t quite get the server to work, but I think the bug
is not in my code.  I think that the server expects to be
able to grab the feed and see the feed&#39;s actual URL in
the link rel=&quot;self&quot;, but the default value for that drops
the :port from the URL, and I cannot for the life of me
figure out how to get the Atom generator deep inside
django not to do that, or even where it is doing that,
or even what code is running to generate the Atom feed.
(I thought I knew but I added some assert False statements
and it kept running!)

Ignoring that particular problem, I would appreciate
feedback on the right way to get the two values at
the top of feeds.py marked NOTE(rsc).


`,
			},
		},
		{
			Title: "rietveld: correct tab handling\n",
			Link: []Link{
				{Rel: "alternate", Href: "http://codereview.appspot.com/124106"},
			},
			Updated: "2009-10-03T23:02:17+00:00",
			Author: Person{
				Name:     "email-address-removed",
				InnerXML: "<name>email-address-removed</name>",
			},
			Id: "urn:md5:0a2a4f19bb815101f0ba2904aed7c35a",
			Summary: Text{
				Type: "html",
				Body: `
  This fixes the buggy tab rendering that can be seen at
http://codereview.appspot.com/116075/diff/1/2

The fundamental problem was that the tab code was
not being told what column the text began in, so it
didn&#39;t know where to put the tab stops.  Another problem
was that some of the code assumed that string byte
offsets were the same as column offsets, which is only
true if there are no tabs.

In the process of fixing this, I cleaned up the arguments
to Fold and ExpandTabs and renamed them Break and
_ExpandTabs so that I could be sure that I found all the
call sites.  I also wanted to verify that ExpandTabs was
not being used from outside intra_region_diff.py.


`,
			},
		},
	},
}

type FieldNameTest struct {
	in, out string
}

var FieldNameTests = []FieldNameTest{
	{"Profile-Image", "profileimage"},
	{"_score", "score"},
}

func TestFieldName(t *testing.T) {
	for _, tt := range FieldNameTests {
		a := fieldName(tt.in)
		if a != tt.out {
			t.Fatalf("have %#v\nwant %#v\n\n", a, tt.out)
		}
	}
}

const pathTestString = `
<result>
    <before>1</before>
    <items>
        <item1>
            <value>A</value>
        </item1>
        <item2>
            <value>B</value>
        </item2>
        <Item1>
            <Value>C</Value>
            <Value>D</Value>
        </Item1>
    </items>
    <after>2</after>
</result>
`

type PathTestItem struct {
	Value string
}

type PathTestA struct {
	Items         []PathTestItem `xml:">item1"`
	Before, After string
}

type PathTestB struct {
	Other         []PathTestItem `xml:"items>Item1"`
	Before, After string
}

type PathTestC struct {
	Values1       []string `xml:"items>item1>value"`
	Values2       []string `xml:"items>item2>value"`
	Before, After string
}

type PathTestSet struct {
	Item1 []PathTestItem
}

type PathTestD struct {
	Other         PathTestSet `xml:"items>"`
	Before, After string
}

var pathTests = []interface{}{
	&PathTestA{Items: []PathTestItem{{"A"}, {"D"}}, Before: "1", After: "2"},
	&PathTestB{Other: []PathTestItem{{"A"}, {"D"}}, Before: "1", After: "2"},
	&PathTestC{Values1: []string{"A", "C", "D"}, Values2: []string{"B"}, Before: "1", After: "2"},
	&PathTestD{Other: PathTestSet{Item1: []PathTestItem{{"A"}, {"D"}}}, Before: "1", After: "2"},
}

func TestUnmarshalPaths(t *testing.T) {
	for _, pt := range pathTests {
		v := reflect.New(reflect.TypeOf(pt).Elem()).Interface()
		if err := Unmarshal(StringReader(pathTestString), v); err != nil {
			t.Fatalf("Unmarshal: %s", err)
		}
		if !reflect.DeepEqual(v, pt) {
			t.Fatalf("have %#v\nwant %#v", v, pt)
		}
	}
}

type BadPathTestA struct {
	First  string `xml:"items>item1"`
	Other  string `xml:"items>item2"`
	Second string `xml:"items>"`
}

type BadPathTestB struct {
	Other  string `xml:"items>item2>value"`
	First  string `xml:"items>item1"`
	Second string `xml:"items>item1>value"`
}

var badPathTests = []struct {
	v, e interface{}
}{
	{&BadPathTestA{}, &TagPathError{reflect.TypeOf(BadPathTestA{}), "First", "items>item1", "Second", "items>"}},
	{&BadPathTestB{}, &TagPathError{reflect.TypeOf(BadPathTestB{}), "First", "items>item1", "Second", "items>item1>value"}},
}

func TestUnmarshalBadPaths(t *testing.T) {
	for _, tt := range badPathTests {
		err := Unmarshal(StringReader(pathTestString), tt.v)
		if !reflect.DeepEqual(err, tt.e) {
			t.Fatalf("Unmarshal with %#v didn't fail properly: %#v", tt.v, err)
		}
	}
}

func TestUnmarshalAttrs(t *testing.T) {
	var f AttrTest
	if err := Unmarshal(StringReader(attrString), &f); err != nil {
		t.Fatalf("Unmarshal: %s", err)
	}
	if !reflect.DeepEqual(f, attrStruct) {
		t.Fatalf("have %#v\nwant %#v", f, attrStruct)
	}
}

type AttrTest struct {
	Test1 Test1
	Test2 Test2
}

type Test1 struct {
	Int   int     `xml:"attr"`
	Float float64 `xml:"attr"`
	Uint8 uint8   `xml:"attr"`
}

type Test2 struct {
	Bool bool `xml:"attr"`
}

const attrString = `
<?xml version="1.0" charset="utf-8"?>
<attrtest>
  <test1 int="8" float="23.5" uint8="255"/>
  <test2 bool="true"/>
</attrtest>
`

var attrStruct = AttrTest{
	Test1: Test1{
		Int:   8,
		Float: 23.5,
		Uint8: 255,
	},
	Test2: Test2{
		Bool: true,
	},
}

// test data for TestUnmarshalWithoutNameType

const OK = "OK"
const withoutNameTypeData = `
<?xml version="1.0" charset="utf-8"?>
<Test3 attr="OK" />`

type TestThree struct {
	XMLName bool   `xml:"Test3"` // XMLName field without an xml.Name type 
	Attr    string `xml:"attr"`
}

func TestUnmarshalWithoutNameType(t *testing.T) {
	var x TestThree
	if err := Unmarshal(StringReader(withoutNameTypeData), &x); err != nil {
		t.Fatalf("Unmarshal: %s", err)
	}
	if x.Attr != OK {
		t.Fatalf("have %v\nwant %v", x.Attr, OK)
	}
}
