// Copyright 2011 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

package xml

import (
	"bytes"
	"errors"
	"fmt"
	"io"
	"reflect"
	"strconv"
	"strings"
	"testing"
	"time"
)

type DriveType int

const (
	HyperDrive DriveType = iota
	ImprobabilityDrive
)

type Passenger struct {
	Name   []string `xml:"name"`
	Weight float32  `xml:"weight"`
}

type Ship struct {
	XMLName struct{} `xml:"spaceship"`

	Name      string       `xml:"name,attr"`
	Pilot     string       `xml:"pilot,attr"`
	Drive     DriveType    `xml:"drive"`
	Age       uint         `xml:"age"`
	Passenger []*Passenger `xml:"passenger"`
	secret    string
}

type NamedType string

type Port struct {
	XMLName struct{} `xml:"port"`
	Type    string   `xml:"type,attr,omitempty"`
	Comment string   `xml:",comment"`
	Number  string   `xml:",chardata"`
}

type Domain struct {
	XMLName struct{} `xml:"domain"`
	Country string   `xml:",attr,omitempty"`
	Name    []byte   `xml:",chardata"`
	Comment []byte   `xml:",comment"`
}

type Book struct {
	XMLName struct{} `xml:"book"`
	Title   string   `xml:",chardata"`
}

type Event struct {
	XMLName struct{} `xml:"event"`
	Year    int      `xml:",chardata"`
}

type Movie struct {
	XMLName struct{} `xml:"movie"`
	Length  uint     `xml:",chardata"`
}

type Pi struct {
	XMLName       struct{} `xml:"pi"`
	Approximation float32  `xml:",chardata"`
}

type Universe struct {
	XMLName struct{} `xml:"universe"`
	Visible float64  `xml:",chardata"`
}

type Particle struct {
	XMLName struct{} `xml:"particle"`
	HasMass bool     `xml:",chardata"`
}

type Departure struct {
	XMLName struct{}  `xml:"departure"`
	When    time.Time `xml:",chardata"`
}

type SecretAgent struct {
	XMLName   struct{} `xml:"agent"`
	Handle    string   `xml:"handle,attr"`
	Identity  string
	Obfuscate string `xml:",innerxml"`
}

type NestedItems struct {
	XMLName struct{} `xml:"result"`
	Items   []string `xml:">item"`
	Item1   []string `xml:"Items>item1"`
}

type NestedOrder struct {
	XMLName struct{} `xml:"result"`
	Field1  string   `xml:"parent>c"`
	Field2  string   `xml:"parent>b"`
	Field3  string   `xml:"parent>a"`
}

type MixedNested struct {
	XMLName struct{} `xml:"result"`
	A       string   `xml:"parent1>a"`
	B       string   `xml:"b"`
	C       string   `xml:"parent1>parent2>c"`
	D       string   `xml:"parent1>d"`
}

type NilTest struct {
	A interface{} `xml:"parent1>parent2>a"`
	B interface{} `xml:"parent1>b"`
	C interface{} `xml:"parent1>parent2>c"`
}

type Service struct {
	XMLName struct{} `xml:"service"`
	Domain  *Domain  `xml:"host>domain"`
	Port    *Port    `xml:"host>port"`
	Extra1  interface{}
	Extra2  interface{} `xml:"host>extra2"`
}

var nilStruct *Ship

type EmbedA struct {
	EmbedC
	EmbedB EmbedB
	FieldA string
}

type EmbedB struct {
	FieldB string
	*EmbedC
}

type EmbedC struct {
	FieldA1 string `xml:"FieldA>A1"`
	FieldA2 string `xml:"FieldA>A2"`
	FieldB  string
	FieldC  string
}

type NameCasing struct {
	XMLName struct{} `xml:"casing"`
	Xy      string
	XY      string
	XyA     string `xml:"Xy,attr"`
	XYA     string `xml:"XY,attr"`
}

type NamePrecedence struct {
	XMLName     Name              `xml:"Parent"`
	FromTag     XMLNameWithoutTag `xml:"InTag"`
	FromNameVal XMLNameWithoutTag
	FromNameTag XMLNameWithTag
	InFieldName string
}

type XMLNameWithTag struct {
	XMLName Name   `xml:"InXMLNameTag"`
	Value   string `xml:",chardata"`
}

type XMLNameWithoutTag struct {
	XMLName Name
	Value   string `xml:",chardata"`
}

type NameInField struct {
	Foo Name `xml:"ns foo"`
}

type AttrTest struct {
	Int   int     `xml:",attr"`
	Named int     `xml:"int,attr"`
	Float float64 `xml:",attr"`
	Uint8 uint8   `xml:",attr"`
	Bool  bool    `xml:",attr"`
	Str   string  `xml:",attr"`
	Bytes []byte  `xml:",attr"`
}

type OmitAttrTest struct {
	Int   int     `xml:",attr,omitempty"`
	Named int     `xml:"int,attr,omitempty"`
	Float float64 `xml:",attr,omitempty"`
	Uint8 uint8   `xml:",attr,omitempty"`
	Bool  bool    `xml:",attr,omitempty"`
	Str   string  `xml:",attr,omitempty"`
	Bytes []byte  `xml:",attr,omitempty"`
}

type OmitFieldTest struct {
	Int   int           `xml:",omitempty"`
	Named int           `xml:"int,omitempty"`
	Float float64       `xml:",omitempty"`
	Uint8 uint8         `xml:",omitempty"`
	Bool  bool          `xml:",omitempty"`
	Str   string        `xml:",omitempty"`
	Bytes []byte        `xml:",omitempty"`
	Ptr   *PresenceTest `xml:",omitempty"`
}

type AnyTest struct {
	XMLName  struct{}  `xml:"a"`
	Nested   string    `xml:"nested>value"`
	AnyField AnyHolder `xml:",any"`
}

type AnyOmitTest struct {
	XMLName  struct{}   `xml:"a"`
	Nested   string     `xml:"nested>value"`
	AnyField *AnyHolder `xml:",any,omitempty"`
}

type AnySliceTest struct {
	XMLName  struct{}    `xml:"a"`
	Nested   string      `xml:"nested>value"`
	AnyField []AnyHolder `xml:",any"`
}

type AnyHolder struct {
	XMLName Name
	XML     string `xml:",innerxml"`
}

type RecurseA struct {
	A string
	B *RecurseB
}

type RecurseB struct {
	A *RecurseA
	B string
}

type PresenceTest struct {
	Exists *struct{}
}

type IgnoreTest struct {
	PublicSecret string `xml:"-"`
}

type MyBytes []byte

type Data struct {
	Bytes  []byte
	Attr   []byte `xml:",attr"`
	Custom MyBytes
}

type Plain struct {
	V interface{}
}

type MyInt int

type EmbedInt struct {
	MyInt
}

type Strings struct {
	X []string `xml:"A>B,omitempty"`
}

type PointerFieldsTest struct {
	XMLName  Name    `xml:"dummy"`
	Name     *string `xml:"name,attr"`
	Age      *uint   `xml:"age,attr"`
	Empty    *string `xml:"empty,attr"`
	Contents *string `xml:",chardata"`
}

type ChardataEmptyTest struct {
	XMLName  Name    `xml:"test"`
	Contents *string `xml:",chardata"`
}

type MyMarshalerTest struct {
}

var _ Marshaler = (*MyMarshalerTest)(nil)

func (m *MyMarshalerTest) MarshalXML(e *Encoder, start StartElement) error {
	e.EncodeToken(start)
	e.EncodeToken(CharData([]byte("hello world")))
	e.EncodeToken(EndElement{start.Name})
	return nil
}

type MyMarshalerAttrTest struct {
}

var _ MarshalerAttr = (*MyMarshalerAttrTest)(nil)

func (m *MyMarshalerAttrTest) MarshalXMLAttr(name Name) (Attr, error) {
	return Attr{name, "hello world"}, nil
}

type MarshalerStruct struct {
	Foo MyMarshalerAttrTest `xml:",attr"`
}

type InnerStruct struct {
	XMLName Name `xml:"testns outer"`
}

type OuterStruct struct {
	InnerStruct
	IntAttr int `xml:"int,attr"`
}

type OuterNamedStruct struct {
	InnerStruct
	XMLName Name `xml:"outerns test"`
	IntAttr int  `xml:"int,attr"`
}

type OuterNamedOrderedStruct struct {
	XMLName Name `xml:"outerns test"`
	InnerStruct
	IntAttr int `xml:"int,attr"`
}

type OuterOuterStruct struct {
	OuterStruct
}

func ifaceptr(x interface{}) interface{} {
	return &x
}

var (
	nameAttr     = "Sarah"
	ageAttr      = uint(12)
	contentsAttr = "lorem ipsum"
)

// Unless explicitly stated as such (or *Plain), all of the
// tests below are two-way tests. When introducing new tests,
// please try to make them two-way as well to ensure that
// marshalling and unmarshalling are as symmetrical as feasible.
var marshalTests = []struct {
	Value         interface{}
	ExpectXML     string
	MarshalOnly   bool
	UnmarshalOnly bool
}{
	// Test nil marshals to nothing
	{Value: nil, ExpectXML: ``, MarshalOnly: true},
	{Value: nilStruct, ExpectXML: ``, MarshalOnly: true},

	// Test value types
	{Value: &Plain{true}, ExpectXML: `<Plain><V>true</V></Plain>`},
	{Value: &Plain{false}, ExpectXML: `<Plain><V>false</V></Plain>`},
	{Value: &Plain{int(42)}, ExpectXML: `<Plain><V>42</V></Plain>`},
	{Value: &Plain{int8(42)}, ExpectXML: `<Plain><V>42</V></Plain>`},
	{Value: &Plain{int16(42)}, ExpectXML: `<Plain><V>42</V></Plain>`},
	{Value: &Plain{int32(42)}, ExpectXML: `<Plain><V>42</V></Plain>`},
	{Value: &Plain{uint(42)}, ExpectXML: `<Plain><V>42</V></Plain>`},
	{Value: &Plain{uint8(42)}, ExpectXML: `<Plain><V>42</V></Plain>`},
	{Value: &Plain{uint16(42)}, ExpectXML: `<Plain><V>42</V></Plain>`},
	{Value: &Plain{uint32(42)}, ExpectXML: `<Plain><V>42</V></Plain>`},
	{Value: &Plain{float32(1.25)}, ExpectXML: `<Plain><V>1.25</V></Plain>`},
	{Value: &Plain{float64(1.25)}, ExpectXML: `<Plain><V>1.25</V></Plain>`},
	{Value: &Plain{uintptr(0xFFDD)}, ExpectXML: `<Plain><V>65501</V></Plain>`},
	{Value: &Plain{"gopher"}, ExpectXML: `<Plain><V>gopher</V></Plain>`},
	{Value: &Plain{[]byte("gopher")}, ExpectXML: `<Plain><V>gopher</V></Plain>`},
	{Value: &Plain{"</>"}, ExpectXML: `<Plain><V>&lt;/&gt;</V></Plain>`},
	{Value: &Plain{[]byte("</>")}, ExpectXML: `<Plain><V>&lt;/&gt;</V></Plain>`},
	{Value: &Plain{[3]byte{'<', '/', '>'}}, ExpectXML: `<Plain><V>&lt;/&gt;</V></Plain>`},
	{Value: &Plain{NamedType("potato")}, ExpectXML: `<Plain><V>potato</V></Plain>`},
	{Value: &Plain{[]int{1, 2, 3}}, ExpectXML: `<Plain><V>1</V><V>2</V><V>3</V></Plain>`},
	{Value: &Plain{[3]int{1, 2, 3}}, ExpectXML: `<Plain><V>1</V><V>2</V><V>3</V></Plain>`},
	{Value: ifaceptr(true), MarshalOnly: true, ExpectXML: `<bool>true</bool>`},

	// Test time.
	{
		Value:     &Plain{time.Unix(1e9, 123456789).UTC()},
		ExpectXML: `<Plain><V>2001-09-09T01:46:40.123456789Z</V></Plain>`,
	},

	// A pointer to struct{} may be used to test for an element's presence.
	{
		Value:     &PresenceTest{new(struct{})},
		ExpectXML: `<PresenceTest><Exists></Exists></PresenceTest>`,
	},
	{
		Value:     &PresenceTest{},
		ExpectXML: `<PresenceTest></PresenceTest>`,
	},

	// A pointer to struct{} may be used to test for an element's presence.
	{
		Value:     &PresenceTest{new(struct{})},
		ExpectXML: `<PresenceTest><Exists></Exists></PresenceTest>`,
	},
	{
		Value:     &PresenceTest{},
		ExpectXML: `<PresenceTest></PresenceTest>`,
	},

	// A []byte field is only nil if the element was not found.
	{
		Value:         &Data{},
		ExpectXML:     `<Data></Data>`,
		UnmarshalOnly: true,
	},
	{
		Value:         &Data{Bytes: []byte{}, Custom: MyBytes{}, Attr: []byte{}},
		ExpectXML:     `<Data Attr=""><Bytes></Bytes><Custom></Custom></Data>`,
		UnmarshalOnly: true,
	},

	// Check that []byte works, including named []byte types.
	{
		Value:     &Data{Bytes: []byte("ab"), Custom: MyBytes("cd"), Attr: []byte{'v'}},
		ExpectXML: `<Data Attr="v"><Bytes>ab</Bytes><Custom>cd</Custom></Data>`,
	},

	// Test innerxml
	{
		Value: &SecretAgent{
			Handle:    "007",
			Identity:  "James Bond",
			Obfuscate: "<redacted/>",
		},
		ExpectXML:   `<agent handle="007"><Identity>James Bond</Identity><redacted/></agent>`,
		MarshalOnly: true,
	},
	{
		Value: &SecretAgent{
			Handle:    "007",
			Identity:  "James Bond",
			Obfuscate: "<Identity>James Bond</Identity><redacted/>",
		},
		ExpectXML:     `<agent handle="007"><Identity>James Bond</Identity><redacted/></agent>`,
		UnmarshalOnly: true,
	},

	// Test structs
	{Value: &Port{Type: "ssl", Number: "443"}, ExpectXML: `<port type="ssl">443</port>`},
	{Value: &Port{Number: "443"}, ExpectXML: `<port>443</port>`},
	{Value: &Port{Type: "<unix>"}, ExpectXML: `<port type="&lt;unix&gt;"></port>`},
	{Value: &Port{Number: "443", Comment: "https"}, ExpectXML: `<port><!--https-->443</port>`},
	{Value: &Port{Number: "443", Comment: "add space-"}, ExpectXML: `<port><!--add space- -->443</port>`, MarshalOnly: true},
	{Value: &Domain{Name: []byte("google.com&friends")}, ExpectXML: `<domain>google.com&amp;friends</domain>`},
	{Value: &Domain{Name: []byte("google.com"), Comment: []byte(" &friends ")}, ExpectXML: `<domain>google.com<!-- &friends --></domain>`},
	{Value: &Book{Title: "Pride & Prejudice"}, ExpectXML: `<book>Pride &amp; Prejudice</book>`},
	{Value: &Event{Year: -3114}, ExpectXML: `<event>-3114</event>`},
	{Value: &Movie{Length: 13440}, ExpectXML: `<movie>13440</movie>`},
	{Value: &Pi{Approximation: 3.14159265}, ExpectXML: `<pi>3.1415927</pi>`},
	{Value: &Universe{Visible: 9.3e13}, ExpectXML: `<universe>9.3e+13</universe>`},
	{Value: &Particle{HasMass: true}, ExpectXML: `<particle>true</particle>`},
	{Value: &Departure{When: ParseTime("2013-01-09T00:15:00-09:00")}, ExpectXML: `<departure>2013-01-09T00:15:00-09:00</departure>`},
	{Value: atomValue, ExpectXML: atomXml},
	{
		Value: &Ship{
			Name:  "Heart of Gold",
			Pilot: "Computer",
			Age:   1,
			Drive: ImprobabilityDrive,
			Passenger: []*Passenger{
				{
					Name:   []string{"Zaphod", "Beeblebrox"},
					Weight: 7.25,
				},
				{
					Name:   []string{"Trisha", "McMillen"},
					Weight: 5.5,
				},
				{
					Name:   []string{"Ford", "Prefect"},
					Weight: 7,
				},
				{
					Name:   []string{"Arthur", "Dent"},
					Weight: 6.75,
				},
			},
		},
		ExpectXML: `<spaceship name="Heart of Gold" pilot="Computer">` +
			`<drive>` + strconv.Itoa(int(ImprobabilityDrive)) + `</drive>` +
			`<age>1</age>` +
			`<passenger>` +
			`<name>Zaphod</name>` +
			`<name>Beeblebrox</name>` +
			`<weight>7.25</weight>` +
			`</passenger>` +
			`<passenger>` +
			`<name>Trisha</name>` +
			`<name>McMillen</name>` +
			`<weight>5.5</weight>` +
			`</passenger>` +
			`<passenger>` +
			`<name>Ford</name>` +
			`<name>Prefect</name>` +
			`<weight>7</weight>` +
			`</passenger>` +
			`<passenger>` +
			`<name>Arthur</name>` +
			`<name>Dent</name>` +
			`<weight>6.75</weight>` +
			`</passenger>` +
			`</spaceship>`,
	},

	// Test a>b
	{
		Value: &NestedItems{Items: nil, Item1: nil},
		ExpectXML: `<result>` +
			`<Items>` +
			`</Items>` +
			`</result>`,
	},
	{
		Value: &NestedItems{Items: []string{}, Item1: []string{}},
		ExpectXML: `<result>` +
			`<Items>` +
			`</Items>` +
			`</result>`,
		MarshalOnly: true,
	},
	{
		Value: &NestedItems{Items: nil, Item1: []string{"A"}},
		ExpectXML: `<result>` +
			`<Items>` +
			`<item1>A</item1>` +
			`</Items>` +
			`</result>`,
	},
	{
		Value: &NestedItems{Items: []string{"A", "B"}, Item1: nil},
		ExpectXML: `<result>` +
			`<Items>` +
			`<item>A</item>` +
			`<item>B</item>` +
			`</Items>` +
			`</result>`,
	},
	{
		Value: &NestedItems{Items: []string{"A", "B"}, Item1: []string{"C"}},
		ExpectXML: `<result>` +
			`<Items>` +
			`<item>A</item>` +
			`<item>B</item>` +
			`<item1>C</item1>` +
			`</Items>` +
			`</result>`,
	},
	{
		Value: &NestedOrder{Field1: "C", Field2: "B", Field3: "A"},
		ExpectXML: `<result>` +
			`<parent>` +
			`<c>C</c>` +
			`<b>B</b>` +
			`<a>A</a>` +
			`</parent>` +
			`</result>`,
	},
	{
		Value: &NilTest{A: "A", B: nil, C: "C"},
		ExpectXML: `<NilTest>` +
			`<parent1>` +
			`<parent2><a>A</a></parent2>` +
			`<parent2><c>C</c></parent2>` +
			`</parent1>` +
			`</NilTest>`,
		MarshalOnly: true, // Uses interface{}
	},
	{
		Value: &MixedNested{A: "A", B: "B", C: "C", D: "D"},
		ExpectXML: `<result>` +
			`<parent1><a>A</a></parent1>` +
			`<b>B</b>` +
			`<parent1>` +
			`<parent2><c>C</c></parent2>` +
			`<d>D</d>` +
			`</parent1>` +
			`</result>`,
	},
	{
		Value:     &Service{Port: &Port{Number: "80"}},
		ExpectXML: `<service><host><port>80</port></host></service>`,
	},
	{
		Value:     &Service{},
		ExpectXML: `<service></service>`,
	},
	{
		Value: &Service{Port: &Port{Number: "80"}, Extra1: "A", Extra2: "B"},
		ExpectXML: `<service>` +
			`<host><port>80</port></host>` +
			`<Extra1>A</Extra1>` +
			`<host><extra2>B</extra2></host>` +
			`</service>`,
		MarshalOnly: true,
	},
	{
		Value: &Service{Port: &Port{Number: "80"}, Extra2: "example"},
		ExpectXML: `<service>` +
			`<host><port>80</port></host>` +
			`<host><extra2>example</extra2></host>` +
			`</service>`,
		MarshalOnly: true,
	},

	// Test struct embedding
	{
		Value: &EmbedA{
			EmbedC: EmbedC{
				FieldA1: "", // Shadowed by A.A
				FieldA2: "", // Shadowed by A.A
				FieldB:  "A.C.B",
				FieldC:  "A.C.C",
			},
			EmbedB: EmbedB{
				FieldB: "A.B.B",
				EmbedC: &EmbedC{
					FieldA1: "A.B.C.A1",
					FieldA2: "A.B.C.A2",
					FieldB:  "", // Shadowed by A.B.B
					FieldC:  "A.B.C.C",
				},
			},
			FieldA: "A.A",
		},
		ExpectXML: `<EmbedA>` +
			`<FieldB>A.C.B</FieldB>` +
			`<FieldC>A.C.C</FieldC>` +
			`<EmbedB>` +
			`<FieldB>A.B.B</FieldB>` +
			`<FieldA>` +
			`<A1>A.B.C.A1</A1>` +
			`<A2>A.B.C.A2</A2>` +
			`</FieldA>` +
			`<FieldC>A.B.C.C</FieldC>` +
			`</EmbedB>` +
			`<FieldA>A.A</FieldA>` +
			`</EmbedA>`,
	},

	// Test that name casing matters
	{
		Value:     &NameCasing{Xy: "mixed", XY: "upper", XyA: "mixedA", XYA: "upperA"},
		ExpectXML: `<casing Xy="mixedA" XY="upperA"><Xy>mixed</Xy><XY>upper</XY></casing>`,
	},

	// Test the order in which the XML element name is chosen
	{
		Value: &NamePrecedence{
			FromTag:     XMLNameWithoutTag{Value: "A"},
			FromNameVal: XMLNameWithoutTag{XMLName: Name{Local: "InXMLName"}, Value: "B"},
			FromNameTag: XMLNameWithTag{Value: "C"},
			InFieldName: "D",
		},
		ExpectXML: `<Parent>` +
			`<InTag>A</InTag>` +
			`<InXMLName>B</InXMLName>` +
			`<InXMLNameTag>C</InXMLNameTag>` +
			`<InFieldName>D</InFieldName>` +
			`</Parent>`,
		MarshalOnly: true,
	},
	{
		Value: &NamePrecedence{
			XMLName:     Name{Local: "Parent"},
			FromTag:     XMLNameWithoutTag{XMLName: Name{Local: "InTag"}, Value: "A"},
			FromNameVal: XMLNameWithoutTag{XMLName: Name{Local: "FromNameVal"}, Value: "B"},
			FromNameTag: XMLNameWithTag{XMLName: Name{Local: "InXMLNameTag"}, Value: "C"},
			InFieldName: "D",
		},
		ExpectXML: `<Parent>` +
			`<InTag>A</InTag>` +
			`<FromNameVal>B</FromNameVal>` +
			`<InXMLNameTag>C</InXMLNameTag>` +
			`<InFieldName>D</InFieldName>` +
			`</Parent>`,
		UnmarshalOnly: true,
	},

	// xml.Name works in a plain field as well.
	{
		Value:     &NameInField{Name{Space: "ns", Local: "foo"}},
		ExpectXML: `<NameInField><foo xmlns="ns"></foo></NameInField>`,
	},
	{
		Value:         &NameInField{Name{Space: "ns", Local: "foo"}},
		ExpectXML:     `<NameInField><foo xmlns="ns"><ignore></ignore></foo></NameInField>`,
		UnmarshalOnly: true,
	},

	// Marshaling zero xml.Name uses the tag or field name.
	{
		Value:       &NameInField{},
		ExpectXML:   `<NameInField><foo xmlns="ns"></foo></NameInField>`,
		MarshalOnly: true,
	},

	// Test attributes
	{
		Value: &AttrTest{
			Int:   8,
			Named: 9,
			Float: 23.5,
			Uint8: 255,
			Bool:  true,
			Str:   "str",
			Bytes: []byte("byt"),
		},
		ExpectXML: `<AttrTest Int="8" int="9" Float="23.5" Uint8="255"` +
			` Bool="true" Str="str" Bytes="byt"></AttrTest>`,
	},
	{
		Value: &AttrTest{Bytes: []byte{}},
		ExpectXML: `<AttrTest Int="0" int="0" Float="0" Uint8="0"` +
			` Bool="false" Str="" Bytes=""></AttrTest>`,
	},
	{
		Value: &OmitAttrTest{
			Int:   8,
			Named: 9,
			Float: 23.5,
			Uint8: 255,
			Bool:  true,
			Str:   "str",
			Bytes: []byte("byt"),
		},
		ExpectXML: `<OmitAttrTest Int="8" int="9" Float="23.5" Uint8="255"` +
			` Bool="true" Str="str" Bytes="byt"></OmitAttrTest>`,
	},
	{
		Value:     &OmitAttrTest{},
		ExpectXML: `<OmitAttrTest></OmitAttrTest>`,
	},

	// pointer fields
	{
		Value:       &PointerFieldsTest{Name: &nameAttr, Age: &ageAttr, Contents: &contentsAttr},
		ExpectXML:   `<dummy name="Sarah" age="12">lorem ipsum</dummy>`,
		MarshalOnly: true,
	},

	// empty chardata pointer field
	{
		Value:       &ChardataEmptyTest{},
		ExpectXML:   `<test></test>`,
		MarshalOnly: true,
	},

	// omitempty on fields
	{
		Value: &OmitFieldTest{
			Int:   8,
			Named: 9,
			Float: 23.5,
			Uint8: 255,
			Bool:  true,
			Str:   "str",
			Bytes: []byte("byt"),
			Ptr:   &PresenceTest{},
		},
		ExpectXML: `<OmitFieldTest>` +
			`<Int>8</Int>` +
			`<int>9</int>` +
			`<Float>23.5</Float>` +
			`<Uint8>255</Uint8>` +
			`<Bool>true</Bool>` +
			`<Str>str</Str>` +
			`<Bytes>byt</Bytes>` +
			`<Ptr></Ptr>` +
			`</OmitFieldTest>`,
	},
	{
		Value:     &OmitFieldTest{},
		ExpectXML: `<OmitFieldTest></OmitFieldTest>`,
	},

	// Test ",any"
	{
		ExpectXML: `<a><nested><value>known</value></nested><other><sub>unknown</sub></other></a>`,
		Value: &AnyTest{
			Nested: "known",
			AnyField: AnyHolder{
				XMLName: Name{Local: "other"},
				XML:     "<sub>unknown</sub>",
			},
		},
	},
	{
		Value: &AnyTest{Nested: "known",
			AnyField: AnyHolder{
				XML:     "<unknown/>",
				XMLName: Name{Local: "AnyField"},
			},
		},
		ExpectXML: `<a><nested><value>known</value></nested><AnyField><unknown/></AnyField></a>`,
	},
	{
		ExpectXML: `<a><nested><value>b</value></nested></a>`,
		Value: &AnyOmitTest{
			Nested: "b",
		},
	},
	{
		ExpectXML: `<a><nested><value>b</value></nested><c><d>e</d></c><g xmlns="f"><h>i</h></g></a>`,
		Value: &AnySliceTest{
			Nested: "b",
			AnyField: []AnyHolder{
				{
					XMLName: Name{Local: "c"},
					XML:     "<d>e</d>",
				},
				{
					XMLName: Name{Space: "f", Local: "g"},
					XML:     "<h>i</h>",
				},
			},
		},
	},
	{
		ExpectXML: `<a><nested><value>b</value></nested></a>`,
		Value: &AnySliceTest{
			Nested: "b",
		},
	},

	// Test recursive types.
	{
		Value: &RecurseA{
			A: "a1",
			B: &RecurseB{
				A: &RecurseA{"a2", nil},
				B: "b1",
			},
		},
		ExpectXML: `<RecurseA><A>a1</A><B><A><A>a2</A></A><B>b1</B></B></RecurseA>`,
	},

	// Test ignoring fields via "-" tag
	{
		ExpectXML: `<IgnoreTest></IgnoreTest>`,
		Value:     &IgnoreTest{},
	},
	{
		ExpectXML:   `<IgnoreTest></IgnoreTest>`,
		Value:       &IgnoreTest{PublicSecret: "can't tell"},
		MarshalOnly: true,
	},
	{
		ExpectXML:     `<IgnoreTest><PublicSecret>ignore me</PublicSecret></IgnoreTest>`,
		Value:         &IgnoreTest{},
		UnmarshalOnly: true,
	},

	// Test escaping.
	{
		ExpectXML: `<a><nested><value>dquote: &#34;; squote: &#39;; ampersand: &amp;; less: &lt;; greater: &gt;;</value></nested><empty></empty></a>`,
		Value: &AnyTest{
			Nested:   `dquote: "; squote: '; ampersand: &; less: <; greater: >;`,
			AnyField: AnyHolder{XMLName: Name{Local: "empty"}},
		},
	},
	{
		ExpectXML: `<a><nested><value>newline: &#xA;; cr: &#xD;; tab: &#x9;;</value></nested><AnyField></AnyField></a>`,
		Value: &AnyTest{
			Nested:   "newline: \n; cr: \r; tab: \t;",
			AnyField: AnyHolder{XMLName: Name{Local: "AnyField"}},
		},
	},
	{
		ExpectXML: "<a><nested><value>1\r2\r\n3\n\r4\n5</value></nested></a>",
		Value: &AnyTest{
			Nested: "1\n2\n3\n\n4\n5",
		},
		UnmarshalOnly: true,
	},
	{
		ExpectXML: `<EmbedInt><MyInt>42</MyInt></EmbedInt>`,
		Value: &EmbedInt{
			MyInt: 42,
		},
	},
	// Test omitempty with parent chain; see golang.org/issue/4168.
	{
		ExpectXML: `<Strings><A></A></Strings>`,
		Value:     &Strings{},
	},
	// Custom marshalers.
	{
		ExpectXML: `<MyMarshalerTest>hello world</MyMarshalerTest>`,
		Value:     &MyMarshalerTest{},
	},
	{
		ExpectXML: `<MarshalerStruct Foo="hello world"></MarshalerStruct>`,
		Value:     &MarshalerStruct{},
	},
	{
		ExpectXML: `<outer xmlns="testns" int="10"></outer>`,
		Value:     &OuterStruct{IntAttr: 10},
	},
	{
		ExpectXML: `<test xmlns="outerns" int="10"></test>`,
		Value:     &OuterNamedStruct{XMLName: Name{Space: "outerns", Local: "test"}, IntAttr: 10},
	},
	{
		ExpectXML: `<test xmlns="outerns" int="10"></test>`,
		Value:     &OuterNamedOrderedStruct{XMLName: Name{Space: "outerns", Local: "test"}, IntAttr: 10},
	},
	{
		ExpectXML: `<outer xmlns="testns" int="10"></outer>`,
		Value:     &OuterOuterStruct{OuterStruct{IntAttr: 10}},
	},
}

func TestMarshal(t *testing.T) {
	for idx, test := range marshalTests {
		if test.UnmarshalOnly {
			continue
		}
		data, err := Marshal(test.Value)
		if err != nil {
			t.Errorf("#%d: Error: %s", idx, err)
			continue
		}
		if got, want := string(data), test.ExpectXML; got != want {
			if strings.Contains(want, "\n") {
				t.Errorf("#%d: marshal(%#v):\nHAVE:\n%s\nWANT:\n%s", idx, test.Value, got, want)
			} else {
				t.Errorf("#%d: marshal(%#v):\nhave %#q\nwant %#q", idx, test.Value, got, want)
			}
		}
	}
}

type AttrParent struct {
	X string `xml:"X>Y,attr"`
}

type BadAttr struct {
	Name []string `xml:"name,attr"`
}

var marshalErrorTests = []struct {
	Value interface{}
	Err   string
	Kind  reflect.Kind
}{
	{
		Value: make(chan bool),
		Err:   "xml: unsupported type: chan bool",
		Kind:  reflect.Chan,
	},
	{
		Value: map[string]string{
			"question": "What do you get when you multiply six by nine?",
			"answer":   "42",
		},
		Err:  "xml: unsupported type: map[string]string",
		Kind: reflect.Map,
	},
	{
		Value: map[*Ship]bool{nil: false},
		Err:   "xml: unsupported type: map[*xml.Ship]bool",
		Kind:  reflect.Map,
	},
	{
		Value: &Domain{Comment: []byte("f--bar")},
		Err:   `xml: comments must not contain "--"`,
	},
	// Reject parent chain with attr, never worked; see golang.org/issue/5033.
	{
		Value: &AttrParent{},
		Err:   `xml: X>Y chain not valid with attr flag`,
	},
	{
		Value: BadAttr{[]string{"X", "Y"}},
		Err:   `xml: unsupported type: []string`,
	},
}

var marshalIndentTests = []struct {
	Value     interface{}
	Prefix    string
	Indent    string
	ExpectXML string
}{
	{
		Value: &SecretAgent{
			Handle:    "007",
			Identity:  "James Bond",
			Obfuscate: "<redacted/>",
		},
		Prefix:    "",
		Indent:    "\t",
		ExpectXML: fmt.Sprintf("<agent handle=\"007\">\n\t<Identity>James Bond</Identity><redacted/>\n</agent>"),
	},
}

func TestMarshalErrors(t *testing.T) {
	for idx, test := range marshalErrorTests {
		data, err := Marshal(test.Value)
		if err == nil {
			t.Errorf("#%d: marshal(%#v) = [success] %q, want error %v", idx, test.Value, data, test.Err)
			continue
		}
		if err.Error() != test.Err {
			t.Errorf("#%d: marshal(%#v) = [error] %v, want %v", idx, test.Value, err, test.Err)
		}
		if test.Kind != reflect.Invalid {
			if kind := err.(*UnsupportedTypeError).Type.Kind(); kind != test.Kind {
				t.Errorf("#%d: marshal(%#v) = [error kind] %s, want %s", idx, test.Value, kind, test.Kind)
			}
		}
	}
}

// Do invertibility testing on the various structures that we test
func TestUnmarshal(t *testing.T) {
	for i, test := range marshalTests {
		if test.MarshalOnly {
			continue
		}
		if _, ok := test.Value.(*Plain); ok {
			continue
		}

		vt := reflect.TypeOf(test.Value)
		dest := reflect.New(vt.Elem()).Interface()
		err := Unmarshal([]byte(test.ExpectXML), dest)

		switch fix := dest.(type) {
		case *Feed:
			fix.Author.InnerXML = ""
			for i := range fix.Entry {
				fix.Entry[i].Author.InnerXML = ""
			}
		}

		if err != nil {
			t.Errorf("#%d: unexpected error: %#v", i, err)
		} else if got, want := dest, test.Value; !reflect.DeepEqual(got, want) {
			t.Errorf("#%d: unmarshal(%q):\nhave %#v\nwant %#v", i, test.ExpectXML, got, want)
		}
	}
}

func TestMarshalIndent(t *testing.T) {
	for i, test := range marshalIndentTests {
		data, err := MarshalIndent(test.Value, test.Prefix, test.Indent)
		if err != nil {
			t.Errorf("#%d: Error: %s", i, err)
			continue
		}
		if got, want := string(data), test.ExpectXML; got != want {
			t.Errorf("#%d: MarshalIndent:\nGot:%s\nWant:\n%s", i, got, want)
		}
	}
}

type limitedBytesWriter struct {
	w      io.Writer
	remain int // until writes fail
}

func (lw *limitedBytesWriter) Write(p []byte) (n int, err error) {
	if lw.remain <= 0 {
		println("error")
		return 0, errors.New("write limit hit")
	}
	if len(p) > lw.remain {
		p = p[:lw.remain]
		n, _ = lw.w.Write(p)
		lw.remain = 0
		return n, errors.New("write limit hit")
	}
	n, err = lw.w.Write(p)
	lw.remain -= n
	return n, err
}

func TestMarshalWriteErrors(t *testing.T) {
	var buf bytes.Buffer
	const writeCap = 1024
	w := &limitedBytesWriter{&buf, writeCap}
	enc := NewEncoder(w)
	var err error
	var i int
	const n = 4000
	for i = 1; i <= n; i++ {
		err = enc.Encode(&Passenger{
			Name:   []string{"Alice", "Bob"},
			Weight: 5,
		})
		if err != nil {
			break
		}
	}
	if err == nil {
		t.Error("expected an error")
	}
	if i == n {
		t.Errorf("expected to fail before the end")
	}
	if buf.Len() != writeCap {
		t.Errorf("buf.Len() = %d; want %d", buf.Len(), writeCap)
	}
}

func TestMarshalWriteIOErrors(t *testing.T) {
	enc := NewEncoder(errWriter{})

	expectErr := "unwritable"
	err := enc.Encode(&Passenger{})
	if err == nil || err.Error() != expectErr {
		t.Errorf("EscapeTest = [error] %v, want %v", err, expectErr)
	}
}

func TestMarshalFlush(t *testing.T) {
	var buf bytes.Buffer
	enc := NewEncoder(&buf)
	if err := enc.EncodeToken(CharData("hello world")); err != nil {
		t.Fatalf("enc.EncodeToken: %v", err)
	}
	if buf.Len() > 0 {
		t.Fatalf("enc.EncodeToken caused actual write: %q", buf.Bytes())
	}
	if err := enc.Flush(); err != nil {
		t.Fatalf("enc.Flush: %v", err)
	}
	if buf.String() != "hello world" {
		t.Fatalf("after enc.Flush, buf.String() = %q, want %q", buf.String(), "hello world")
	}
}

func BenchmarkMarshal(b *testing.B) {
	b.ReportAllocs()
	for i := 0; i < b.N; i++ {
		Marshal(atomValue)
	}
}

func BenchmarkUnmarshal(b *testing.B) {
	b.ReportAllocs()
	xml := []byte(atomXml)
	for i := 0; i < b.N; i++ {
		Unmarshal(xml, &Feed{})
	}
}

// golang.org/issue/6556
func TestStructPointerMarshal(t *testing.T) {
	type A struct {
		XMLName string `xml:"a"`
		B       []interface{}
	}
	type C struct {
		XMLName Name
		Value   string `xml:"value"`
	}

	a := new(A)
	a.B = append(a.B, &C{
		XMLName: Name{Local: "c"},
		Value:   "x",
	})

	b, err := Marshal(a)
	if err != nil {
		t.Fatal(err)
	}
	if x := string(b); x != "<a><c><value>x</value></c></a>" {
		t.Fatal(x)
	}
	var v A
	err = Unmarshal(b, &v)
	if err != nil {
		t.Fatal(err)
	}
}

var encodeTokenTests = []struct {
	tok  Token
	want string
	ok   bool
}{
	{StartElement{Name{"space", "local"}, nil}, "<local xmlns=\"space\">", true},
	{StartElement{Name{"space", ""}, nil}, "", false},
	{EndElement{Name{"space", ""}}, "", false},
	{CharData("foo"), "foo", true},
	{Comment("foo"), "<!--foo-->", true},
	{Comment("foo-->"), "", false},
	{ProcInst{"Target", []byte("Instruction")}, "<?Target Instruction?>", true},
	{ProcInst{"", []byte("Instruction")}, "", false},
	{ProcInst{"Target", []byte("Instruction?>")}, "", false},
	{Directive("foo"), "<!foo>", true},
	{Directive("foo>"), "", false},
}

func TestEncodeToken(t *testing.T) {
	for _, tt := range encodeTokenTests {
		var buf bytes.Buffer
		enc := NewEncoder(&buf)
		err := enc.EncodeToken(tt.tok)
		switch {
		case !tt.ok && err == nil:
			t.Errorf("enc.EncodeToken(%#v): expected error; got none", tt.tok)
		case tt.ok && err != nil:
			t.Fatalf("enc.EncodeToken: %v", err)
		case !tt.ok && err != nil:
			// expected error, got one
		}
		if err := enc.Flush(); err != nil {
			t.Fatalf("enc.EncodeToken: %v", err)
		}
		if got := buf.String(); got != tt.want {
			t.Errorf("enc.EncodeToken = %s; want: %s", got, tt.want)
		}
	}
}

func TestProcInstEncodeToken(t *testing.T) {
	var buf bytes.Buffer
	enc := NewEncoder(&buf)

	if err := enc.EncodeToken(ProcInst{"xml", []byte("Instruction")}); err != nil {
		t.Fatalf("enc.EncodeToken: expected to be able to encode xml target ProcInst as first token, %s", err)
	}

	if err := enc.EncodeToken(ProcInst{"Target", []byte("Instruction")}); err != nil {
		t.Fatalf("enc.EncodeToken: expected to be able to add non-xml target ProcInst")
	}

	if err := enc.EncodeToken(ProcInst{"xml", []byte("Instruction")}); err == nil {
		t.Fatalf("enc.EncodeToken: expected to not be allowed to encode xml target ProcInst when not first token")
	}
}

func TestDecodeEncode(t *testing.T) {
	var in, out bytes.Buffer
	in.WriteString(`<?xml version="1.0" encoding="UTF-8"?>
<?Target Instruction?>
<root>
</root>	
`)
	dec := NewDecoder(&in)
	enc := NewEncoder(&out)
	for tok, err := dec.Token(); err == nil; tok, err = dec.Token() {
		err = enc.EncodeToken(tok)
		if err != nil {
			t.Fatalf("enc.EncodeToken: Unable to encode token (%#v), %v", tok, err)
		}
	}
}
