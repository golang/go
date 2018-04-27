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
	"sync"
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
	A any `xml:"parent1>parent2>a"`
	B any `xml:"parent1>b"`
	C any `xml:"parent1>parent2>c"`
}

type Service struct {
	XMLName struct{} `xml:"service"`
	Domain  *Domain  `xml:"host>domain"`
	Port    *Port    `xml:"host>port"`
	Extra1  any
	Extra2  any `xml:"host>extra2"`
}

var nilStruct *Ship

type EmbedA struct {
	EmbedC
	EmbedB EmbedB
	FieldA string
	embedD
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

type embedD struct {
	fieldD string
	FieldE string // Promoted and visible when embedD is embedded.
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

type AttrsTest struct {
	Attrs []Attr  `xml:",any,attr"`
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
	PStr  *string `xml:",attr,omitempty"`
}

type OmitFieldTest struct {
	Int   int           `xml:",omitempty"`
	Named int           `xml:"int,omitempty"`
	Float float64       `xml:",omitempty"`
	Uint8 uint8         `xml:",omitempty"`
	Bool  bool          `xml:",omitempty"`
	Str   string        `xml:",omitempty"`
	Bytes []byte        `xml:",omitempty"`
	PStr  *string       `xml:",omitempty"`
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
	V any
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

type PointerAnonFields struct {
	*MyInt
	*NamedType
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

func (m *MyMarshalerAttrTest) UnmarshalXMLAttr(attr Attr) error {
	return nil
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

type NestedAndChardata struct {
	AB       []string `xml:"A>B"`
	Chardata string   `xml:",chardata"`
}

type NestedAndComment struct {
	AB      []string `xml:"A>B"`
	Comment string   `xml:",comment"`
}

type CDataTest struct {
	Chardata string `xml:",cdata"`
}

type NestedAndCData struct {
	AB    []string `xml:"A>B"`
	CDATA string   `xml:",cdata"`
}

func ifaceptr(x any) any {
	return &x
}

func stringptr(x string) *string {
	return &x
}

type T1 struct{}
type T2 struct{}

type IndirComment struct {
	T1      T1
	Comment *string `xml:",comment"`
	T2      T2
}

type DirectComment struct {
	T1      T1
	Comment string `xml:",comment"`
	T2      T2
}

type IfaceComment struct {
	T1      T1
	Comment any `xml:",comment"`
	T2      T2
}

type IndirChardata struct {
	T1       T1
	Chardata *string `xml:",chardata"`
	T2       T2
}

type DirectChardata struct {
	T1       T1
	Chardata string `xml:",chardata"`
	T2       T2
}

type IfaceChardata struct {
	T1       T1
	Chardata any `xml:",chardata"`
	T2       T2
}

type IndirCDATA struct {
	T1    T1
	CDATA *string `xml:",cdata"`
	T2    T2
}

type DirectCDATA struct {
	T1    T1
	CDATA string `xml:",cdata"`
	T2    T2
}

type IfaceCDATA struct {
	T1    T1
	CDATA any `xml:",cdata"`
	T2    T2
}

type IndirInnerXML struct {
	T1       T1
	InnerXML *string `xml:",innerxml"`
	T2       T2
}

type DirectInnerXML struct {
	T1       T1
	InnerXML string `xml:",innerxml"`
	T2       T2
}

type IfaceInnerXML struct {
	T1       T1
	InnerXML any `xml:",innerxml"`
	T2       T2
}

type IndirElement struct {
	T1      T1
	Element *string
	T2      T2
}

type DirectElement struct {
	T1      T1
	Element string
	T2      T2
}

type IfaceElement struct {
	T1      T1
	Element any
	T2      T2
}

type IndirOmitEmpty struct {
	T1        T1
	OmitEmpty *string `xml:",omitempty"`
	T2        T2
}

type DirectOmitEmpty struct {
	T1        T1
	OmitEmpty string `xml:",omitempty"`
	T2        T2
}

type IfaceOmitEmpty struct {
	T1        T1
	OmitEmpty any `xml:",omitempty"`
	T2        T2
}

type IndirAny struct {
	T1  T1
	Any *string `xml:",any"`
	T2  T2
}

type DirectAny struct {
	T1  T1
	Any string `xml:",any"`
	T2  T2
}

type IfaceAny struct {
	T1  T1
	Any any `xml:",any"`
	T2  T2
}

type Generic[T any] struct {
	X T
}

type EPP struct {
	XMLName struct{} `xml:"urn:ietf:params:xml:ns:epp-1.0 epp"`
	Command *Command `xml:"command,omitempty"`
}

type Command struct {
	Check *Check `xml:"urn:ietf:params:xml:ns:epp-1.0 check,omitempty"`
}

type Check struct {
	DomainCheck *DomainCheck `xml:"urn:ietf:params:xml:ns:domain-1.0 domain:check,omitempty"`
}

type DomainCheck struct {
	DomainNames []string `xml:"urn:ietf:params:xml:ns:domain-1.0 domain:name,omitempty"`
}

type SecureEnvelope struct {
	XMLName struct{}       `xml:"urn:test:secure-1.0 sec:envelope"`
	Message *SecureMessage `xml:"urn:test:message-1.0 msg,omitempty"`
}

type SecureMessage struct {
	Body   string `xml:"urn:test:message-1.0 body,omitempty"`
	Signer string `xml:"urn:test:secure-1.0 sec:signer,attr,omitempty"`
}

type NamespacedNested struct {
	XMLName struct{} `xml:"urn:test:nested-1.0 nested"`
	Value   string   `xml:"urn:test:nested-1.0 nested:wrapper>nested:value"`
}

var (
	nameAttr     = "Sarah"
	ageAttr      = uint(12)
	contentsAttr = "lorem ipsum"
	empty        = ""
)

// Unless explicitly stated as such (or *Plain), all of the
// tests below are two-way tests. When introducing new tests,
// please try to make them two-way as well to ensure that
// marshaling and unmarshaling are as symmetrical as feasible.
var marshalTests = []struct {
	Value          any
	ExpectXML      string
	MarshalOnly    bool
	MarshalError   string
	UnmarshalOnly  bool
	UnmarshalError string
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
	// Marshal is not symmetric to Unmarshal for these oddities because &apos is written as &#39
	{Value: &Port{Type: "<un'ix>"}, ExpectXML: `<port type="&lt;un&apos;ix&gt;"></port>`, UnmarshalOnly: true},
	{Value: &Port{Type: "<un\"ix>"}, ExpectXML: `<port type="&lt;un&quot;ix&gt;"></port>`, UnmarshalOnly: true},
	{Value: &Port{Type: "<un&ix>"}, ExpectXML: `<port type="&lt;un&amp;ix&gt;"></port>`},
	{Value: &Port{Type: "<unix>"}, ExpectXML: `<port type="&lt;unix&gt;"></port>`, UnmarshalOnly: true},
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
	{Value: atomValue, ExpectXML: atomXML},
	{Value: &Generic[int]{1}, ExpectXML: `<Generic><X>1</X></Generic>`},
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
	{
		Value: &struct {
			XMLName struct{} `xml:"space top"`
			A       string   `xml:"x>a"`
			B       string   `xml:"x>b"`
			C       string   `xml:"space x>c"`
			C1      string   `xml:"space1 x>c"`
			D1      string   `xml:"space1 x>d"`
		}{
			A:  "a",
			B:  "b",
			C:  "c",
			C1: "c1",
			D1: "d1",
		},
		ExpectXML: `<top xmlns="space">` +
			`<x><a>a</a><b>b</b><c>c</c>` +
			`<c xmlns="space1">c1</c>` +
			`<d xmlns="space1">d1</d>` +
			`</x>` +
			`</top>`,
	},
	{
		Value: &struct {
			XMLName Name
			A       string `xml:"x>a"`
			B       string `xml:"x>b"`
			C       string `xml:"space x>c"`
			C1      string `xml:"space1 x>c"`
			D1      string `xml:"space1 x>d"`
		}{
			XMLName: Name{
				Space: "space0",
				Local: "top",
			},
			A:  "a",
			B:  "b",
			C:  "c",
			C1: "c1",
			D1: "d1",
		},
		ExpectXML: `<top xmlns="space0">` +
			`<x><a>a</a><b>b</b>` +
			`<c xmlns="space">c</c>` +
			`<c xmlns="space1">c1</c>` +
			`<d xmlns="space1">d1</d>` +
			`</x>` +
			`</top>`,
	},
	{
		Value: &struct {
			XMLName struct{} `xml:"top"`
			B       string   `xml:"space x>b"`
			B1      string   `xml:"space1 x>b"`
		}{
			B:  "b",
			B1: "b1",
		},
		ExpectXML: `<top>` +
			`<x><b xmlns="space">b</b>` +
			`<b xmlns="space1">b1</b></x>` +
			`</top>`,
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
			embedD: embedD{
				FieldE: "A.D.E",
			},
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
			`<FieldE>A.D.E</FieldE>` +
			`</EmbedA>`,
	},

	// Anonymous struct pointer field which is nil
	{
		Value:     &EmbedB{},
		ExpectXML: `<EmbedB><FieldB></FieldB></EmbedB>`,
	},

	// Other kinds of nil anonymous fields
	{
		Value:     &PointerAnonFields{},
		ExpectXML: `<PointerAnonFields></PointerAnonFields>`,
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
		Value: &AttrsTest{
			Attrs: []Attr{
				{Name: Name{Local: "Answer"}, Value: "42"},
				{Name: Name{Local: "Int"}, Value: "8"},
				{Name: Name{Local: "int"}, Value: "9"},
				{Name: Name{Local: "Float"}, Value: "23.5"},
				{Name: Name{Local: "Uint8"}, Value: "255"},
				{Name: Name{Local: "Bool"}, Value: "true"},
				{Name: Name{Local: "Str"}, Value: "str"},
				{Name: Name{Local: "Bytes"}, Value: "byt"},
			},
		},
		ExpectXML:   `<AttrsTest Answer="42" Int="8" int="9" Float="23.5" Uint8="255" Bool="true" Str="str" Bytes="byt" Int="0" int="0" Float="0" Uint8="0" Bool="false" Str="" Bytes=""></AttrsTest>`,
		MarshalOnly: true,
	},
	{
		Value: &AttrsTest{
			Attrs: []Attr{
				{Name: Name{Local: "Answer"}, Value: "42"},
			},
			Int:   8,
			Named: 9,
			Float: 23.5,
			Uint8: 255,
			Bool:  true,
			Str:   "str",
			Bytes: []byte("byt"),
		},
		ExpectXML: `<AttrsTest Answer="42" Int="8" int="9" Float="23.5" Uint8="255" Bool="true" Str="str" Bytes="byt"></AttrsTest>`,
	},
	{
		Value: &AttrsTest{
			Attrs: []Attr{
				{Name: Name{Local: "Int"}, Value: "0"},
				{Name: Name{Local: "int"}, Value: "0"},
				{Name: Name{Local: "Float"}, Value: "0"},
				{Name: Name{Local: "Uint8"}, Value: "0"},
				{Name: Name{Local: "Bool"}, Value: "false"},
				{Name: Name{Local: "Str"}},
				{Name: Name{Local: "Bytes"}},
			},
			Bytes: []byte{},
		},
		ExpectXML:   `<AttrsTest Int="0" int="0" Float="0" Uint8="0" Bool="false" Str="" Bytes="" Int="0" int="0" Float="0" Uint8="0" Bool="false" Str="" Bytes=""></AttrsTest>`,
		MarshalOnly: true,
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
			PStr:  &empty,
		},
		ExpectXML: `<OmitAttrTest Int="8" int="9" Float="23.5" Uint8="255"` +
			` Bool="true" Str="str" Bytes="byt" PStr=""></OmitAttrTest>`,
	},
	{
		Value:     &OmitAttrTest{},
		ExpectXML: `<OmitAttrTest></OmitAttrTest>`,
	},
	{
		Value:     &OmitAttrTest{Str: "gopher@golang.org"},
		ExpectXML: `<OmitAttrTest Str="gopher@golang.org"></OmitAttrTest>`,
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
			PStr:  &empty,
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
			`<PStr></PStr>` +
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
				XMLName: Name{Local: "other"}, // Overriding the field name is the purpose of the test
			},
		},
		ExpectXML: `<a><nested><value>known</value></nested><other><unknown/></other></a>`,
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
	// Test outputting CDATA-wrapped text.
	{
		ExpectXML: `<CDataTest></CDataTest>`,
		Value:     &CDataTest{},
	},
	{
		ExpectXML: `<CDataTest><![CDATA[http://example.com/tests/1?foo=1&bar=baz]]></CDataTest>`,
		Value: &CDataTest{
			Chardata: "http://example.com/tests/1?foo=1&bar=baz",
		},
	},
	{
		ExpectXML: `<CDataTest><![CDATA[Literal <![CDATA[Nested]]]]><![CDATA[>!]]></CDataTest>`,
		Value: &CDataTest{
			Chardata: "Literal <![CDATA[Nested]]>!",
		},
	},
	{
		ExpectXML: `<CDataTest><![CDATA[<![CDATA[Nested]]]]><![CDATA[> Literal!]]></CDataTest>`,
		Value: &CDataTest{
			Chardata: "<![CDATA[Nested]]> Literal!",
		},
	},
	{
		ExpectXML: `<CDataTest><![CDATA[<![CDATA[Nested]]]]><![CDATA[> Literal! <![CDATA[Nested]]]]><![CDATA[> Literal!]]></CDataTest>`,
		Value: &CDataTest{
			Chardata: "<![CDATA[Nested]]> Literal! <![CDATA[Nested]]> Literal!",
		},
	},
	{
		ExpectXML: `<CDataTest><![CDATA[<![CDATA[<![CDATA[Nested]]]]><![CDATA[>]]]]><![CDATA[>]]></CDataTest>`,
		Value: &CDataTest{
			Chardata: "<![CDATA[<![CDATA[Nested]]>]]>",
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
	{
		ExpectXML: `<NestedAndChardata><A><B></B><B></B></A>test</NestedAndChardata>`,
		Value:     &NestedAndChardata{AB: make([]string, 2), Chardata: "test"},
	},
	{
		ExpectXML: `<NestedAndComment><A><B></B><B></B></A><!--test--></NestedAndComment>`,
		Value:     &NestedAndComment{AB: make([]string, 2), Comment: "test"},
	},
	{
		ExpectXML: `<NestedAndCData><A><B></B><B></B></A><![CDATA[test]]></NestedAndCData>`,
		Value:     &NestedAndCData{AB: make([]string, 2), CDATA: "test"},
	},
	// Test pointer indirection in various kinds of fields.
	// https://golang.org/issue/19063
	{
		ExpectXML:   `<IndirComment><T1></T1><!--hi--><T2></T2></IndirComment>`,
		Value:       &IndirComment{Comment: stringptr("hi")},
		MarshalOnly: true,
	},
	{
		ExpectXML:   `<IndirComment><T1></T1><T2></T2></IndirComment>`,
		Value:       &IndirComment{Comment: stringptr("")},
		MarshalOnly: true,
	},
	{
		ExpectXML:    `<IndirComment><T1></T1><T2></T2></IndirComment>`,
		Value:        &IndirComment{Comment: nil},
		MarshalError: "xml: bad type for comment field of xml.IndirComment",
	},
	{
		ExpectXML:     `<IndirComment><T1></T1><!--hi--><T2></T2></IndirComment>`,
		Value:         &IndirComment{Comment: nil},
		UnmarshalOnly: true,
	},
	{
		ExpectXML:   `<IfaceComment><T1></T1><!--hi--><T2></T2></IfaceComment>`,
		Value:       &IfaceComment{Comment: "hi"},
		MarshalOnly: true,
	},
	{
		ExpectXML:     `<IfaceComment><T1></T1><!--hi--><T2></T2></IfaceComment>`,
		Value:         &IfaceComment{Comment: nil},
		UnmarshalOnly: true,
	},
	{
		ExpectXML:    `<IfaceComment><T1></T1><T2></T2></IfaceComment>`,
		Value:        &IfaceComment{Comment: nil},
		MarshalError: "xml: bad type for comment field of xml.IfaceComment",
	},
	{
		ExpectXML:     `<IfaceComment><T1></T1><T2></T2></IfaceComment>`,
		Value:         &IfaceComment{Comment: nil},
		UnmarshalOnly: true,
	},
	{
		ExpectXML: `<DirectComment><T1></T1><!--hi--><T2></T2></DirectComment>`,
		Value:     &DirectComment{Comment: string("hi")},
	},
	{
		ExpectXML: `<DirectComment><T1></T1><T2></T2></DirectComment>`,
		Value:     &DirectComment{Comment: string("")},
	},
	{
		ExpectXML: `<IndirChardata><T1></T1>hi<T2></T2></IndirChardata>`,
		Value:     &IndirChardata{Chardata: stringptr("hi")},
	},
	{
		ExpectXML:     `<IndirChardata><T1></T1><![CDATA[hi]]><T2></T2></IndirChardata>`,
		Value:         &IndirChardata{Chardata: stringptr("hi")},
		UnmarshalOnly: true, // marshals without CDATA
	},
	{
		ExpectXML: `<IndirChardata><T1></T1><T2></T2></IndirChardata>`,
		Value:     &IndirChardata{Chardata: stringptr("")},
	},
	{
		ExpectXML:   `<IndirChardata><T1></T1><T2></T2></IndirChardata>`,
		Value:       &IndirChardata{Chardata: nil},
		MarshalOnly: true, // unmarshal leaves Chardata=stringptr("")
	},
	{
		ExpectXML:      `<IfaceChardata><T1></T1>hi<T2></T2></IfaceChardata>`,
		Value:          &IfaceChardata{Chardata: string("hi")},
		UnmarshalError: "cannot unmarshal into interface {}",
	},
	{
		ExpectXML:      `<IfaceChardata><T1></T1><![CDATA[hi]]><T2></T2></IfaceChardata>`,
		Value:          &IfaceChardata{Chardata: string("hi")},
		UnmarshalOnly:  true, // marshals without CDATA
		UnmarshalError: "cannot unmarshal into interface {}",
	},
	{
		ExpectXML:      `<IfaceChardata><T1></T1><T2></T2></IfaceChardata>`,
		Value:          &IfaceChardata{Chardata: string("")},
		UnmarshalError: "cannot unmarshal into interface {}",
	},
	{
		ExpectXML:      `<IfaceChardata><T1></T1><T2></T2></IfaceChardata>`,
		Value:          &IfaceChardata{Chardata: nil},
		UnmarshalError: "cannot unmarshal into interface {}",
	},
	{
		ExpectXML: `<DirectChardata><T1></T1>hi<T2></T2></DirectChardata>`,
		Value:     &DirectChardata{Chardata: string("hi")},
	},
	{
		ExpectXML:     `<DirectChardata><T1></T1><![CDATA[hi]]><T2></T2></DirectChardata>`,
		Value:         &DirectChardata{Chardata: string("hi")},
		UnmarshalOnly: true, // marshals without CDATA
	},
	{
		ExpectXML: `<DirectChardata><T1></T1><T2></T2></DirectChardata>`,
		Value:     &DirectChardata{Chardata: string("")},
	},
	{
		ExpectXML: `<IndirCDATA><T1></T1><![CDATA[hi]]><T2></T2></IndirCDATA>`,
		Value:     &IndirCDATA{CDATA: stringptr("hi")},
	},
	{
		ExpectXML:     `<IndirCDATA><T1></T1>hi<T2></T2></IndirCDATA>`,
		Value:         &IndirCDATA{CDATA: stringptr("hi")},
		UnmarshalOnly: true, // marshals with CDATA
	},
	{
		ExpectXML: `<IndirCDATA><T1></T1><T2></T2></IndirCDATA>`,
		Value:     &IndirCDATA{CDATA: stringptr("")},
	},
	{
		ExpectXML:   `<IndirCDATA><T1></T1><T2></T2></IndirCDATA>`,
		Value:       &IndirCDATA{CDATA: nil},
		MarshalOnly: true, // unmarshal leaves CDATA=stringptr("")
	},
	{
		ExpectXML:      `<IfaceCDATA><T1></T1><![CDATA[hi]]><T2></T2></IfaceCDATA>`,
		Value:          &IfaceCDATA{CDATA: string("hi")},
		UnmarshalError: "cannot unmarshal into interface {}",
	},
	{
		ExpectXML:      `<IfaceCDATA><T1></T1>hi<T2></T2></IfaceCDATA>`,
		Value:          &IfaceCDATA{CDATA: string("hi")},
		UnmarshalOnly:  true, // marshals with CDATA
		UnmarshalError: "cannot unmarshal into interface {}",
	},
	{
		ExpectXML:      `<IfaceCDATA><T1></T1><T2></T2></IfaceCDATA>`,
		Value:          &IfaceCDATA{CDATA: string("")},
		UnmarshalError: "cannot unmarshal into interface {}",
	},
	{
		ExpectXML:      `<IfaceCDATA><T1></T1><T2></T2></IfaceCDATA>`,
		Value:          &IfaceCDATA{CDATA: nil},
		UnmarshalError: "cannot unmarshal into interface {}",
	},
	{
		ExpectXML: `<DirectCDATA><T1></T1><![CDATA[hi]]><T2></T2></DirectCDATA>`,
		Value:     &DirectCDATA{CDATA: string("hi")},
	},
	{
		ExpectXML:     `<DirectCDATA><T1></T1>hi<T2></T2></DirectCDATA>`,
		Value:         &DirectCDATA{CDATA: string("hi")},
		UnmarshalOnly: true, // marshals with CDATA
	},
	{
		ExpectXML: `<DirectCDATA><T1></T1><T2></T2></DirectCDATA>`,
		Value:     &DirectCDATA{CDATA: string("")},
	},
	{
		ExpectXML:   `<IndirInnerXML><T1></T1><hi/><T2></T2></IndirInnerXML>`,
		Value:       &IndirInnerXML{InnerXML: stringptr("<hi/>")},
		MarshalOnly: true,
	},
	{
		ExpectXML:   `<IndirInnerXML><T1></T1><T2></T2></IndirInnerXML>`,
		Value:       &IndirInnerXML{InnerXML: stringptr("")},
		MarshalOnly: true,
	},
	{
		ExpectXML: `<IndirInnerXML><T1></T1><T2></T2></IndirInnerXML>`,
		Value:     &IndirInnerXML{InnerXML: nil},
	},
	{
		ExpectXML:     `<IndirInnerXML><T1></T1><hi/><T2></T2></IndirInnerXML>`,
		Value:         &IndirInnerXML{InnerXML: nil},
		UnmarshalOnly: true,
	},
	{
		ExpectXML:   `<IfaceInnerXML><T1></T1><hi/><T2></T2></IfaceInnerXML>`,
		Value:       &IfaceInnerXML{InnerXML: "<hi/>"},
		MarshalOnly: true,
	},
	{
		ExpectXML:     `<IfaceInnerXML><T1></T1><hi/><T2></T2></IfaceInnerXML>`,
		Value:         &IfaceInnerXML{InnerXML: nil},
		UnmarshalOnly: true,
	},
	{
		ExpectXML: `<IfaceInnerXML><T1></T1><T2></T2></IfaceInnerXML>`,
		Value:     &IfaceInnerXML{InnerXML: nil},
	},
	{
		ExpectXML:     `<IfaceInnerXML><T1></T1><T2></T2></IfaceInnerXML>`,
		Value:         &IfaceInnerXML{InnerXML: nil},
		UnmarshalOnly: true,
	},
	{
		ExpectXML:   `<DirectInnerXML><T1></T1><hi/><T2></T2></DirectInnerXML>`,
		Value:       &DirectInnerXML{InnerXML: string("<hi/>")},
		MarshalOnly: true,
	},
	{
		ExpectXML:     `<DirectInnerXML><T1></T1><hi/><T2></T2></DirectInnerXML>`,
		Value:         &DirectInnerXML{InnerXML: string("<T1></T1><hi/><T2></T2>")},
		UnmarshalOnly: true,
	},
	{
		ExpectXML:   `<DirectInnerXML><T1></T1><T2></T2></DirectInnerXML>`,
		Value:       &DirectInnerXML{InnerXML: string("")},
		MarshalOnly: true,
	},
	{
		ExpectXML:     `<DirectInnerXML><T1></T1><T2></T2></DirectInnerXML>`,
		Value:         &DirectInnerXML{InnerXML: string("<T1></T1><T2></T2>")},
		UnmarshalOnly: true,
	},
	{
		ExpectXML: `<IndirElement><T1></T1><Element>hi</Element><T2></T2></IndirElement>`,
		Value:     &IndirElement{Element: stringptr("hi")},
	},
	{
		ExpectXML: `<IndirElement><T1></T1><Element></Element><T2></T2></IndirElement>`,
		Value:     &IndirElement{Element: stringptr("")},
	},
	{
		ExpectXML: `<IndirElement><T1></T1><T2></T2></IndirElement>`,
		Value:     &IndirElement{Element: nil},
	},
	{
		ExpectXML:   `<IfaceElement><T1></T1><Element>hi</Element><T2></T2></IfaceElement>`,
		Value:       &IfaceElement{Element: "hi"},
		MarshalOnly: true,
	},
	{
		ExpectXML:     `<IfaceElement><T1></T1><Element>hi</Element><T2></T2></IfaceElement>`,
		Value:         &IfaceElement{Element: nil},
		UnmarshalOnly: true,
	},
	{
		ExpectXML: `<IfaceElement><T1></T1><T2></T2></IfaceElement>`,
		Value:     &IfaceElement{Element: nil},
	},
	{
		ExpectXML:     `<IfaceElement><T1></T1><T2></T2></IfaceElement>`,
		Value:         &IfaceElement{Element: nil},
		UnmarshalOnly: true,
	},
	{
		ExpectXML: `<DirectElement><T1></T1><Element>hi</Element><T2></T2></DirectElement>`,
		Value:     &DirectElement{Element: string("hi")},
	},
	{
		ExpectXML: `<DirectElement><T1></T1><Element></Element><T2></T2></DirectElement>`,
		Value:     &DirectElement{Element: string("")},
	},
	{
		ExpectXML: `<IndirOmitEmpty><T1></T1><OmitEmpty>hi</OmitEmpty><T2></T2></IndirOmitEmpty>`,
		Value:     &IndirOmitEmpty{OmitEmpty: stringptr("hi")},
	},
	{
		// Note: Changed in Go 1.8 to include <OmitEmpty> element (because x.OmitEmpty != nil).
		ExpectXML:   `<IndirOmitEmpty><T1></T1><OmitEmpty></OmitEmpty><T2></T2></IndirOmitEmpty>`,
		Value:       &IndirOmitEmpty{OmitEmpty: stringptr("")},
		MarshalOnly: true,
	},
	{
		ExpectXML:     `<IndirOmitEmpty><T1></T1><OmitEmpty></OmitEmpty><T2></T2></IndirOmitEmpty>`,
		Value:         &IndirOmitEmpty{OmitEmpty: stringptr("")},
		UnmarshalOnly: true,
	},
	{
		ExpectXML: `<IndirOmitEmpty><T1></T1><T2></T2></IndirOmitEmpty>`,
		Value:     &IndirOmitEmpty{OmitEmpty: nil},
	},
	{
		ExpectXML:   `<IfaceOmitEmpty><T1></T1><OmitEmpty>hi</OmitEmpty><T2></T2></IfaceOmitEmpty>`,
		Value:       &IfaceOmitEmpty{OmitEmpty: "hi"},
		MarshalOnly: true,
	},
	{
		ExpectXML:     `<IfaceOmitEmpty><T1></T1><OmitEmpty>hi</OmitEmpty><T2></T2></IfaceOmitEmpty>`,
		Value:         &IfaceOmitEmpty{OmitEmpty: nil},
		UnmarshalOnly: true,
	},
	{
		ExpectXML: `<IfaceOmitEmpty><T1></T1><T2></T2></IfaceOmitEmpty>`,
		Value:     &IfaceOmitEmpty{OmitEmpty: nil},
	},
	{
		ExpectXML:     `<IfaceOmitEmpty><T1></T1><T2></T2></IfaceOmitEmpty>`,
		Value:         &IfaceOmitEmpty{OmitEmpty: nil},
		UnmarshalOnly: true,
	},
	{
		ExpectXML: `<DirectOmitEmpty><T1></T1><OmitEmpty>hi</OmitEmpty><T2></T2></DirectOmitEmpty>`,
		Value:     &DirectOmitEmpty{OmitEmpty: string("hi")},
	},
	{
		ExpectXML: `<DirectOmitEmpty><T1></T1><T2></T2></DirectOmitEmpty>`,
		Value:     &DirectOmitEmpty{OmitEmpty: string("")},
	},
	{
		ExpectXML: `<IndirAny><T1></T1><Any>hi</Any><T2></T2></IndirAny>`,
		Value:     &IndirAny{Any: stringptr("hi")},
	},
	{
		ExpectXML: `<IndirAny><T1></T1><Any></Any><T2></T2></IndirAny>`,
		Value:     &IndirAny{Any: stringptr("")},
	},
	{
		ExpectXML: `<IndirAny><T1></T1><T2></T2></IndirAny>`,
		Value:     &IndirAny{Any: nil},
	},
	{
		ExpectXML:   `<IfaceAny><T1></T1><Any>hi</Any><T2></T2></IfaceAny>`,
		Value:       &IfaceAny{Any: "hi"},
		MarshalOnly: true,
	},
	{
		ExpectXML:     `<IfaceAny><T1></T1><Any>hi</Any><T2></T2></IfaceAny>`,
		Value:         &IfaceAny{Any: nil},
		UnmarshalOnly: true,
	},
	{
		ExpectXML: `<IfaceAny><T1></T1><T2></T2></IfaceAny>`,
		Value:     &IfaceAny{Any: nil},
	},
	{
		ExpectXML:     `<IfaceAny><T1></T1><T2></T2></IfaceAny>`,
		Value:         &IfaceAny{Any: nil},
		UnmarshalOnly: true,
	},
	{
		ExpectXML: `<DirectAny><T1></T1><Any>hi</Any><T2></T2></DirectAny>`,
		Value:     &DirectAny{Any: string("hi")},
	},
	{
		ExpectXML: `<DirectAny><T1></T1><Any></Any><T2></T2></DirectAny>`,
		Value:     &DirectAny{Any: string("")},
	},
	{
		ExpectXML:     `<IndirFoo><T1></T1><Foo>hi</Foo><T2></T2></IndirFoo>`,
		Value:         &IndirAny{Any: stringptr("hi")},
		UnmarshalOnly: true,
	},
	{
		ExpectXML:     `<IndirFoo><T1></T1><Foo></Foo><T2></T2></IndirFoo>`,
		Value:         &IndirAny{Any: stringptr("")},
		UnmarshalOnly: true,
	},
	{
		ExpectXML:     `<IndirFoo><T1></T1><T2></T2></IndirFoo>`,
		Value:         &IndirAny{Any: nil},
		UnmarshalOnly: true,
	},
	{
		ExpectXML:     `<IfaceFoo><T1></T1><Foo>hi</Foo><T2></T2></IfaceFoo>`,
		Value:         &IfaceAny{Any: nil},
		UnmarshalOnly: true,
	},
	{
		ExpectXML:     `<IfaceFoo><T1></T1><T2></T2></IfaceFoo>`,
		Value:         &IfaceAny{Any: nil},
		UnmarshalOnly: true,
	},
	{
		ExpectXML:     `<IfaceFoo><T1></T1><T2></T2></IfaceFoo>`,
		Value:         &IfaceAny{Any: nil},
		UnmarshalOnly: true,
	},
	{
		ExpectXML:     `<DirectFoo><T1></T1><Foo>hi</Foo><T2></T2></DirectFoo>`,
		Value:         &DirectAny{Any: string("hi")},
		UnmarshalOnly: true,
	},
	{
		ExpectXML:     `<DirectFoo><T1></T1><Foo></Foo><T2></T2></DirectFoo>`,
		Value:         &DirectAny{Any: string("")},
		UnmarshalOnly: true,
	},

	// Test namespace prefixes
	{
		ExpectXML: `<epp xmlns="urn:ietf:params:xml:ns:epp-1.0"></epp>`,
		Value:     &EPP{},
	},
	{
		ExpectXML: `<epp xmlns="urn:ietf:params:xml:ns:epp-1.0"><command></command></epp>`,
		Value:     &EPP{Command: &Command{}},
	},
	{
		ExpectXML: `<epp xmlns="urn:ietf:params:xml:ns:epp-1.0"><command><check></check></command></epp>`,
		Value:     &EPP{Command: &Command{Check: &Check{}}},
	},
	{
		ExpectXML: `<epp xmlns="urn:ietf:params:xml:ns:epp-1.0"><command><check><domain:check xmlns:domain="urn:ietf:params:xml:ns:domain-1.0"></domain:check></check></command></epp>`,
		Value:     &EPP{Command: &Command{Check: &Check{DomainCheck: &DomainCheck{}}}},
	},
	{
		ExpectXML: `<epp xmlns="urn:ietf:params:xml:ns:epp-1.0"><command><check><domain:check xmlns:domain="urn:ietf:params:xml:ns:domain-1.0"><domain:name>golang.org</domain:name></domain:check></check></command></epp>`,
		Value:     &EPP{Command: &Command{Check: &Check{DomainCheck: &DomainCheck{DomainNames: []string{"golang.org"}}}}},
	},
	{
		ExpectXML: `<epp xmlns="urn:ietf:params:xml:ns:epp-1.0"><command><check><domain:check xmlns:domain="urn:ietf:params:xml:ns:domain-1.0"><domain:name>golang.org</domain:name><domain:name>go.dev</domain:name></domain:check></check></command></epp>`,
		Value:     &EPP{Command: &Command{Check: &Check{DomainCheck: &DomainCheck{DomainNames: []string{"golang.org", "go.dev"}}}}},
	},
	{
		ExpectXML: `<sec:envelope xmlns:sec="urn:test:secure-1.0"></sec:envelope>`,
		Value:     &SecureEnvelope{},
	},
	{
		ExpectXML: `<sec:envelope xmlns:sec="urn:test:secure-1.0"><msg xmlns="urn:test:message-1.0"></msg></sec:envelope>`,
		Value:     &SecureEnvelope{Message: &SecureMessage{}},
	},
	{
		ExpectXML: `<sec:envelope xmlns:sec="urn:test:secure-1.0"><msg xmlns="urn:test:message-1.0"><body>Hello, world.</body></msg></sec:envelope>`,
		Value:     &SecureEnvelope{Message: &SecureMessage{Body: "Hello, world."}},
	},
	{
		ExpectXML: `<sec:envelope xmlns:sec="urn:test:secure-1.0"><msg xmlns="urn:test:message-1.0" sec:signer="gopher@golang.org"><body>Thanks</body></msg></sec:envelope>`,
		Value:     &SecureEnvelope{Message: &SecureMessage{Body: "Thanks", Signer: "gopher@golang.org"}},
	},
	{
		ExpectXML: `<nested xmlns="urn:test:nested-1.0"><wrapper><nested:value xmlns:nested="urn:test:nested-1.0">You’re welcome!</nested:value></wrapper></nested>`,
		Value:     &NamespacedNested{Value: "You’re welcome!"},
	},
	{
		ExpectXML: `<space:name><space:value>value</space:value></space:name>`,
		Value: &struct {
			XMLName struct{} `xml:"space:name"`
			Value   string   `xml:"space:value"`
		}{Value: "value"},
	},
}

func TestMarshal(t *testing.T) {
	for idx, test := range marshalTests {
		if test.UnmarshalOnly {
			continue
		}

		t.Run(fmt.Sprintf("%d", idx), func(t *testing.T) {
			data, err := Marshal(test.Value)
			if err != nil {
				if test.MarshalError == "" {
					t.Errorf("marshal(%#v): %s", test.Value, err)
					return
				}
				if !strings.Contains(err.Error(), test.MarshalError) {
					t.Errorf("marshal(%#v): %s, want %q", test.Value, err, test.MarshalError)
				}
				return
			}
			if test.MarshalError != "" {
				t.Errorf("Marshal succeeded, want error %q", test.MarshalError)
				return
			}
			if got, want := string(data), test.ExpectXML; got != want {
				if strings.Contains(want, "\n") {
					t.Errorf("marshal(%#v):\nHAVE:\n%s\nWANT:\n%s", test.Value, got, want)
				} else {
					t.Errorf("marshal(%#v):\nhave %#q\nwant %#q", test.Value, got, want)
				}
			}
		})
	}
}

type AttrParent struct {
	X string `xml:"X>Y,attr"`
}

type BadAttr struct {
	Name map[string]string `xml:"name,attr"`
}

var marshalErrorTests = []struct {
	Value any
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
		Value: BadAttr{map[string]string{"X": "Y"}},
		Err:   `xml: unsupported type: map[string]string`,
	},
}

var marshalIndentTests = []struct {
	Value     any
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
		ExpectXML: "<agent handle=\"007\">\n\t<Identity>James Bond</Identity><redacted/>\n</agent>",
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
		if test.ExpectXML == `<top>`+
			`<x><b xmlns="space">b</b>`+
			`<b xmlns="space1">b1</b></x>`+
			`</top>` {
			// TODO(rogpeppe): re-enable this test in
			// https://go-review.googlesource.com/#/c/5910/
			continue
		}

		vt := reflect.TypeOf(test.Value)
		dest := reflect.New(vt.Elem()).Interface()
		err := Unmarshal([]byte(test.ExpectXML), dest)

		t.Run(fmt.Sprintf("%d", i), func(t *testing.T) {
			switch fix := dest.(type) {
			case *Feed:
				fix.Author.InnerXML = ""
				for i := range fix.Entry {
					fix.Entry[i].Author.InnerXML = ""
				}
			}

			if err != nil {
				if test.UnmarshalError == "" {
					t.Errorf("unmarshal(%s): %s", test.ExpectXML, err)
					return
				}
				if !strings.Contains(err.Error(), test.UnmarshalError) {
					t.Errorf("unmarshal(%s): %s, want %q", test.ExpectXML, err, test.UnmarshalError)
				}
				return
			}
			if got, want := dest, test.Value; !reflect.DeepEqual(got, want) {
				t.Errorf("unmarshal(%s):\nhave %#v\nwant %#v", test.ExpectXML, got, want)
			}
		})
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
	var buf strings.Builder
	enc := NewEncoder(&buf)
	if err := enc.EncodeToken(CharData("hello world")); err != nil {
		t.Fatalf("enc.EncodeToken: %v", err)
	}
	if buf.Len() > 0 {
		t.Fatalf("enc.EncodeToken caused actual write: %q", buf.String())
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
	b.RunParallel(func(pb *testing.PB) {
		for pb.Next() {
			Marshal(atomValue)
		}
	})
}

func BenchmarkUnmarshal(b *testing.B) {
	b.ReportAllocs()
	xml := []byte(atomXML)
	b.RunParallel(func(pb *testing.PB) {
		for pb.Next() {
			Unmarshal(xml, &Feed{})
		}
	})
}

// golang.org/issue/6556
func TestStructPointerMarshal(t *testing.T) {
	type A struct {
		XMLName string `xml:"a"`
		B       []any
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
	desc string
	toks []Token
	want string
	err  string
}{{
	desc: "start element with namespace",
	toks: []Token{
		StartElement{Name{"space", "local"}, nil},
	},
	want: `<local xmlns="space">`,
}, {
	desc: "start element with no name",
	toks: []Token{
		StartElement{Name{"space", ""}, nil},
	},
	err: "xml: start tag with no name",
}, {
	desc: "end element with no name",
	toks: []Token{
		EndElement{Name{"space", ""}},
	},
	err: "xml: end tag with no name",
}, {
	desc: "char data",
	toks: []Token{
		CharData("foo"),
	},
	want: `foo`,
}, {
	desc: "char data with escaped chars",
	toks: []Token{
		CharData(" \t\n"),
	},
	want: " &#x9;\n",
}, {
	desc: "comment",
	toks: []Token{
		Comment("foo"),
	},
	want: `<!--foo-->`,
}, {
	desc: "comment with invalid content",
	toks: []Token{
		Comment("foo-->"),
	},
	err: "xml: EncodeToken of Comment containing --> marker",
}, {
	desc: "proc instruction",
	toks: []Token{
		ProcInst{"Target", []byte("Instruction")},
	},
	want: `<?Target Instruction?>`,
}, {
	desc: "proc instruction with empty target",
	toks: []Token{
		ProcInst{"", []byte("Instruction")},
	},
	err: "xml: EncodeToken of ProcInst with invalid Target",
}, {
	desc: "proc instruction with bad content",
	toks: []Token{
		ProcInst{"", []byte("Instruction?>")},
	},
	err: "xml: EncodeToken of ProcInst with invalid Target",
}, {
	desc: "directive",
	toks: []Token{
		Directive("foo"),
	},
	want: `<!foo>`,
}, {
	desc: "more complex directive",
	toks: []Token{
		Directive("DOCTYPE doc [ <!ELEMENT doc '>'> <!-- com>ment --> ]"),
	},
	want: `<!DOCTYPE doc [ <!ELEMENT doc '>'> <!-- com>ment --> ]>`,
}, {
	desc: "directive instruction with bad name",
	toks: []Token{
		Directive("foo>"),
	},
	err: "xml: EncodeToken of Directive containing wrong < or > markers",
}, {
	desc: "end tag without start tag",
	toks: []Token{
		EndElement{Name{"foo", "bar"}},
	},
	err: "xml: end tag </bar> without start tag",
}, {
	desc: "mismatching end tag local name",
	toks: []Token{
		StartElement{Name{"", "foo"}, nil},
		EndElement{Name{"", "bar"}},
	},
	err:  "xml: end tag </bar> does not match start tag <foo>",
	want: `<foo>`,
}, {
	desc: "mismatching end tag namespace",
	toks: []Token{
		StartElement{Name{"space", "foo"}, nil},
		EndElement{Name{"another", "foo"}},
	},
	err:  `xml: end namespace "another" does not match start namespace "space"`,
	want: `<foo xmlns="space">`,
}, {
	desc: "start element with explicit namespace",
	toks: []Token{
		StartElement{Name{"space", "local"}, []Attr{
			{Name{"xmlns", "x"}, "space"},
			{Name{"space", "foo"}, "value"},
		}},
	},
	want: `<local xmlns="space" xmlns:_xmlns="xmlns" _xmlns:x="space" xmlns:space="space" space:foo="value">`,
}, {
	desc: "start element with explicit namespace prefix",
	toks: []Token{
		StartElement{Name{"space", "local"}, []Attr{
			{Name{"http://www.w3.org/2000/xmlns/", "x"}, "space"},
			{Name{"space", "foo"}, "value"},
		}},
	},
	want: `<x:local xmlns:x="space" x:foo="value">`,
}, {
	desc: "start element with prefixed local name",
	toks: []Token{
		StartElement{Name{"space", "x:local"}, []Attr{
			{Name{"space", "foo"}, "value"},
		}},
	},
	want: `<x:local xmlns:x="space" x:foo="value">`,
}, {
	desc: "start element with explicit namespace and colliding prefix",
	toks: []Token{
		StartElement{Name{"space", "local"}, []Attr{
			{Name{"xmlns", "x"}, "space"},
			{Name{"space", "foo"}, "value"},
			{Name{"x", "bar"}, "other"},
		}},
	},
	want: `<local xmlns="space" xmlns:_xmlns="xmlns" _xmlns:x="space" xmlns:space="space" space:foo="value" xmlns:x="x" x:bar="other">`,
}, {
	desc: "start element with explicit namespace prefix and colliding prefix",
	toks: []Token{
		StartElement{Name{"space", "local"}, []Attr{
			{Name{"http://www.w3.org/2000/xmlns/", "x"}, "space"},
			{Name{"space", "foo"}, "value"},
			{Name{"x", "bar"}, "other"},
		}},
	},
	want: `<x:local xmlns:x="space" x:foo="value" xmlns:x_1="x" x_1:bar="other">`,
}, {
	desc: "start element with prefixed local name and colliding prefix",
	toks: []Token{
		StartElement{Name{"space", "x:local"}, []Attr{
			{Name{"space", "foo"}, "value"},
			{Name{"x", "bar"}, "other"},
		}},
	},
	want: `<x:local xmlns:x="space" x:foo="value" xmlns:x_1="x" x_1:bar="other">`,
}, {
	desc: "start element using previously defined namespace",
	toks: []Token{
		StartElement{Name{"", "local"}, []Attr{
			{Name{"xmlns", "x"}, "space"},
		}},
		StartElement{Name{"space", "foo"}, []Attr{
			{Name{"space", "x"}, "y"},
		}},
	},
	want: `<local xmlns:_xmlns="xmlns" _xmlns:x="space"><foo xmlns="space" xmlns:space="space" space:x="y">`,
}, {
	desc: "start element using previously defined namespace prefix",
	toks: []Token{
		StartElement{Name{"", "local"}, []Attr{
			{Name{"http://www.w3.org/2000/xmlns/", "x"}, "space"},
		}},
		StartElement{Name{"space", "foo"}, []Attr{
			{Name{"space", "x"}, "y"},
		}},
	},
	want: `<local xmlns:x="space"><x:foo x:x="y">`,
}, {
	desc: "start element using prefixed local name",
	toks: []Token{
		StartElement{Name: Name{"space", "x:local"}},
		StartElement{Name{"space", "foo"}, []Attr{
			{Name{"space", "x"}, "y"},
		}},
	},
	want: `<x:local xmlns:x="space"><x:foo x:x="y">`,
}, {
	desc: "nested namespace with same prefix",
	toks: []Token{
		StartElement{Name{"", "foo"}, []Attr{
			{Name{"xmlns", "x"}, "space1"},
		}},
		StartElement{Name{"", "foo"}, []Attr{
			{Name{"xmlns", "x"}, "space2"},
		}},
		StartElement{Name{"", "foo"}, []Attr{
			{Name{"space1", "a"}, "space1 value"},
			{Name{"space2", "b"}, "space2 value"},
		}},
		EndElement{Name{"", "foo"}},
		EndElement{Name{"", "foo"}},
		StartElement{Name{"", "foo"}, []Attr{
			{Name{"space1", "a"}, "space1 value"},
			{Name{"space2", "b"}, "space2 value"},
		}},
	},
	want: `<foo xmlns:_xmlns="xmlns" _xmlns:x="space1"><foo _xmlns:x="space2"><foo xmlns:space1="space1" space1:a="space1 value" xmlns:space2="space2" space2:b="space2 value"></foo></foo><foo xmlns:space1="space1" space1:a="space1 value" xmlns:space2="space2" space2:b="space2 value">`,
}, {
	desc: "nested namespace with same prefix",
	toks: []Token{
		StartElement{Name{"", "foo"}, []Attr{
			{Name{"http://www.w3.org/2000/xmlns/", "x"}, "space1"},
		}},
		StartElement{Name{"", "foo"}, []Attr{
			{Name{"http://www.w3.org/2000/xmlns/", "x"}, "space2"},
		}},
		StartElement{Name{"", "foo"}, []Attr{
			{Name{"space1", "a"}, "space1 value"},
			{Name{"space2", "b"}, "space2 value"},
		}},
		EndElement{Name{"", "foo"}},
		EndElement{Name{"", "foo"}},
		StartElement{Name{"", "foo"}, []Attr{
			{Name{"space1", "a"}, "space1 value"},
			{Name{"space2", "b"}, "space2 value"},
		}},
	},
	want: `<foo xmlns:x="space1"><foo xmlns:x="space2"><foo xmlns:space1="space1" space1:a="space1 value" x:b="space2 value"></foo></foo><foo x:a="space1 value" xmlns:space2="space2" space2:b="space2 value">`,
}, {
	desc: "start element defining several prefixes for the same namespace",
	toks: []Token{
		StartElement{Name{"space", "foo"}, []Attr{
			{Name{"xmlns", "a"}, "space"},
			{Name{"xmlns", "b"}, "space"},
			{Name{"space", "x"}, "value"},
		}},
	},
	want: `<foo xmlns="space" xmlns:_xmlns="xmlns" _xmlns:a="space" _xmlns:b="space" xmlns:space="space" space:x="value">`,
}, {
	desc: "start element explicitly defining several prefixes for the same namespace",
	toks: []Token{
		StartElement{Name{"space", "foo"}, []Attr{
			{Name{"http://www.w3.org/2000/xmlns/", "a"}, "space"},
			{Name{"http://www.w3.org/2000/xmlns/", "b"}, "space"},
			{Name{"space", "x"}, "value"},
		}},
	},
	want: `<a:foo xmlns:a="space" xmlns:b="space" a:x="value">`,
}, {
	desc: "nested element redefines namespace",
	toks: []Token{
		StartElement{Name{"", "foo"}, []Attr{
			{Name{"xmlns", "x"}, "space"},
		}},
		StartElement{Name{"space", "foo"}, []Attr{
			{Name{"xmlns", "y"}, "space"},
			{Name{"space", "a"}, "value"},
		}},
	},
	want: `<foo xmlns:_xmlns="xmlns" _xmlns:x="space"><foo xmlns="space" _xmlns:y="space" xmlns:space="space" space:a="value">`,
}, {
	desc: "nested element redefines namespace prefix",
	toks: []Token{
		StartElement{Name{"", "foo"}, []Attr{
			{Name{"http://www.w3.org/2000/xmlns/", "x"}, "space"},
		}},
		StartElement{Name{"space", "foo"}, []Attr{
			{Name{"http://www.w3.org/2000/xmlns/", "y"}, "space"},
			{Name{"space", "a"}, "value"},
		}},
	},
	want: `<foo xmlns:x="space"><y:foo xmlns:y="space" y:a="value">`,
}, {
	desc: "nested element explicitly redefines namespace prefix",
	toks: []Token{
		StartElement{Name{"", "foo"}, []Attr{
			{Name{"http://www.w3.org/2000/xmlns/", "x"}, "space"},
		}},
		StartElement{Name{"space", "y:foo"}, []Attr{
			{Name{"space", "a"}, "value"},
		}},
	},
	want: `<foo xmlns:x="space"><y:foo xmlns:y="space" y:a="value">`,
}, {
	desc: "nested element creates alias for default namespace",
	toks: []Token{
		StartElement{Name{"space", "foo"}, []Attr{
			{Name{"", "xmlns"}, "space"},
		}},
		StartElement{Name{"space", "foo"}, []Attr{
			{Name{"xmlns", "y"}, "space"},
			{Name{"space", "a"}, "value"},
		}},
	},
	want: `<foo xmlns="space"><foo xmlns:_xmlns="xmlns" _xmlns:y="space" xmlns:space="space" space:a="value">`,
}, {
	desc: "nested element creates alias prefix for default namespace",
	toks: []Token{
		StartElement{Name{"space", "foo"}, []Attr{
			{Name{"", "xmlns"}, "space"},
		}},
		StartElement{Name{"space", "foo"}, []Attr{
			{Name{"http://www.w3.org/2000/xmlns/", "y"}, "space"},
			{Name{"space", "a"}, "value"},
		}},
	},
	want: `<foo xmlns="space"><foo xmlns:y="space" y:a="value">`,
}, {
	desc: "nested element explicitly creates alias prefix for default namespace",
	toks: []Token{
		StartElement{Name{"space", "foo"}, []Attr{
			{Name{"", "xmlns"}, "space"},
		}},
		StartElement{Name{"space", "y:foo"}, []Attr{
			{Name{"space", "a"}, "value"},
		}},
	},
	want: `<foo xmlns="space"><y:foo xmlns:y="space" y:a="value">`,
}, {
	desc: "nested element defines default namespace with existing prefix",
	toks: []Token{
		StartElement{Name{"", "foo"}, []Attr{
			{Name{"xmlns", "x"}, "space"},
		}},
		StartElement{Name{"space", "foo"}, []Attr{
			{Name{"", "xmlns"}, "space"},
			{Name{"space", "a"}, "value"},
		}},
	},
	want: `<foo xmlns:_xmlns="xmlns" _xmlns:x="space"><foo xmlns="space" xmlns:space="space" space:a="value">`,
}, {
	desc: "nested element defines default namespace prefix with existing prefix",
	toks: []Token{
		StartElement{Name{"", "foo"}, []Attr{
			{Name{"http://www.w3.org/2000/xmlns/", "x"}, "space"},
		}},
		StartElement{Name{"space", "foo"}, []Attr{
			{Name{"", "xmlns"}, "space"},
			{Name{"space", "a"}, "value"},
		}},
	},
	want: `<foo xmlns:x="space"><foo xmlns="space" x:a="value">`,
}, {
	desc: "nested element defines explicit attribute namespace prefix with existing prefix",
	toks: []Token{
		StartElement{Name: Name{"", "foo"}},
		StartElement{Name{"space", "foo"}, []Attr{
			{Name{"", "xmlns"}, "space"},
			{Name{"space", "x:a"}, "value"},
		}},
	},
	want: `<foo><foo xmlns="space" xmlns:x="space" x:a="value">`,
}, {
	desc: "nested element uses empty attribute namespace when default ns defined",
	toks: []Token{
		StartElement{Name{"space", "foo"}, []Attr{
			{Name{"", "xmlns"}, "space"},
		}},
		StartElement{Name{"space", "foo"}, []Attr{
			{Name{"", "attr"}, "value"},
		}},
	},
	want: `<foo xmlns="space"><foo attr="value">`,
}, {
	desc: "redefine xmlns",
	toks: []Token{
		StartElement{Name{"", "foo"}, []Attr{
			{Name{"foo", "xmlns"}, "space"},
		}},
	},
	want: `<foo xmlns:foo="foo" foo:xmlns="space">`,
}, {
	desc: "xmlns with explicit namespace #1",
	toks: []Token{
		StartElement{Name{"space", "foo"}, []Attr{
			{Name{"xml", "xmlns"}, "space"},
		}},
	},
	want: `<foo xmlns="space" xmlns:_xml="xml" _xml:xmlns="space">`,
}, {
	desc: "xmlns with explicit namespace #2",
	toks: []Token{
		StartElement{Name{"space", "foo"}, []Attr{
			{Name{xmlURL, "xmlns"}, "space"},
		}},
	},
	want: `<foo xmlns="space" xml:xmlns="space">`,
}, {
	desc: "empty namespace declaration is ignored",
	toks: []Token{
		StartElement{Name{"", "foo"}, []Attr{
			{Name{"xmlns", "foo"}, ""},
		}},
	},
	want: `<foo xmlns:_xmlns="xmlns" _xmlns:foo="">`,
}, {
	desc: "empty namespace prefix declaration is ignored",
	toks: []Token{
		StartElement{Name{"", "foo"}, []Attr{
			{Name{"http://www.w3.org/2000/xmlns/", "foo"}, ""},
		}},
	},
	want: `<foo xmlns:foo="">`,
}, {
	desc: "attribute with no name is ignored",
	toks: []Token{
		StartElement{Name{"", "foo"}, []Attr{
			{Name{"", ""}, "value"},
		}},
	},
	want: `<foo>`,
}, {
	desc: "namespace URL with non-valid name",
	toks: []Token{
		StartElement{Name{"/34", "foo"}, []Attr{
			{Name{"/34", "x"}, "value"},
		}},
	},
	want: `<foo xmlns="/34" xmlns:_="/34" _:x="value">`,
}, {
	desc: "nested element resets default namespace to empty",
	toks: []Token{
		StartElement{Name{"space", "foo"}, []Attr{
			{Name{"", "xmlns"}, "space"},
		}},
		StartElement{Name{"", "foo"}, []Attr{
			{Name{"", "xmlns"}, ""},
			{Name{"", "x"}, "value"},
			{Name{"space", "x"}, "value"},
		}},
	},
	want: `<foo xmlns="space"><foo xmlns="" x="value" xmlns:space="space" space:x="value">`,
}, {
	desc: "nested element requires empty default namespace",
	toks: []Token{
		StartElement{Name{"space", "foo"}, []Attr{
			{Name{"", "xmlns"}, "space"},
		}},
		StartElement{Name{"", "foo"}, nil},
	},
	want: `<foo xmlns="space"><foo>`,
}, {
	desc: "attribute uses namespace from xmlns",
	toks: []Token{
		StartElement{Name{"some/space", "foo"}, []Attr{
			{Name{"", "attr"}, "value"},
			{Name{"some/space", "other"}, "other value"},
		}},
	},
	want: `<foo xmlns="some/space" attr="value" xmlns:space="some/space" space:other="other value">`,
}, {
	desc: "default namespace should not be used by attributes",
	toks: []Token{
		StartElement{Name{"space", "foo"}, []Attr{
			{Name{"", "xmlns"}, "space"},
			{Name{"xmlns", "bar"}, "space"},
			{Name{"space", "baz"}, "foo"},
		}},
		StartElement{Name{"space", "baz"}, nil},
		EndElement{Name{"space", "baz"}},
		EndElement{Name{"space", "foo"}},
	},
	want: `<foo xmlns="space" xmlns:_xmlns="xmlns" _xmlns:bar="space" xmlns:space="space" space:baz="foo"><baz></baz></foo>`,
}, {
	desc: "default namespace prefix should not be used by attributes",
	toks: []Token{
		StartElement{Name{"space", "foo"}, []Attr{
			{Name{"", "xmlns"}, "space"},
			{Name{"http://www.w3.org/2000/xmlns/", "bar"}, "space"},
			{Name{"space", "baz"}, "foo"},
		}},
		StartElement{Name{"space", "baz"}, nil},
		EndElement{Name{"space", "baz"}},
		EndElement{Name{"space", "foo"}},
	},
	want: `<foo xmlns="space" xmlns:bar="space" bar:baz="foo"><baz></baz></foo>`,
}, {
	desc: "default namespace not used by attributes, not explicitly defined",
	toks: []Token{
		StartElement{Name{"space", "foo"}, []Attr{
			{Name{"", "xmlns"}, "space"},
			{Name{"space", "baz"}, "foo"},
		}},
		StartElement{Name{"space", "baz"}, nil},
		EndElement{Name{"space", "baz"}},
		EndElement{Name{"space", "foo"}},
	},
	want: `<foo xmlns="space" xmlns:space="space" space:baz="foo"><baz></baz></foo>`,
}, {
	desc: "impossible xmlns declaration",
	toks: []Token{
		StartElement{Name{"", "foo"}, []Attr{
			{Name{"", "xmlns"}, "space"},
		}},
		StartElement{Name{"space", "bar"}, []Attr{
			{Name{"space", "attr"}, "value"},
		}},
	},
	want: `<foo xmlns="space"><bar xmlns="space" xmlns:space="space" space:attr="value">`,
}, {
	desc: "reserved namespace prefix -- all lower case",
	toks: []Token{
		StartElement{Name{"", "foo"}, []Attr{
			{Name{"http://www.w3.org/2001/xmlSchema-instance", "nil"}, "true"},
		}},
	},
	want: `<foo xmlns:_xmlSchema-instance="http://www.w3.org/2001/xmlSchema-instance" _xmlSchema-instance:nil="true">`,
}, {
	desc: "reserved namespace prefix -- all upper case",
	toks: []Token{
		StartElement{Name{"", "foo"}, []Attr{
			{Name{"http://www.w3.org/2001/XMLSchema-instance", "nil"}, "true"},
		}},
	},
	want: `<foo xmlns:_XMLSchema-instance="http://www.w3.org/2001/XMLSchema-instance" _XMLSchema-instance:nil="true">`,
}, {
	desc: "reserved namespace prefix -- all mixed case",
	toks: []Token{
		StartElement{Name{"", "foo"}, []Attr{
			{Name{"http://www.w3.org/2001/XmLSchema-instance", "nil"}, "true"},
		}},
	},
	want: `<foo xmlns:_XmLSchema-instance="http://www.w3.org/2001/XmLSchema-instance" _XmLSchema-instance:nil="true">`,
}}

func TestEncodeToken(t *testing.T) {
loop:
	for i, tt := range encodeTokenTests {
		var buf strings.Builder
		enc := NewEncoder(&buf)
		var err error
		for j, tok := range tt.toks {
			err = enc.EncodeToken(tok)
			if err != nil && j < len(tt.toks)-1 {
				t.Errorf("#%d %s; token #%d: %v", i, tt.desc, j, err)
				continue loop
			}
		}
		errorf := func(f string, a ...any) {
			t.Errorf("#%d %s token #%d:%s", i, tt.desc, len(tt.toks)-1, fmt.Sprintf(f, a...))
		}
		switch {
		case tt.err != "" && err == nil:
			errorf(" expected error; got none")
			continue
		case tt.err == "" && err != nil:
			errorf(" got error: %v", err)
			continue
		case tt.err != "" && err != nil && tt.err != err.Error():
			errorf(" error mismatch; got %v, want %v", err, tt.err)
			continue
		}
		if err := enc.Flush(); err != nil {
			errorf(" %v", err)
			continue
		}
		if got := buf.String(); got != tt.want {
			errorf("\ngot  %v\nwant %v", got, tt.want)
			continue
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

// Issue 9796. Used to fail with GORACE="halt_on_error=1" -race.
func TestRace9796(t *testing.T) {
	type A struct{}
	type B struct {
		C []A `xml:"X>Y"`
	}
	var wg sync.WaitGroup
	for i := 0; i < 2; i++ {
		wg.Add(1)
		go func() {
			Marshal(B{[]A{{}}})
			wg.Done()
		}()
	}
	wg.Wait()
}

func TestIsValidDirective(t *testing.T) {
	testOK := []string{
		"<>",
		"< < > >",
		"<!DOCTYPE '<' '>' '>' <!--nothing-->>",
		"<!DOCTYPE doc [ <!ELEMENT doc ANY> <!ELEMENT doc ANY> ]>",
		"<!DOCTYPE doc [ <!ELEMENT doc \"ANY> '<' <!E\" LEMENT '>' doc ANY> ]>",
		"<!DOCTYPE doc <!-- just>>>> a < comment --> [ <!ITEM anything> ] >",
	}
	testKO := []string{
		"<",
		">",
		"<!--",
		"-->",
		"< > > < < >",
		"<!dummy <!-- > -->",
		"<!DOCTYPE doc '>",
		"<!DOCTYPE doc '>'",
		"<!DOCTYPE doc <!--comment>",
	}
	for _, s := range testOK {
		if !isValidDirective(Directive(s)) {
			t.Errorf("Directive %q is expected to be valid", s)
		}
	}
	for _, s := range testKO {
		if isValidDirective(Directive(s)) {
			t.Errorf("Directive %q is expected to be invalid", s)
		}
	}
}

// Issue 11719. EncodeToken used to silently eat tokens with an invalid type.
func TestSimpleUseOfEncodeToken(t *testing.T) {
	var buf strings.Builder
	enc := NewEncoder(&buf)
	if err := enc.EncodeToken(&StartElement{Name: Name{"", "object1"}}); err == nil {
		t.Errorf("enc.EncodeToken: pointer type should be rejected")
	}
	if err := enc.EncodeToken(&EndElement{Name: Name{"", "object1"}}); err == nil {
		t.Errorf("enc.EncodeToken: pointer type should be rejected")
	}
	if err := enc.EncodeToken(StartElement{Name: Name{"", "object2"}}); err != nil {
		t.Errorf("enc.EncodeToken: StartElement %s", err)
	}
	if err := enc.EncodeToken(EndElement{Name: Name{"", "object2"}}); err != nil {
		t.Errorf("enc.EncodeToken: EndElement %s", err)
	}
	if err := enc.EncodeToken(Universe{}); err == nil {
		t.Errorf("enc.EncodeToken: invalid type not caught")
	}
	if err := enc.Flush(); err != nil {
		t.Errorf("enc.Flush: %s", err)
	}
	if buf.Len() == 0 {
		t.Errorf("enc.EncodeToken: empty buffer")
	}
	want := "<object2></object2>"
	if buf.String() != want {
		t.Errorf("enc.EncodeToken: expected %q; got %q", want, buf.String())
	}
}

// Issue 16158. Decoder.unmarshalAttr ignores the return value of copyValue.
func TestIssue16158(t *testing.T) {
	const data = `<foo b="HELLOWORLD"></foo>`
	err := Unmarshal([]byte(data), &struct {
		B byte `xml:"b,attr,omitempty"`
	}{})
	if err == nil {
		t.Errorf("Unmarshal: expected error, got nil")
	}
}

// Issue 20953. Crash on invalid XMLName attribute.

type InvalidXMLName struct {
	XMLName Name `xml:"error"`
	Type    struct {
		XMLName Name `xml:"type,attr"`
	}
}

func TestInvalidXMLName(t *testing.T) {
	var buf bytes.Buffer
	enc := NewEncoder(&buf)
	if err := enc.Encode(InvalidXMLName{}); err == nil {
		t.Error("unexpected success")
	} else if want := "invalid tag"; !strings.Contains(err.Error(), want) {
		t.Errorf("error %q does not contain %q", err, want)
	}
}

// Issue 50164. Crash on zero value XML attribute.
type LayerOne struct {
	XMLName Name `xml:"l1"`

	Value     *float64 `xml:"value,omitempty"`
	*LayerTwo `xml:",omitempty"`
}

type LayerTwo struct {
	ValueTwo *int `xml:"value_two,attr,omitempty"`
}

func TestMarshalZeroValue(t *testing.T) {
	proofXml := `<l1><value>1.2345</value></l1>`
	var l1 LayerOne
	err := Unmarshal([]byte(proofXml), &l1)
	if err != nil {
		t.Fatalf("unmarshal XML error: %v", err)
	}
	want := float64(1.2345)
	got := *l1.Value
	if got != want {
		t.Fatalf("unexpected unmarshal result, want %f but got %f", want, got)
	}

	// Marshal again (or Encode again)
	// In issue 50164, here `Marshal(l1)` will panic because of the zero value of xml attribute ValueTwo `value_two`.
	anotherXML, err := Marshal(l1)
	if err != nil {
		t.Fatalf("marshal XML error: %v", err)
	}
	if string(anotherXML) != proofXml {
		t.Fatalf("unexpected unmarshal result, want %q but got %q", proofXml, anotherXML)
	}
}

var closeTests = []struct {
	desc string
	toks []Token
	want string
	err  string
}{{
	desc: "unclosed start element",
	toks: []Token{
		StartElement{Name{"", "foo"}, nil},
	},
	want: `<foo>`,
	err:  "unclosed tag <foo>",
}, {
	desc: "closed element",
	toks: []Token{
		StartElement{Name{"", "foo"}, nil},
		EndElement{Name{"", "foo"}},
	},
	want: `<foo></foo>`,
}, {
	desc: "directive",
	toks: []Token{
		Directive("foo"),
	},
	want: `<!foo>`,
}}

func TestClose(t *testing.T) {
	for _, tt := range closeTests {
		tt := tt
		t.Run(tt.desc, func(t *testing.T) {
			var out strings.Builder
			enc := NewEncoder(&out)
			for j, tok := range tt.toks {
				if err := enc.EncodeToken(tok); err != nil {
					t.Fatalf("token #%d: %v", j, err)
				}
			}
			err := enc.Close()
			switch {
			case tt.err != "" && err == nil:
				t.Error(" expected error; got none")
			case tt.err == "" && err != nil:
				t.Errorf(" got error: %v", err)
			case tt.err != "" && err != nil && tt.err != err.Error():
				t.Errorf(" error mismatch; got %v, want %v", err, tt.err)
			}
			if got := out.String(); got != tt.want {
				t.Errorf("\ngot  %v\nwant %v", got, tt.want)
			}
			t.Log(enc.p.closed)
			if err := enc.EncodeToken(Directive("foo")); err == nil {
				t.Errorf("unexpected success when encoding after Close")
			}
		})
	}
}
