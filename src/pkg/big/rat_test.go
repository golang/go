// Copyright 2010 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

package big

import "testing"


type setStringTest struct {
	in, out string
}

var setStringTests = []setStringTest{
	setStringTest{"0", "0"},
	setStringTest{"1", "1"},
	setStringTest{"-1", "-1"},
	setStringTest{"2/4", "1/2"},
	setStringTest{".25", "1/4"},
	setStringTest{"-1/5", "-1/5"},
}

func TestRatSetString(t *testing.T) {
	for i, test := range setStringTests {
		x, _ := new(Rat).SetString(test.in)

		if x.String() != test.out {
			t.Errorf("#%d got %s want %s", i, x.String(), test.out)
		}
	}
}


type floatStringTest struct {
	in   string
	prec int
	out  string
}

var floatStringTests = []floatStringTest{
	floatStringTest{"0", 0, "0"},
	floatStringTest{"0", 4, "0"},
	floatStringTest{"1", 0, "1"},
	floatStringTest{"1", 2, "1"},
	floatStringTest{"-1", 0, "-1"},
	floatStringTest{".25", 2, "0.25"},
	floatStringTest{".25", 1, "0.3"},
	floatStringTest{"-1/3", 3, "-0.333"},
	floatStringTest{"-2/3", 4, "-0.6667"},
}

func TestFloatString(t *testing.T) {
	for i, test := range floatStringTests {
		x, _ := new(Rat).SetString(test.in)

		if x.FloatString(test.prec) != test.out {
			t.Errorf("#%d got %s want %s", i, x.FloatString(test.prec), test.out)
		}
	}
}


type ratCmpTest struct {
	rat1, rat2 string
	out        int
}

var ratCmpTests = []ratCmpTest{
	ratCmpTest{"0", "0/1", 0},
	ratCmpTest{"1/1", "1", 0},
	ratCmpTest{"-1", "-2/2", 0},
	ratCmpTest{"1", "0", 1},
	ratCmpTest{"0/1", "1/1", -1},
	ratCmpTest{"-5/1434770811533343057144", "-5/1434770811533343057145", -1},
	ratCmpTest{"49832350382626108453/8964749413", "49832350382626108454/8964749413", -1},
	ratCmpTest{"-37414950961700930/7204075375675961", "37414950961700930/7204075375675961", -1},
	ratCmpTest{"37414950961700930/7204075375675961", "74829901923401860/14408150751351922", 0},
}

func TestRatCmp(t *testing.T) {
	for i, test := range ratCmpTests {
		x, _ := new(Rat).SetString(test.rat1)
		y, _ := new(Rat).SetString(test.rat2)

		out := x.Cmp(y)
		if out != test.out {
			t.Errorf("#%d got out = %v; want %v", i, out, test.out)
		}
	}
}


type ratBinFun func(z, x, y *Rat) *Rat
type ratBinArg struct {
	x   string
	y   string
	out string
}

func testRatBin(t *testing.T, f ratBinFun, a []ratBinArg) {
	for i, test := range a {
		x, _ := NewRat(0, 1).SetString(test.x)
		y, _ := NewRat(0, 1).SetString(test.y)
		expected, _ := NewRat(0, 1).SetString(test.out)
		out := f(NewRat(0, 1), x, y)

		if out.Cmp(expected) != 0 {
			t.Errorf("#%d got %s want %s", i, out, expected)
		}
	}
}


var ratAddTests = []ratBinArg{
	ratBinArg{"0", "0", "0"},
	ratBinArg{"0", "1", "1"},
	ratBinArg{"-1", "0", "-1"},
	ratBinArg{"-1", "1", "0"},
	ratBinArg{"1", "1", "2"},
	ratBinArg{"1/2", "1/2", "1"},
	ratBinArg{"1/4", "1/3", "7/12"},
	ratBinArg{"2/5", "-14/3", "-64/15"},
	ratBinArg{"4707/49292519774798173060", "-3367/70976135186689855734", "84058377121001851123459/1749296273614329067191168098769082663020"},
	ratBinArg{"-61204110018146728334/3", "-31052192278051565633/2", "-215564796870448153567/6"},
}

func TestRatAdd(t *testing.T) {
	testRatBin(t, (*Rat).Add, ratAddTests)
}


var ratSubTests = []ratBinArg{
	ratBinArg{"0", "0", "0"},
	ratBinArg{"0", "1", "-1"},
	ratBinArg{"-1", "0", "-1"},
	ratBinArg{"-1", "1", "-2"},
	ratBinArg{"1", "1", "0"},
	ratBinArg{"1/2", "1/3", "1/6"},
	ratBinArg{"1/4", "1/3", "-1/12"},
	ratBinArg{"2/5", "-14/3", "76/15"},
	ratBinArg{"4707/49292519774798173060", "-3367/70976135186689855734", "250026291202747299816479/1749296273614329067191168098769082663020"},
	ratBinArg{"-27/133467566250814981", "-18/31750379913563777419", "-854857841473707320655/4237645934602118692642972629634714039"},
	ratBinArg{"27674141753240653/30123979153216", "-19948846211000086/637313996471", "618575745270541348005638912139/19198433543745179392300736"},
}

func TestRatSub(t *testing.T) {
	testRatBin(t, (*Rat).Sub, ratSubTests)
}


var ratMulTests = []ratBinArg{
	ratBinArg{"0", "0", "0"},
	ratBinArg{"0", "1", "0"},
	ratBinArg{"-1", "0", "0"},
	ratBinArg{"-1", "1", "-1"},
	ratBinArg{"1", "1", "1"},
	ratBinArg{"1/2", "1/2", "1/4"},
	ratBinArg{"1/4", "1/3", "1/12"},
	ratBinArg{"2/5", "-14/3", "-28/15"},
	ratBinArg{"-3/26206484091896184128", "5/2848423294177090248", "-5/24882386581946146755650075889827061248"},
	ratBinArg{"26946729/330400702820", "41563965/225583428284", "224002580204097/14906584649915733312176"},
	ratBinArg{"-8259900599013409474/7", "-84829337473700364773/56707961321161574960", "350340947706464153265156004876107029701/198477864624065512360"},
}

func TestRatMul(t *testing.T) {
	testRatBin(t, (*Rat).Mul, ratMulTests)
}


var ratQuoTests = []ratBinArg{
	ratBinArg{"0", "1", "0"},
	ratBinArg{"0", "-1", "0"},
	ratBinArg{"-1", "1", "-1"},
	ratBinArg{"1", "1", "1"},
	ratBinArg{"1/2", "1/2", "1"},
	ratBinArg{"1/4", "1/3", "3/4"},
	ratBinArg{"2/5", "-14/3", "-3/35"},
	ratBinArg{"808/45524274987585732633", "29/712593081308", "575775209696864/1320203974639986246357"},
	ratBinArg{"8967230/3296219033", "6269770/1992362624741777", "1786597389946320496771/2066653520653241"},
	ratBinArg{"-3784609207827/3426986245", "9381566963714/9633539", "-36459180403360509753/32150500941194292113930"},
}

func TestRatQuo(t *testing.T) {
	testRatBin(t, (*Rat).Quo, ratQuoTests)
}
