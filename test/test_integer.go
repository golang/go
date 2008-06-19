// Copyright 2009 The Go Authors.  All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

package main

import Integer "integer"

type Int Integer.Integer;

const (
  sa = "991";
  sb = "2432902008176640000";  // 20!
  sc = "93326215443944152681699238856266700490715968264381621468592963895217599993229915608941463976156518286253697920827223758251185210916864000000000000000000000000";  // 100!
)

var (
  m, z, p,
  a, b, c,
  a_a, a_b, a_c, b_b, b_c, c_c, a_b_c
  Int
)


func CHECK(p bool) {
  if !p {
    panic "CHECK failed\n";
  }
}


func Init() {
  m = Integer.FromInt(-1);
  z = Integer.FromInt(0);
  p = Integer.FromInt(1);
  
  a = Integer.FromString(sa);
  b = Integer.FromString(sb);
  c = Integer.FromString(sc);
  
  a_a = Integer.FromInt(991 + 991);
  a_b = Integer.FromString("2432902008176640991");
  a_c = Integer.FromString("93326215443944152681699238856266700490715968264381621468592963895217599993229915608941463976156518286253697920827223758251185210916864000000000000000000000991");
}


func TestConv() {
  print "TestConv\n";
  CHECK(a.eql(Integer.FromInt(991)));
  CHECK(b.eql(Integer.Fact(20)));
  CHECK(c.eql(Integer.Fact(100)));
  
  CHECK(a.ToString() == sa);
  CHECK(b.ToString() == sb);
  CHECK(c.ToString() == sc);
  
  // also tested much via TestFact
}


func TestAdd() {
  print "TestAdd\n";
  CHECK(z.add(z).eql(z));
  CHECK(a.add(z).eql(a));
  CHECK(z.add(a).eql(a));

  CHECK(c.add(z).eql(c));
  CHECK(z.add(c).eql(c));

  CHECK(m.add(p).eql(z));
  
  CHECK(a.add(a).eql(a_a));
  CHECK(a.add(b).eql(a_b));
  CHECK(a.add(c).eql(a_c));
  
  // needs more
}


func TestSub() {
  print "TestSub\n";
  CHECK(z.sub(z).eql(z));
  CHECK(a.sub(z).eql(a));
  CHECK(z.sub(a).eql(a.neg()));

  CHECK(c.sub(z).eql(c));
  CHECK(z.sub(c).eql(c.neg()));
  
  CHECK(p.sub(m).eql(p.add(p)));

  CHECK(a.sub(a).eql(z));
  
  // needs more
}


func TestMul() {
  print "TestMul\n";
  // tested much via TestFact for now
}


func TestDiv() {
  print "TestDiv\n";
  // no div implemented yet
}


func TestMod() {
  print "TestMod\n";
  // no mod implemented yet
}


func TestFact() {
  print "TestFact\n";
  for n := 990; n < 1010; n++ {
    f := Integer.Fact(n);
    CHECK(Integer.FromString(f.ToString()).eql(f));
  }
}


func main() {
  Init();
  
  TestConv();
  TestAdd();
  TestSub();
  TestMul();
  TestDiv();
  TestMod();

  TestFact();
  
  print "PASSED\n";
}
