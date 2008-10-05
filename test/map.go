// $G $F.go && $L $F.$A && ./$A.out

// Copyright 2009 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

package main

import fmt "fmt"

const arraylen = 2; // BUG: shouldn't need this

func P(a *[]string) string {
	s := "{";
	for i := 0; i < len(a); i++ {
		if i > 0 {
			s += ","
		}
		s += `"` + a[i] + `"`;
	}
	s +="}";
	return s;
}

func main() {
	F := fmt.New();

	// BUG: should test a map literal when there's syntax

	mib := new(map[int] bool);
	mii := new(map[int] int);
	mfi := new(map[float] int);
	mif := new(map[int] float);
	msi := new(map[string] int);
	mis := new(map[int] string);
	mss := new(map[string] string);
	mspa := new(map[string] *[]string);
	// BUG need an interface map both ways too

	type T struct {
		s string;
		f float;
	};
	mipT := new(map[int] *T);
	mpTi := new(map[*T] int);
	//mit := new(map[int] T);	// should be able to do a value but:  fatal error: algtype: cant find type <T>{}
	//mti := new(map[T] int);	// should be able to do a value but:  fatal error: algtype: cant find type <T>{}

	type M map[int] int; 
	mipM := new(map[int] *M);

	const count = 100; // BUG: should be bigger but maps do linear lookup
	var apT [2*count]*T;

	for i := 0; i < count; i++ {
		s := F.d(i).str();
		f := float(i);
		apT[i] = new(T);
		apT[i].s = s;
		apT[i].f = f;
		apT[2*i] = new(T);	// need twice as many entries as we use, for the nonexistence check
		apT[2*i].s = s;
		apT[2*i].f = f;
		// BUG t := T(s, f);
		t := new(T); t.s = s; t.f = f;
		// BUG m := M(i, i+1);
		m := new(M); m[i] = i+1;
		mib[i] = (i != 0);
		mii[i] = 10*i;
		mfi[float(i)] = 10*i;
		mif[i] = 10.0*f;
		mis[i] = s;
		msi[F.d(i).str()] = i;
		mss[F.d(i).str()] = F.d(10*i).str();
		mss[F.d(i).str()] = F.d(10*i).str();
		as := new([arraylen]string);
			as[0] = F.d(10*i).str();
			as[1] = F.d(10*i).str();
		mspa[F.d(i).str()] = as;
		mipT[i] = t;
		mpTi[apT[i]] = i;
		// BUG mti[t] = i;
		mipM[i] = m;
	}

	// test len
	if len(mib) != count {
		F.s("len(mib) = ").d(len(mib)).putnl();
	}
	if len(mii) != count {
		F.s("len(mii) = ").d(len(mii)).putnl();
	}
	if len(mfi) != count {
		F.s("len(mfi) = ").d(len(mfi)).putnl();
	}
	if len(mif) != count {
		F.s("len(mif) = ").d(len(mif)).putnl();
	}
	if len(msi) != count {
		F.s("len(msi) = ").d(len(msi)).putnl();
	}
	if len(mis) != count {
		F.s("len(mis) = ").d(len(mis)).putnl();
	}
	if len(mss) != count {
		F.s("len(mss) = ").d(len(mss)).putnl();
	}
	if len(mspa) != count {
		F.s("len(mspa) = ").d(len(mspa)).putnl();
	}
	if len(mipT) != count {
		F.s("len(mipT) = ").d(len(mipT)).putnl();
	}
	if len(mpTi) != count {
		F.s("len(mpTi) = ").d(len(mpTi)).putnl();
	}
//	if len(mti) != count {
//		F.s("len(mti) = ").d(len(mti)).putnl();
//	}
	if len(mipM) != count {
		F.s("len(mipM) = ").d(len(mipM)).putnl();
	}
	
	// test construction directly
	for i := 0; i < count; i++ {
		s := F.d(i).str();
		f := float(i);
		// BUG t := T(s, f);
		var t T; t.s = s; t.f = f;
		// BUG m := M(i, i+1);
		if mib[i] != (i != 0) {
			F.s("mib[").d(i).s("] = ").boolean(mib[i]).putnl();
		}
		if(mii[i] != 10*i) {
			F.s("mii[").d(i).s("] = ").d(mii[i]).putnl();
		}
		if(mfi[f] != 10*i) {
			F.s("mfi[").d(i).s("] = ").d(mfi[f]).putnl();
		}
		if(mif[i] != 10.0*f) {
			F.s("mif[").d(i).s("] = ").g(mif[i]).putnl();
		}
		if(mis[i] != s) {
			F.s("mis[").d(i).s("] = ").s(mis[i]).putnl();
		}
		if(msi[s] != i) {
			F.s("msi[").s(s).s("] = ").d(msi[s]).putnl();
		}
		if mss[s] != F.d(10*i).str() {
			F.s("mss[").s(s).s("] = ").s(mss[s]).putnl();
		}
		for j := 0; j < arraylen; j++ {
			if mspa[s][j] != F.d(10*i).str() {
				F.s("mspa[").s(s).s("][").d(j).s("] = ").s(mspa[s][j]).putnl();
			}
		}
		if(mipT[i].s != s || mipT[i].f != f) {
			F.s("mipT[").d(i).s("] = {").s(mipT[i].s).s(", ").g(mipT[i].f).s("}").putnl();
		}
		if(mpTi[apT[i]] != i) {
			F.s("mpTi[apT[").d(i).s("]] = ").d(mpTi[apT[i]]).putnl();
		}
//		if(mti[t] != i) {
//			F.s("mti[").s(s).s("] = ").s(mti[s]).putnl();
//		}
		if (mipM[i][i] != i + 1) {
			F.s("mipM[").d(i).s("][").d(i).s("] =").d(mipM[i][i]).putnl();
		}
	}

	// test existence with tuple check
	// failed lookups yield a false value for the boolean.
	for i := 0; i < count; i++ {
		s := F.d(i).str();
		f := float(i);
		// BUG t := T(s, f);
		var t T; t.s = s; t.f = f;
		// BUG m := M(i, i+1);
		{
			a, b := mib[i];
			if !b {
				F.s("tuple existence decl: mib[").d(i).s("]").putnl();
			}
			a, b = mib[i];
			if !b {
				F.s("tuple existence assign: mib[").d(i).s("]").putnl();
			}
		}
		{
			a, b := mii[i];
			if !b {
				F.s("tuple existence decl: mii[").d(i).s("]").putnl();
			}
			a, b = mii[i];
			if !b {
				F.s("tuple existence assign: mii[").d(i).s("]").putnl();
			}
		}
		{
			a, b := mfi[f];
			if !b {
				F.s("tuple existence decl: mfi[").g(f).s("]").putnl();
			}
			a, b = mfi[f];
			if !b {
				F.s("tuple existence assign: mfi[").g(f).s("]").putnl();
			}
		}
		{
			a, b := mif[i];
			if !b {
				F.s("tuple existence decl: mif[").d(i).s("]").putnl();
			}
			a, b = mif[i];
			if !b {
				F.s("tuple existence assign: mif[").d(i).s("]").putnl();
			}
		}
		{
			a, b := mis[i];
			if !b {
				F.s("tuple existence decl: mis[").d(i).s("]").putnl();
			}
			a, b = mis[i];
			if !b {
				F.s("tuple existence assign: mis[").d(i).s("]").putnl();
			}
		}
		{
			a, b := msi[s];
			if !b {
				F.s("tuple existence decl: msi[").s(s).s("]").putnl();
			}
			a, b = msi[s];
			if !b {
				F.s("tuple existence assign: msi[").s(s).s("]").putnl();
			}
		}
		{
			a, b := mss[s];
			if !b {
				F.s("tuple existence decl: mss[").s(s).s("]").putnl();
			}
			a, b = mss[s];
			if !b {
				F.s("tuple existence assign: mss[").s(s).s("]").putnl();
			}
		}
		{
			a, b := mspa[s];
			if !b {
				F.s("tuple existence decl: mspa[").s(s).s("]").putnl();
			}
			a, b = mspa[s];
			if !b {
				F.s("tuple existence assign: mspa[").s(s).s("]").putnl();
			}
		}
		{
			a, b := mipT[i];
			if !b {
				F.s("tuple existence decl: mipT[").d(i).s("]").putnl();
			}
			a, b = mipT[i];
			if !b {
				F.s("tuple existence assign: mipT[").d(i).s("]").putnl();
			}
		}
		{
			a, b := mpTi[apT[i]];
			if !b {
				F.s("tuple existence decl: mpTi[apT[").d(i).s("]]").putnl();
			}
			a, b = mpTi[apT[i]];
			if !b {
				F.s("tuple existence assign: mpTi[apT[").d(i).s("]]").putnl();
			}
		}
//		a,b := mti[t]...
//			emit stdout <- format("haskey mti[%s] false", string(t));
		{
			a, b := mipM[i];
			if !b {
				F.s("tuple existence decl: mipM[").d(i).s("]").putnl();
			}
			a, b = mipM[i];
			if !b {
				F.s("tuple existence assign: mipM[").d(i).s("]").putnl();
			}
		}
	}

	// test nonexistence with tuple check
	// failed lookups yield a false value for the boolean.
	for i := count; i < 2*count; i++ {
		s := F.d(i).str();
		f := float(i);
		// BUG t := T(s, f);
		var t T; t.s = s; t.f = f;
		// BUG m := M(i, i+1);
		{
			a, b := mib[i];
			if b {
				F.s("tuple nonexistence decl: mib[").d(i).s("]").putnl();
			}
			a, b = mib[i];
			if b {
				F.s("tuple nonexistence assign: mib[").d(i).s("]").putnl();
			}
		}
		{
			a, b := mii[i];
			if b {
				F.s("tuple nonexistence decl: mii[").d(i).s("]").putnl();
			}
			a, b = mii[i];
			if b {
				F.s("tuple nonexistence assign: mii[").d(i).s("]").putnl();
			}
		}
		{
			a, b := mfi[f];
			if b {
				F.s("tuple nonexistence decl: mfi[").g(f).s("]").putnl();
			}
			a, b = mfi[f];
			if b {
				F.s("tuple nonexistence assign: mfi[").g(f).s("]").putnl();
			}
		}
		{
			a, b := mif[i];
			if b {
				F.s("tuple nonexistence decl: mif[").d(i).s("]").putnl();
			}
			a, b = mif[i];
			if b {
				F.s("tuple nonexistence assign: mif[").d(i).s("]").putnl();
			}
		}
		{
			a, b := mis[i];
			if b {
				F.s("tuple nonexistence decl: mis[").d(i).s("]").putnl();
			}
			a, b = mis[i];
			if b {
				F.s("tuple nonexistence assign: mis[").d(i).s("]").putnl();
			}
		}
		{
			a, b := msi[s];
			if b {
				F.s("tuple nonexistence decl: msi[").s(s).s("]").putnl();
			}
			a, b = msi[s];
			if b {
				F.s("tuple nonexistence assign: msi[").s(s).s("]").putnl();
			}
		}
		{
			a, b := mss[s];
			if b {
				F.s("tuple nonexistence decl: mss[").s(s).s("]").putnl();
			}
			a, b = mss[s];
			if b {
				F.s("tuple nonexistence assign: mss[").s(s).s("]").putnl();
			}
		}
		{
			a, b := mspa[s];
			if b {
				F.s("tuple nonexistence decl: mspa[").s(s).s("]").putnl();
			}
			a, b = mspa[s];
			if b {
				F.s("tuple nonexistence assign: mspa[").s(s).s("]").putnl();
			}
		}
		{
			a, b := mipT[i];
			if b {
				F.s("tuple nonexistence decl: mipT[").d(i).s("]").putnl();
			}
			a, b = mipT[i];
			if b {
				F.s("tuple nonexistence assign: mipT[").d(i).s("]").putnl();
			}
		}
		{
			a, b := mpTi[apT[i]];
			if b {
				F.s("tuple nonexistence decl: mpTi[apt[").d(i).s("]]").putnl();
			}
			a, b = mpTi[apT[i]];
			if b {
				F.s("tuple nonexistence assign: mpTi[apT[").d(i).s("]]").putnl();
			}
		}
//		a,b := mti[t]...
//			emit stdout <- format("haskey mti[%s] false", string(t));
		{
			a, b := mipM[i];
			if b {
				F.s("tuple nonexistence decl: mipM[").d(i).s("]").putnl();
			}
			a, b = mipM[i];
			if b {
				F.s("tuple nonexistence assign: mipM[").d(i).s("]").putnl();
			}
		}
	}
	

	// tests for structured map element updates
	for i := 0; i < count; i++ {
		s := F.d(i).str();
		mspa[s][i % 2] = "deleted";
		if mspa[s][i % 2] != "deleted" {
			F.s("mspa[").d(i).s("][").d(i).s("%2] =").s(mspa[s][i % 2]).putnl();
		}
		mipT[i].s = string('a' + i % 26) + mipT[i].s[1:len(s)];
		first := string('a' + i % 26);
		if mipT[i].s != first + s[1:len(s)] {
			F.s("mit[").d(i).s("].s = ").s(mipT[i].s).putnl();
		}
		mipT[i].f = float(i + 1);
		if (mipT[i].f != float(i + 1)) {
			F.s("mipT[").d(i).s("].f = ").g(mipT[i].f).putnl();
		}
		mipM[i][i]++;
		if mipM[i][i] != (i + 1) + 1 {
			F.s("mipM[").d(i).s("][").d(i).s("] = ").d(mipM[i][i]).putnl();
		}
	}
}
