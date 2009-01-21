// errchk $G $D/$F.go

// Copyright 2009 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

package main

type T struct {
	x, x int  // this should be a compile-time error
}

/*
Accessing obj.x for obj of type T will lead to an error so this cannot
be used in a program, but I would argue that this should be a compile-
tume error at the declaration point.
*/

/* Condensed e-mail thread:

---------- Russ Cox	
I don't think this is an error as long as you don't refer to x. I like the fact that you could name
multiple elements in the struct "pad".


---------- Rob 'Commander' Pike to Russ, me, go-dev, reviewlog2
the real question is whether this program matches the spec and if not, which is in error.


---------- Russ Cox to Rob, me, go-dev, reviewlog2
true enough.  the spec disagrees with 6g.
when we discussed the disambiguation
rules for anonymous structs i thought we'd
mentioned this issue too and decided the
opposite, but i'm happy to make 6g agree
with the spec.


---------- Robert Griesemer to Russ, Rob, go-dev, reviewlog2
I think the spec could perhaps be more definitive. Note that 6g also accepts:

type T struct {
	x int
}

func (p *T) x() {
}

func (p *T) x() {
}

The spec says that the scope of methods and fields is selectors of the form obj.selector. In a scope an identifier can be declared only once. I'd conclude that in the scope of fields and selectors of T, there are multiple x. But it's somewhat indirect. From a programmer's point of view making this an error seems less surprising, at least to me.


---------- Ken Thompson to me, Russ, Rob, go-dev, reviewlog2
obviously i dont think this is an error, or
i would have made it an error. it seems like
a small point if the error comes up at
declaration or use.


---------- Robert Griesemer to Ken, Russ, Rob, go-dev, reviewlog2
I don't really care too much, but I think it should be consistent. The following code:

type T struct {
	x int;
}

func (p *T) x() {
}

func (p *T) x(a, b, c int) {
}

does result in an error (method redeclared). I don't quite see why this should behave any different then if both x() had the same parameter list.

PS: I agree that the spec is saying two different things here - or at least should be more precise; the section on selectors (which arguably is the newer section), would allow such a declaration. I suspect that the case below leads to an early error possibly due to some interaction with code that checks for forward declaration of functions/methods (this not having looked at 6g).

I am happy to go either way. It's a small item and we can close it once we all agree.
*/
