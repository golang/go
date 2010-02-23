// Copyright 2009 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

package main

import (
	"fmt"
	"os"
	"xgb"
)

func main() {
	c, err := xgb.Dial(os.Getenv("DISPLAY"))
	if err != nil {
		fmt.Printf("cannot connect: %v\n", err)
		os.Exit(1)
	}

	fmt.Printf("vendor = '%s'\n", string(c.Setup.Vendor))

	win := c.NewId()
	gc := c.NewId()

	c.CreateWindow(0, win, c.DefaultScreen().Root, 150, 150, 200, 200, 0, 0, 0, 0, nil)
	c.ChangeWindowAttributes(win, xgb.CWEventMask,
		[]uint32{xgb.EventMaskExposure | xgb.EventMaskKeyRelease})
	c.CreateGC(gc, win, 0, nil)
	c.MapWindow(win)

	atom, _ := c.InternAtom(0, "HELLO")
	fmt.Printf("atom = %d\n", atom.Atom)

	points := make([]xgb.Point, 2)
	points[0] = xgb.Point{5, 5}
	points[1] = xgb.Point{100, 120}

	hosts, _ := c.ListHosts()
	fmt.Printf("hosts = %+v\n", hosts)

	ecookie := c.ListExtensionsRequest()
	exts, _ := c.ListExtensionsReply(ecookie)
	for _, name := range exts.Names {
		fmt.Printf("exts = '%s'\n", name.Name)
	}

	for {
		reply, err := c.WaitForEvent()
		if err != nil {
			fmt.Printf("error: %v\n", err)
			os.Exit(1)
		}
		fmt.Printf("event %T\n", reply)
		switch event := reply.(type) {
		case xgb.ExposeEvent:
			c.PolyLine(xgb.CoordModeOrigin, win, gc, points)
		case xgb.KeyReleaseEvent:
			fmt.Printf("key release!\n")
			points[0].X = event.EventX
			points[0].Y = event.EventY
			c.PolyLine(xgb.CoordModeOrigin, win, gc, points)
			c.Bell(75)
		}
	}

	c.Close()
}
