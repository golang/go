// Copyright 2016 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

import {HamburgerElement, HeadingElement, SidebarElement, main} from './main';

describe('main', () => {
  it('sets the document\'s title', () => {
    main();
    expect(document.title).toBe('Go Heap Viewer');
  });

  it('has a heading', () => {
    main();
    expect(document.querySelector(HeadingElement.NAME)).toBeDefined();
  });

  it('has a sidebar', () => {
    main();
    const hamburger = document.querySelector(HamburgerElement.NAME);
    const sidebar =
        document.querySelector(SidebarElement.NAME) as SidebarElement;
    expect(sidebar.style.display).toBe('none');

    // Click on the hamburger. Sidebar should then be visible.
    hamburger.dispatchEvent(new Event('click'));
    expect(sidebar.style.display).toBe('block');
  })
});