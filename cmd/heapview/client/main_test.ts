// Copyright 2016 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

import {main} from './main';

describe('main', () => {
  it('sets the document\'s title', () => {
    main();
    expect(document.title).toBe('Go Heap Viewer');
  });

  it('has a heading', () => {
    main();
    expect(document.querySelector('heap-heading')).toBeDefined();
  });
});