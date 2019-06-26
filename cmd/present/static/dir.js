// Copyright 2012 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

// copied from $GOROOT/doc/godocs.js

function bindEvent(el, e, fn) {
  if (el.addEventListener) {
    el.addEventListener(e, fn, false);
  } else if (el.attachEvent) {
    el.attachEvent('on' + e, fn);
  }
}

function godocs_bindSearchEvents() {
  var search = document.getElementById('search');
  if (!search) {
    // no search box (index disabled)
    return;
  }
  function clearInactive() {
    if (search.className == 'inactive') {
      search.value = '';
      search.className = '';
    }
  }
  function restoreInactive() {
    if (search.value !== '') {
      return;
    }
    if (search.type != 'search') {
      search.value = search.getAttribute('placeholder');
    }
    search.className = 'inactive';
  }
  restoreInactive();
  bindEvent(search, 'focus', clearInactive);
  bindEvent(search, 'blur', restoreInactive);
}

bindEvent(window, 'load', godocs_bindSearchEvents);
