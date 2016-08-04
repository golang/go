// Copyright 2016 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

// Configure module loader.
System.transpiler = 'typescript'
System.typescriptOptions = {
  target: ts.ScriptTarget.ES2015
};
System.locate = (load) => load.name + '.ts';

// Determine set of test files.
var tests = [];
for (var file in window.__karma__.files) {
  if (window.__karma__.files.hasOwnProperty(file)) {
    if (/_test\.ts$/.test(file)) {
      tests.push(file.slice(0, -3));
    }
  }
}

// Steal loaded callback so we can block until we're
// done loading all test modules.
var loadedCallback = window.__karma__.loaded.bind(window.__karma__);
window.__karma__.loaded = () => {};

// Load all test modules, and then call loadedCallback.
var promises = [];
for (var i = 0; i < tests.length; i++) {
  promises.push(System.import(tests[i]));
}
Promise.all(promises).then(loadedCallback);