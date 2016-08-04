// Copyright 2016 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

module.exports = config => {
    config.set({
        frameworks: ['jasmine'],
        basePath: '../../../..',
        files: [
            'third_party/webcomponents/customelements.js',
            'third_party/typescript/typescript.js',
            'third_party/moduleloader/moduleloader.js',
            'cmd/heapview/client/testing/test_main.js',
            {pattern: 'cmd/heapview/client/**/*.ts', included: false},
        ],
        browsers: ['Chrome'],
        plugins: [
            'karma-jasmine',
            'karma-chrome-launcher'
        ],
    })
}