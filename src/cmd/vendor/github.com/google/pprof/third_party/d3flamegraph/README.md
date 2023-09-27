# Building a customized D3.js bundle

The D3.js version distributed with pprof is customized to only include the
modules required by pprof.

## Dependencies

- Install [npm](https://www.npmjs.com).

## Building

- Run `update.sh` to:
  - Download npm package dependencies (declared in `package.json` and `package-lock.json`)
  - Create a d3.js bundle containing the JavScript of d3 and d3-flame-graph (by running `webpack`)

This will `d3_flame_graph.go`, the minified custom D3.js bundle as Go source code.

# References / Appendix

## D3 Custom Bundle

A demonstration of building a custom D3 4.0 bundle using ES2015 modules and Rollup. 

[bl.ocks.org/mbostock/bb09af4c39c79cffcde4](https://bl.ocks.org/mbostock/bb09af4c39c79cffcde4)

## Old version of d3-pprof

A previous version of d3-flame-graph bundled for pprof used Rollup instead of
Webpack. This has now been migrated directly into this directory.

The repository configuring Rollup was here:

[github.com/spiermar/d3-pprof](https://github.com/spiermar/d3-pprof)
