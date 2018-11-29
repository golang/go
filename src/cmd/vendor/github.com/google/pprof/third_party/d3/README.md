# Building a customized D3.js bundle

The D3.js version distributed with pprof is customized to only include the modules required by pprof.

## Dependencies

First, it's necessary to pull all bundle dependencies. We will use a JavaScript package manager, [npm](https://www.npmjs.com/), to accomplish that. npm dependencies are declared in a `package.json` file, so create one with the following configuration:

```js
{
  "name": "d3-pprof",
  "version": "1.0.0",
  "description": "A d3.js bundle for pprof.",
  "scripts": {
    "prepare": "rollup -c && uglifyjs d3.js -c -m -o d3.min.js"
  },
  "license": "Apache-2.0",
  "devDependencies": {
    "d3-selection": "1.1.0",
    "d3-hierarchy": "1.1.5",
    "d3-scale": "1.0.6",
    "d3-format": "1.2.0",
    "d3-ease": "1.0.3",
    "d3-array": "1.2.1",
    "d3-collection": "1.0.4",
    "d3-transition": "1.1.0",
    "rollup": "0.51.8",
    "rollup-plugin-node-resolve": "3",
    "uglify-js": "3.1.10"
  }
}
```

Besides the bundle dependencies, the `package.json` file also specifies a script called `prepare`, which will be executed to create the bundle after `Rollup` is installed.

## Bundler

The simplest way of creating a custom bundle is to use a bundler, such as [Rollup](https://rollupjs.org/) or [Webpack](https://webpack.js.org/). Rollup will be used in this example.

First, create a `rollup.config.js` file, containing the configuration Rollup should use to build the bundle.

```js
import node from "rollup-plugin-node-resolve";

export default {
  input: "index.js",
  output: {
    format: "umd",
    file: "d3.js"
  },
  name: "d3",
  plugins: [node()],
  sourcemap: false
};
```

Then create an `index.js` file containing all the functions that need to be exported in the bundle.

```js
export {
  select,
  selection,
  event,
} from "d3-selection";

export {
    hierarchy,
    partition,
} from "d3-hierarchy";

export {
    scaleLinear,
} from "d3-scale";

export {
    format,
} from "d3-format";

export {
    easeCubic,
} from "d3-ease";

export {
    ascending,
} from "d3-array";

export {
    map,
} from "d3-collection";

export {
    transition,
} from "d3-transition";
```

## Building

Once all files were created, execute the following commands to pull all dependencies and build the bundle.

```
% npm install
% npm run prepare
```

This will create two files, `d3.js` and `d3.min.js`, the custom D3.js bundle and its minified version respectively.

# References

## D3 Custom Bundle

A demonstration of building a custom D3 4.0 bundle using ES2015 modules and Rollup. 

[bl.ocks.org/mbostock/bb09af4c39c79cffcde4](https://bl.ocks.org/mbostock/bb09af4c39c79cffcde4)

## d3-pprof

A repository containing all previously mentioned configuration files and the generated custom bundle.

[github.com/spiermar/d3-pprof](https://github.com/spiermar/d3-pprof)